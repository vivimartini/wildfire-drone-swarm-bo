"""Randers-Finsler geometry for the wind-driven fire front.

Wind-driven fire spread is the canonical worked example of a *Randers metric*: a
Riemannian part (the no-wind rate of spread) plus a one-form (the wind drift).
Richards' elliptical growth model -- the operational standard in fire modelling --
is a Randers metric in another notation, and fire arrival time is the Finslerian
distance from the ignition set.

This module

1. fits a Randers metric to the cellular automaton's *own* spread law
   (:func:`randers_from_env`), so the anisotropy the surrogate sees is a
   consequence of the simulator's physics rather than a hand-built frame;
2. solves for the arrival-time field as a Finsler distance from the fire
   perimeter (:func:`arrival_time`);
3. turns that field into a **deterministic feature map** (:class:`FinslerWarp`)
   that a stationary Matern kernel can legitimately consume.

Why warp, and not put a Finsler distance inside the kernel
----------------------------------------------------------
Finsler distance is asymmetric: upwind travel is slower than downwind, so
``d_F(x, y) != d_F(y, x)``. A covariance function has to be symmetric and
positive semi-definite, and a directed distance is neither. Substituting it into
a Matern kernel yields a matrix that is not even symmetric, let alone PSD --
:func:`directed_distance_matrix` exists so that this can be demonstrated rather
than asserted.

Symmetrising (mean or min of the two directions) restores symmetry, still does
not restore positive definiteness in general, and throws away precisely the
asymmetry that made the geometry worth having.

The fix used here is to **warp rather than substitute**: map each point through
a fixed, deterministic function of the arrival-time field, then use an ordinary
stationary kernel in that warped space. Positive definiteness comes for free --
``k(phi(x), phi(y))`` is PSD for any deterministic ``phi`` whenever ``k`` is --
and the asymmetry survives, because the forward and reverse arrival times enter
the feature vector as two separate coordinates.

Scope note
----------
The metric is fitted to the CA's *mean-field* directional rate. A stochastic
percolation front does not advance at exactly the mean-field rate, so
:mod:`fire_model.finsler_validation` regresses simulated arrival times onto the
geodesic prediction and reports the calibration factor and the fit quality
instead of assuming they agree.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field as _dc_field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, Matern

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The metric itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandersField:
    """A per-cell Randers metric stored in Zermelo (speed, drift) form.

    The fire is modelled as a front that spreads isotropically at ``speed``
    m/s and is simultaneously advected by ``drift`` m/s. The reachable-in-unit-
    time velocity set is therefore the circle of radius ``speed`` displaced by
    ``drift`` -- an off-centre ellipse, which is exactly the indicatrix of a
    Randers metric and exactly Richards' elliptical growth model.

    Strong convexity requires a subunit drift, ``|drift| < speed``; the fit in
    :func:`randers_from_env` enforces it and records how often it had to bite.

    Attributes
    ----------
    speed : (nx, ny) float array
        Riemannian part: no-drift rate of spread, m/s.
    drift : (nx, ny, 2) float array
        One-form part: wind advection of the front, m/s.
    dx_m : float
        Cell size in metres.
    burnable : (nx, ny) bool array
        Cells the front can enter at all (positive fuel and positive speed).
    fit_residual : (nx, ny) float array
        Relative L2 misfit of the Randers profile against the CA's directional
        rate law, per cell.
    clipped_fraction : float
        Fraction of burnable cells whose drift had to be shrunk to keep the
        metric strongly convex.
    """

    speed: np.ndarray
    drift: np.ndarray
    dx_m: float
    burnable: np.ndarray
    fit_residual: np.ndarray = _dc_field(default_factory=lambda: np.zeros(0))
    clipped_fraction: float = 0.0

    def __post_init__(self) -> None:
        speed = np.asarray(self.speed, dtype=float)
        drift = np.asarray(self.drift, dtype=float)
        if speed.ndim != 2:
            raise ValueError(f"speed must be (nx, ny); got {speed.shape}")
        if drift.shape != speed.shape + (2,):
            raise ValueError(f"drift must be {speed.shape + (2,)}; got {drift.shape}")
        if np.asarray(self.burnable).shape != speed.shape:
            raise ValueError("burnable must match the grid shape")

    @property
    def grid_size(self) -> tuple[int, int]:
        return (int(self.speed.shape[0]), int(self.speed.shape[1]))

    # -- the metric in its two equivalent forms ---------------------------

    def directional_speed(self, u: np.ndarray) -> np.ndarray:
        """Rate of spread in unit direction(s) ``u``.

        This is the radial function of the indicatrix -- the circle of radius
        ``speed`` centred on ``drift`` -- obtained by solving
        ``|sigma u - W| = s`` for ``sigma``::

            sigma(u) = W.u + sqrt(s^2 - |W|^2 + (W.u)^2)

        It is emphatically **not** ``s + W.u``. A Randers metric has
        ``F(v) = sqrt(a(v,v)) + b.v``, so its radial function is
        ``1 / (sqrt(a(u,u)) + b.u)`` -- the *reciprocal* of a linear form. The
        linear expression is only the first-order expansion in ``|W|/s`` and
        overstates the flank rate by 67% at ``|W|/s = 0.8``.

        Parameters
        ----------
        u : (2,) or (k, 2) array of unit vectors.

        Returns
        -------
        (nx, ny) or (k, nx, ny) array of m/s.
        """
        u = np.atleast_2d(np.asarray(u, dtype=float))
        u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), _EPS)
        along = np.einsum("kd,xyd->kxy", u, self.drift)
        disc = self.speed[None, ...] ** 2 - np.sum(self.drift ** 2, axis=-1)[None, ...] + along ** 2
        sigma = along + np.sqrt(np.maximum(disc, 0.0))
        sigma = np.where(self.burnable[None, ...], sigma, 0.0)
        return sigma[0] if sigma.shape[0] == 1 else sigma

    def wind_ratio(self) -> np.ndarray:
        """``|drift| / speed`` -- the Randers ``||b||_a``; must stay below 1."""
        mag = np.linalg.norm(self.drift, axis=-1)
        out = np.zeros_like(mag)
        np.divide(mag, self.speed, out=out, where=self.speed > _EPS)
        return out

    def is_strongly_convex(self, tol: float = 1e-9) -> bool:
        """True when every burnable cell has a subunit drift."""
        return bool(np.all(self.wind_ratio()[self.burnable] < 1.0 - tol))

    def randers_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the Randers data ``(a_ij, b_i)`` with ``F = sqrt(a v v) + b v``.

        Uses the Bao-Robles-Shen solution of the Zermelo navigation problem for
        the Riemannian metric ``h_ij = delta_ij / speed^2`` (unit-``h`` velocity
        == spreading at ``speed``) under wind ``W = drift``::

            lambda = 1 - h(W, W)
            a_ij   = (lambda * h_ij + W_i W_j) / lambda^2
            b_i    = -W_i / lambda,          W_i = h_ij W^j

        Returns
        -------
        a : (nx, ny, 2, 2) array
        b : (nx, ny, 2) array
            Both are filled with ``nan`` on non-burnable cells.
        """
        nx, ny = self.grid_size
        s2 = np.where(self.burnable, self.speed ** 2, np.nan)
        h = np.zeros((nx, ny, 2, 2))
        h[..., 0, 0] = 1.0 / s2
        h[..., 1, 1] = 1.0 / s2

        W_low = self.drift / s2[..., None]                      # W_i = h_ij W^j
        hWW = np.einsum("xyd,xyd->xy", W_low, self.drift)       # h(W, W)
        lam = 1.0 - hWW

        a = (lam[..., None, None] * h + W_low[..., :, None] * W_low[..., None, :]) / (lam ** 2)[..., None, None]
        b = -W_low / lam[..., None]
        return a, b

    def finsler_norm(self, v: np.ndarray, ix: int, iy: int) -> float:
        """Evaluate ``F(x, v)`` at cell ``(ix, iy)`` -- the travel time for the
        displacement ``v`` (metres) under a locally frozen metric."""
        a, b = self.randers_tensors()
        v = np.asarray(v, dtype=float)
        quad = float(v @ a[ix, iy] @ v)
        return float(np.sqrt(max(quad, 0.0)) + b[ix, iy] @ v)

    def describe(self) -> dict:
        """Human-readable summary of the fitted metric."""
        ratio = self.wind_ratio()[self.burnable]
        speeds = self.speed[self.burnable]
        return {
            "grid_size": self.grid_size,
            "dx_m": float(self.dx_m),
            "speed_mps": {
                "min": float(speeds.min()) if speeds.size else 0.0,
                "median": float(np.median(speeds)) if speeds.size else 0.0,
                "max": float(speeds.max()) if speeds.size else 0.0,
            },
            "wind_ratio_b_norm": {
                "median": float(np.median(ratio)) if ratio.size else 0.0,
                "max": float(np.max(ratio, initial=0.0)),
            },
            "strongly_convex": self.is_strongly_convex(),
            "drift_clipped_fraction": float(self.clipped_fraction),
            "fit_residual_median": (
                float(np.median(self.fit_residual[self.burnable]))
                if self.fit_residual.size and self.burnable.any()
                else float("nan")
            ),
        }


def uniform_directions(n_dirs: int) -> np.ndarray:
    """``n_dirs`` unit vectors evenly spaced on the circle, shape ``(n_dirs, 2)``."""
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_dirs), endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def zermelo_speed(s: np.ndarray, W: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Radial function of the indicatrix: the circle of radius ``s`` centred at ``W``.

    ``sigma(u) = W.u + sqrt(s^2 - |W|^2 + (W.u)^2)``, the rate of spread in
    direction ``u`` for a front that spreads isotropically at ``s`` while being
    advected at ``W``. Returns ``(n_dirs, ...)``.
    """
    along = np.einsum("kd,...d->k...", dirs, W)
    disc = (s ** 2 - np.sum(W ** 2, axis=-1))[None, ...] + along ** 2
    return along + np.sqrt(np.maximum(disc, 0.0))


def fit_randers_profile(
    sigma: np.ndarray,
    dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a directional rate-of-spread profile onto the Randers family.

    The target is the *indicatrix* radial function
    ``sigma(u) = W.u + sqrt(s^2 - |W|^2 + (W.u)^2)``, not the linear form
    ``s + W.u`` -- see :meth:`RandersField.directional_speed` for why those are
    different curves.

    Although that target is nonlinear in ``(s, W)``, squaring it rearranges to

        sigma(u)^2 = 2 sigma(u) (W.u) + q,      q := s^2 - |W|^2,

    which is *linear* in the unknowns ``(W, q)``. So the fit is an ordinary
    least-squares solve of a 3x3 normal system per cell -- no iteration, and
    exact when the profile is genuinely Randers. Strong convexity is simply
    ``q > 0``.

    Parameters
    ----------
    sigma : (n_dirs, ...) array
        Rate of spread in each direction, m/s.
    dirs : (n_dirs, 2) array
        The corresponding unit direction vectors.

    Returns
    -------
    s : (...) array          Riemannian part (no-drift spread rate), m/s
    W : (..., 2) array       drift, m/s
    rel_residual : (...) array
        Relative L2 misfit, measured in rate-of-spread units against the fitted
        indicatrix itself rather than against the linearised surrogate.
    """
    sigma = np.asarray(sigma, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    if sigma.shape[0] != dirs.shape[0]:
        raise ValueError("sigma and dirs must agree on the direction axis")

    tail = sigma.shape[1:]
    flat = sigma.reshape(sigma.shape[0], -1)                     # (K, M)
    n_dirs, n_cells = flat.shape

    # design row per direction: [2 sigma u_x, 2 sigma u_y, 1] ; target sigma^2
    design = np.empty((n_dirs, n_cells, 3))
    design[..., 0] = 2.0 * flat * dirs[:, 0:1]
    design[..., 1] = 2.0 * flat * dirs[:, 1:2]
    design[..., 2] = 1.0
    target = flat ** 2

    normal = np.einsum("kmi,kmj->mij", design, design)
    rhs = np.einsum("kmi,km->mi", design, target)
    # Ridge term keeps degenerate cells (zero speed, no fuel) solvable.
    normal[:, [0, 1, 2], [0, 1, 2]] += 1e-12
    params = np.linalg.solve(normal, rhs[..., None])[..., 0]     # (M, 3)

    W = params[:, :2]
    q = params[:, 2]
    s = np.sqrt(np.maximum(q + np.sum(W ** 2, axis=-1), 0.0))

    pred = zermelo_speed(s, W, dirs)                             # (K, M)
    denom = np.linalg.norm(flat, axis=0)
    rel = np.zeros(n_cells)
    np.divide(np.linalg.norm(flat - pred, axis=0), denom, out=rel, where=denom > _EPS)

    return s.reshape(tail), W.reshape(tail + (2,)), rel.reshape(tail)


def ca_directional_speed(
    env,
    dirs: np.ndarray,
    *,
    ros_mps: float | np.ndarray | None = None,
    wind_coeff: float | None = None,
    t_index: int = 0,
) -> np.ndarray:
    """Mean-field rate of spread of :class:`~fire_model.ca.CAFireModel`, per direction.

    Mirrors the per-direction rate used inside ``CAFireModel.step_batch``. There,
    a neighbour at lattice distance ``d`` ignites with hazard
    ``lambda = (ros / dx) / d * fuel * bias``, so the expected crossing time is
    ``d * dx / (ros * fuel * bias)`` and the implied front speed in direction
    ``u`` is

        sigma(x, u) = ros(x) * fuel(x) * (1 + wind_coeff * max(0, w(x) . u)) * slope_factor(x, u)

    independent of the lattice spacing -- which is what makes it a metric.

    Returns
    -------
    (n_dirs, nx, ny) array of m/s.
    """
    dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), _EPS)

    fuel = np.asarray(env.fuel, dtype=float)
    nx, ny = fuel.shape

    ros = np.asarray(env.ros_mps if ros_mps is None else ros_mps, dtype=float)
    if ros.ndim == 0:
        ros_field = np.full((nx, ny), float(ros))
    elif ros.ndim == 1:                                   # (T,)
        ros_field = np.full((nx, ny), float(ros[int(np.clip(t_index, 0, ros.size - 1))]))
    elif ros.ndim == 2:                                   # (nx, ny)
        ros_field = ros
    elif ros.ndim == 3:                                   # (T, nx, ny)
        ros_field = ros[int(np.clip(t_index, 0, ros.shape[0] - 1))]
    else:
        raise ValueError("ros_mps must be scalar, (T,), (nx,ny) or (T,nx,ny)")

    wind = np.asarray(env.wind, dtype=float)
    if wind.ndim == 4:                                    # (T, nx, ny, 2)
        wind = wind[int(np.clip(t_index, 0, wind.shape[0] - 1))]
    c = float(env.wind_coeff if wind_coeff is None else wind_coeff)

    align = np.einsum("kd,xyd->kxy", dirs, wind)
    response = str(getattr(env, "wind_response", "clipped")).lower().strip()
    if response not in {"clipped", "elliptical"}:
        raise ValueError("wind_response must be 'clipped' or 'elliptical'")
    if response == "clipped":
        bias = 1.0 + c * np.maximum(0.0, align)
    else:
        drift_along = c * align
        cw2 = np.sum((c * wind) ** 2, axis=-1)[None, ...]
        bias = drift_along + np.sqrt(np.maximum(1.0 - cw2 + drift_along ** 2, 1e-6))
    bias = np.maximum(bias, 1e-3)

    if getattr(env, "slope", None) is not None:
        slope = np.asarray(env.slope, dtype=float)
        grade = np.einsum("kd,xyd->kxy", dirs, slope)
        theta = np.clip(np.arctan(grade) * 180.0 / np.pi, -30.0, 30.0)
        bias = bias * np.power(2.0, theta / 10.0)

    return (ros_field * fuel)[None, ...] * bias


def randers_from_env(
    env,
    *,
    ros_mps: float | np.ndarray | None = None,
    wind_coeff: float | None = None,
    t_index: int = 0,
    n_dirs: int = 64,
    max_wind_ratio: float = 0.95,
) -> RandersField:
    """Fit the Randers metric implied by a :class:`~fire_model.ca.FireEnv`.

    The CA's directional rate law is sampled on ``n_dirs`` directions and
    projected onto the Randers family. With ``FireEnv.wind_response="elliptical"``,
    sub-critical wind (``wind_coeff * |w| < 1``) and no slope the projection is
    *exact*: the CA's law is then the radial function of a circle of radius
    ``ros * fuel`` centred on the drift ``ros * fuel * wind_coeff * w``, which is
    a Randers indicatrix, so the metric is a restatement of the simulator rather
    than an approximation of it.

    With the default ``wind_response="clipped"`` the CA takes ``max(0, w . u)``,
    so it backs into the wind at its full no-wind rate. That profile is flat
    across the entire upwind half-plane and no ellipse can reproduce it; the
    misfit grows with ``wind_coeff * |w|``. The per-cell relative residual is
    recorded on the returned field so the size of that approximation stays
    visible, and :mod:`fire_model.finsler_validation` quantifies where it starts
    to matter.

    ``max_wind_ratio`` shrinks the drift where the fitted ``|W| / s`` would reach
    1, which is the condition for the metric to stay strongly convex (and for
    arrival time to stay a well-posed shortest-path problem).
    """
    dirs = uniform_directions(n_dirs)
    sigma = ca_directional_speed(env, dirs, ros_mps=ros_mps, wind_coeff=wind_coeff, t_index=t_index)
    speed, drift, residual = fit_randers_profile(sigma, dirs)

    burnable = (np.asarray(env.fuel, dtype=float) > 0.0) & (speed > _EPS)
    speed = np.where(burnable, speed, 0.0)
    drift = np.where(burnable[..., None], drift, 0.0)

    ratio = np.zeros_like(speed)
    np.divide(np.linalg.norm(drift, axis=-1), speed, out=ratio, where=speed > _EPS)
    clipped = burnable & (ratio > max_wind_ratio)
    if clipped.any():
        shrink = np.ones_like(ratio)
        np.divide(max_wind_ratio, ratio, out=shrink, where=clipped)
        drift = drift * np.where(clipped, shrink, 1.0)[..., None]

    nx, _ = env.grid_size
    dx_m = float(env.domain_km) * 1000.0 / float(nx)
    denom = max(int(burnable.sum()), 1)
    return RandersField(
        speed=speed,
        drift=drift,
        dx_m=dx_m,
        burnable=burnable,
        fit_residual=residual,
        clipped_fraction=float(clipped.sum()) / denom,
    )


# ---------------------------------------------------------------------------
# Arrival time as a Finsler distance
# ---------------------------------------------------------------------------


def neighbour_offsets(neighbourhood: int = 16) -> np.ndarray:
    """Lattice offsets used by the shortest-path solver.

    The 16-neighbour stencil adds the knight moves to the Moore neighbourhood,
    which cuts the lattice's own directional bias (a shortest lattice path can
    only turn in a finite set of directions) from a few percent to well under
    one, at four times the edge count.
    """
    moore = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    if neighbourhood == 4:
        return np.array(moore[:4], dtype=int)
    if neighbourhood == 8:
        return np.array(moore, dtype=int)
    if neighbourhood == 16:
        knight = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
        return np.array(moore + knight, dtype=int)
    raise ValueError("neighbourhood must be 4, 8 or 16")


@dataclass(frozen=True)
class ArrivalField:
    """Output of :func:`arrival_time`.

    Attributes
    ----------
    time_s : (nx, ny) float array
        Finsler distance from (or to) the source set, in seconds; ``inf`` where
        unreachable.
    source_index : (nx, ny) int array
        Index into the source list of the seed each cell's geodesic starts from;
        ``-1`` where unreachable. This is the "which part of the front does the
        fire arrive from" label.
    parent : (nx, ny) int array
        Flat index of the predecessor on the shortest path, for
        :func:`geodesic_path`; ``-1`` at seeds and unreachable cells.
    reverse : bool
        ``False`` for ``d_F(source, x)``, ``True`` for ``d_F(x, source)``.
    """

    time_s: np.ndarray
    source_index: np.ndarray
    parent: np.ndarray
    reverse: bool


def arrival_time(
    field: RandersField,
    source_mask: np.ndarray,
    *,
    reverse: bool = False,
    neighbourhood: int = 16,
) -> ArrivalField:
    """Solve for the Finsler distance field from a source set by Dijkstra.

    Arrival time solves the anisotropic eikonal ``F*(x, dT) = 1``; on the CA's
    own lattice the faithful discretisation is a shortest-path problem with
    *directed* edge weights, where the weight of the step ``p -> q`` is the
    trapezoidal slowness integral

        w(p -> q) = |q - p| / 2 * (1 / sigma(p, u) + 1 / sigma(q, u)),
        u = (q - p) / |q - p|.

    Because ``sigma(x, u) != sigma(x, -u)``, ``w(p -> q) != w(q -> p)``: the
    resulting distance is genuinely asymmetric, which is the whole point.

    Parameters
    ----------
    reverse : bool
        ``False`` gives ``T(x) = d_F(source, x)`` -- when the fire reaches ``x``.
        ``True`` gives ``T(x) = d_F(x, source)`` -- how long a fire starting at
        ``x`` would take to reach the source set. The two differ under wind, and
        :class:`FinslerWarp` keeps both.
    """
    speed = field.speed
    nx, ny = field.grid_size
    source_mask = np.asarray(source_mask, dtype=bool)
    if source_mask.shape != (nx, ny):
        raise ValueError(f"source_mask must be {(nx, ny)}; got {source_mask.shape}")

    offsets = neighbour_offsets(neighbourhood)
    n_off = offsets.shape[0]
    lengths = np.linalg.norm(offsets, axis=1) * field.dx_m
    units = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)

    # Per-direction slowness, 1 / sigma(x, u). A direction the front cannot move
    # in at all -- no fuel, or a drift so strong the metric stops being strongly
    # convex and the upwind speed goes non-positive -- costs infinite time, not
    # zero: taking the reciprocal of an infinite speed would make it a free edge.
    sigma = zermelo_speed(field.speed, field.drift, units)
    passable = field.burnable[None, ...] & (sigma > _EPS)
    slowness = np.where(passable, 1.0 / np.where(passable, sigma, 1.0), np.inf)

    # Travelling backwards along an edge costs what the opposite direction costs.
    opposite = np.array(
        [int(np.argmin(np.abs(units + units[k]).sum(axis=1))) for k in range(n_off)], dtype=int
    )
    cost_dir = slowness[opposite] if reverse else slowness

    seeds = np.flatnonzero(source_mask.ravel() & field.burnable.ravel())
    time_s = np.full(nx * ny, np.inf)
    src_idx = np.full(nx * ny, -1, dtype=int)
    parent = np.full(nx * ny, -1, dtype=int)
    settled = np.zeros(nx * ny, dtype=bool)

    heap: list[tuple[float, int]] = []
    for label, flat in enumerate(seeds):
        time_s[flat] = 0.0
        src_idx[flat] = label
        heap.append((0.0, int(flat)))
    heapq.heapify(heap)

    flat_cost = cost_dir.reshape(n_off, -1)
    burnable_flat = field.burnable.ravel()

    while heap:
        dist, flat = heapq.heappop(heap)
        if settled[flat]:
            continue
        settled[flat] = True
        px, py = divmod(flat, ny)

        for k in range(n_off):
            qx = px + offsets[k, 0]
            qy = py + offsets[k, 1]
            if qx < 0 or qx >= nx or qy < 0 or qy >= ny:
                continue
            q = qx * ny + qy
            if settled[q] or not burnable_flat[q]:
                continue
            step = 0.5 * lengths[k] * (flat_cost[k, flat] + flat_cost[k, q])
            if not np.isfinite(step):
                continue
            nd = dist + step
            if nd < time_s[q]:
                time_s[q] = nd
                src_idx[q] = src_idx[flat]
                parent[q] = flat
                heapq.heappush(heap, (float(nd), int(q)))

    return ArrivalField(
        time_s=time_s.reshape(nx, ny),
        source_index=src_idx.reshape(nx, ny),
        parent=parent.reshape(nx, ny),
        reverse=bool(reverse),
    )


def geodesic_path(arrival: ArrivalField, ix: int, iy: int) -> np.ndarray:
    """Backtrack the minimum-time path between a cell and the source set.

    Returns ``(m, 2)`` cell coordinates ordered from the source to ``(ix, iy)``
    for a forward field, and from ``(ix, iy)`` to the source for a reverse one.
    """
    ny = arrival.time_s.shape[1]
    flat = int(ix) * ny + int(iy)
    if not np.isfinite(arrival.time_s.ravel()[flat]):
        return np.empty((0, 2), dtype=float)

    parent = arrival.parent.ravel()
    path = [flat]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    cells = np.array([divmod(f, ny) for f in reversed(path)], dtype=float)
    return cells if not arrival.reverse else cells[::-1].copy()


def directed_distance_matrix(
    field: RandersField,
    points: np.ndarray,
    *,
    neighbourhood: int = 16,
) -> np.ndarray:
    """All-pairs directed Finsler distances ``D[i, j] = d_F(points[i], points[j])``.

    Provided so the asymmetry can be *shown*: ``D`` is not symmetric, therefore
    ``exp(-D)`` (or any Matern of it) is not a covariance matrix. This function
    exists to justify the warp in :class:`FinslerWarp`, not to be fed to a GP.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float)).astype(int)
    nx, ny = field.grid_size
    out = np.zeros((pts.shape[0], pts.shape[0]))
    for i, (px, py) in enumerate(pts):
        seed = np.zeros((nx, ny), dtype=bool)
        seed[px, py] = True
        arr = arrival_time(field, seed, neighbourhood=neighbourhood)
        out[i] = arr.time_s[pts[:, 0], pts[:, 1]]
    return out


# ---------------------------------------------------------------------------
# The warp: geometry -> kernel-safe features
# ---------------------------------------------------------------------------


def _bilinear(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a scalar grid at continuous cell coordinates."""
    nx, ny = grid.shape
    x = np.clip(np.asarray(x, dtype=float), 0.0, nx - 1.0)
    y = np.clip(np.asarray(y, dtype=float), 0.0, ny - 1.0)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    fx = x - x0
    fy = y - y0
    return (
        grid[x0, y0] * (1 - fx) * (1 - fy)
        + grid[x1, y0] * fx * (1 - fy)
        + grid[x0, y1] * (1 - fx) * fy
        + grid[x1, y1] * fx * fy
    )


def _fill_infinite(values: np.ndarray, mask: np.ndarray, fill: float) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    out[~mask] = fill
    return out


@dataclass
class FinslerWarp:
    """Deterministic map from a drop ``(x, y, phi)`` to kernel-safe features.

    The six features per drone are

    ==== ======================================================================
    0    forward arrival time ``d_F(front, x)``, normalised
    1    reverse arrival time ``d_F(x, front)``, normalised
    2-3  ``cos``/``sin`` of the front-arclength label of the geodesic's source
    4-5  ``cos``/``sin`` of twice the drop's angle to the local front normal
    ==== ======================================================================

    Features 0 and 1 are the honest treatment of the asymmetry: rather than
    symmetrising ``d_F`` and discarding the wind information, both directed
    distances are kept as separate coordinates, and their difference -- the
    upwind/downwind signature of the location -- is recoverable from the pair.

    Features 2-3 use a periodic embedding because the front arclength wraps, and
    4-5 use ``2 * angle`` because the retardant rectangle has no head or tail:
    ``delta`` and ``delta + pi`` are the same drop.

    Because the map is a fixed function of the fire state -- it involves no
    kernel hyperparameters and no learned quantities -- any PSD stationary
    kernel evaluated on these features is itself PSD.
    """

    field: RandersField
    forward: ArrivalField
    reverse: ArrivalField
    source_arclength: np.ndarray            # (n_sources,) in [0, 1)
    time_scale_s: float
    valid: np.ndarray                       # (nx, ny) reachable both ways

    # Derived grids, filled in __post_init__.
    t_forward: np.ndarray = _dc_field(init=False)
    t_reverse: np.ndarray = _dc_field(init=False)
    cos_s: np.ndarray = _dc_field(init=False)
    sin_s: np.ndarray = _dc_field(init=False)
    normal: np.ndarray = _dc_field(init=False)

    N_FEATURES: int = 6

    def __post_init__(self) -> None:
        scale = float(self.time_scale_s)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("time_scale_s must be finite and positive")

        reachable_f = np.isfinite(self.forward.time_s)
        reachable_r = np.isfinite(self.reverse.time_s)
        # Unreachable cells are pinned one scale beyond the horizon rather than
        # left at inf, so the surrogate sees "very far" instead of NaN.
        self.t_forward = _fill_infinite(self.forward.time_s / scale, reachable_f, 2.0)
        self.t_reverse = _fill_infinite(self.reverse.time_s / scale, reachable_r, 2.0)

        labels = np.asarray(self.source_arclength, dtype=float)
        idx = self.forward.source_index
        ang = np.where(idx >= 0, 2.0 * np.pi * labels[np.clip(idx, 0, labels.size - 1)], 0.0)
        self.cos_s = np.where(idx >= 0, np.cos(ang), 0.0)
        self.sin_s = np.where(idx >= 0, np.sin(ang), 0.0)

        # Front normal = direction of increasing arrival time.
        filled = _fill_infinite(self.forward.time_s, reachable_f, np.nanmax(self.forward.time_s[reachable_f], initial=scale))
        gx, gy = np.gradient(filled)
        mag = np.hypot(gx, gy)
        self.normal = np.stack(
            [np.divide(gx, mag, out=np.zeros_like(gx), where=mag > _EPS),
             np.divide(gy, mag, out=np.zeros_like(gy), where=mag > _EPS)],
            axis=-1,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_firestate(
        cls,
        env,
        firestate,
        *,
        ros_mps: float | np.ndarray | None = None,
        wind_coeff: float | None = None,
        neighbourhood: int = 16,
        time_scale_quantile: float = 0.9,
        max_wind_ratio: float = 0.95,
        randers: RandersField | None = None,
    ) -> "FinslerWarp":
        """Build the warp from the current fire perimeter of ``firestate``."""
        rf = randers_from_env(
            env, ros_mps=ros_mps, wind_coeff=wind_coeff, max_wind_ratio=max_wind_ratio
        ) if randers is None else randers

        burning = np.asarray(firestate.burning)
        burned = np.asarray(firestate.burned)
        if burning.ndim == 3:
            burning, burned = burning[0], burned[0]
        if np.issubdtype(burning.dtype, np.floating):
            affected = np.clip(burning + burned, 0.0, 1.0) > 0.5
        else:
            affected = burning | burned
        if not affected.any():
            raise ValueError("firestate has no ignited cells to propagate from")

        return cls.from_source_mask(
            rf, affected, neighbourhood=neighbourhood, time_scale_quantile=time_scale_quantile
        )

    @classmethod
    def from_source_mask(
        cls,
        field: RandersField,
        source_mask: np.ndarray,
        *,
        neighbourhood: int = 16,
        time_scale_quantile: float = 0.9,
    ) -> "FinslerWarp":
        """Build the warp from an explicit ignition set."""
        source_mask = np.asarray(source_mask, dtype=bool)
        fwd = arrival_time(field, source_mask, neighbourhood=neighbourhood)
        rev = arrival_time(field, source_mask, reverse=True, neighbourhood=neighbourhood)

        # Arclength label per seed, ordered by angle about the source centroid,
        # so that neighbouring front cells get neighbouring labels.
        seeds = np.argwhere(source_mask & field.burnable).astype(float)
        if seeds.size == 0:
            raise ValueError("source_mask selects no burnable cells")
        centroid = seeds.mean(axis=0)
        ang = np.arctan2(seeds[:, 1] - centroid[1], seeds[:, 0] - centroid[0])
        arclength = np.mod(ang, 2.0 * np.pi) / (2.0 * np.pi)

        finite = np.isfinite(fwd.time_s)
        scale = float(np.quantile(fwd.time_s[finite], time_scale_quantile)) if finite.any() else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        return cls(
            field=field,
            forward=fwd,
            reverse=rev,
            source_arclength=arclength,
            time_scale_s=scale,
            valid=finite & np.isfinite(rev.time_s),
        )

    # -- the map -----------------------------------------------------------

    def features(self, params: np.ndarray) -> np.ndarray:
        """Warp drop parameters into kernel space.

        Parameters
        ----------
        params : (d, 3) array of ``(x_cell, y_cell, phi)``.

        Returns
        -------
        (d, 6) array of features, in the order documented on the class.
        """
        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.shape[1] != 3:
            raise ValueError(f"expected (d, 3) drop parameters; got {params.shape}")
        x, y, phi = params[:, 0], params[:, 1], params[:, 2]

        t_f = _bilinear(self.t_forward, x, y)
        t_r = _bilinear(self.t_reverse, x, y)
        cos_s = _bilinear(self.cos_s, x, y)
        sin_s = _bilinear(self.sin_s, x, y)
        norm = np.hypot(cos_s, sin_s)
        cos_s = np.divide(cos_s, norm, out=np.zeros_like(cos_s), where=norm > _EPS)
        sin_s = np.divide(sin_s, norm, out=np.zeros_like(sin_s), where=norm > _EPS)

        nxg = _bilinear(self.normal[..., 0], x, y)
        nyg = _bilinear(self.normal[..., 1], x, y)
        normal_angle = np.arctan2(nyg, nxg)
        # apply_retardant_cartesian aligns the rectangle's long axis with
        # [sin(phi), cos(phi)], i.e. a long-axis angle of pi/2 - phi.
        long_axis_angle = 0.5 * np.pi - phi
        delta = long_axis_angle - normal_angle

        return np.stack([t_f, t_r, cos_s, sin_s, np.cos(2.0 * delta), np.sin(2.0 * delta)], axis=1)

    def asymmetry(self, params: np.ndarray) -> np.ndarray:
        """``(t_reverse - t_forward) / (t_reverse + t_forward)`` at each drop.

        Positive downwind of the front, negative upwind; identically zero for a
        symmetric (Riemannian) metric. A diagnostic, not a feature: it is a
        function of features 0 and 1, which the GP already has.
        """
        f = self.features(params)
        denom = np.maximum(f[:, 0] + f[:, 1], _EPS)
        return (f[:, 1] - f[:, 0]) / denom

    def describe(self) -> dict:
        """Summary of the warp, including how asymmetric this fire actually is."""
        v = self.valid
        fwd = self.forward.time_s[v]
        rev = self.reverse.time_s[v]
        denom = np.maximum(fwd + rev, _EPS)
        asym = (rev - fwd) / denom
        return {
            "metric": self.field.describe(),
            "time_scale_s": float(self.time_scale_s),
            "reachable_cells": int(v.sum()),
            "median_abs_asymmetry": float(np.median(np.abs(asym))) if asym.size else 0.0,
            "max_abs_asymmetry": float(np.max(np.abs(asym), initial=0.0)),
            "n_features_per_drone": int(self.N_FEATURES),
        }


# ---------------------------------------------------------------------------
# A front-free search map
# ---------------------------------------------------------------------------


@dataclass
class FinslerSearchMap:
    """Invert Finsler front coordinates without forecasting a future boundary.

    The SR search map needs a Monte-Carlo rollout to construct its outer
    boundary.  This map instead uses the deterministic arrival field already
    carried by :class:`FinslerWarp`.  A point is represented by

    ``(s, tau)``
        ``s`` is the periodic label of the current-perimeter source reached by
        its geodesic; ``tau`` is normalised forward arrival time in an
        actionable window.  The optimiser's third coordinate is the long-axis
        angle relative to the local arrival-time normal.

    Inversion is a nearest-neighbour query in ``[cos(2πs), sin(2πs), tau]``.
    The periodic embedding avoids a seam at ``s=0``.  No simulated future fire
    state, extracted outer boundary, or Laplace-smoothed strip is involved.
    """

    warp: FinslerWarp
    min_time_s: float
    max_time_s: float
    mask: np.ndarray = _dc_field(init=False)
    cells: np.ndarray = _dc_field(init=False)
    coordinates: np.ndarray = _dc_field(init=False)
    tree: cKDTree = _dc_field(init=False)

    def __post_init__(self) -> None:
        self.min_time_s = float(self.min_time_s)
        self.max_time_s = float(self.max_time_s)
        if self.min_time_s < 0.0 or self.max_time_s <= self.min_time_s:
            raise ValueError("require 0 <= min_time_s < max_time_s")

        time = np.asarray(self.warp.forward.time_s, dtype=float)
        self.mask = (
            self.warp.valid
            & self.warp.field.burnable
            & np.isfinite(time)
            & (time >= self.min_time_s)
            & (time <= self.max_time_s)
        )
        self.cells = np.argwhere(self.mask).astype(float)
        if self.cells.size == 0:
            raise ValueError(
                "Finsler search window contains no reachable cells; "
                "increase max_time_s or reduce min_time_s"
            )

        ix = self.cells[:, 0].astype(int)
        iy = self.cells[:, 1].astype(int)
        source_angle = np.arctan2(self.warp.sin_s[ix, iy], self.warp.cos_s[ix, iy])
        source_s = np.mod(source_angle, 2.0 * np.pi) / (2.0 * np.pi)
        tau = (time[ix, iy] - self.min_time_s) / (self.max_time_s - self.min_time_s)
        # The factor 2 gives front label and arrival time comparable influence
        # in nearest-neighbour inversion because the unit circle has diameter 2.
        self.coordinates = np.column_stack(
            [np.cos(2.0 * np.pi * source_s), np.sin(2.0 * np.pi * source_s), 2.0 * tau]
        )
        self.tree = cKDTree(self.coordinates)

    def lookup(self, s: float, tau: float) -> tuple[np.ndarray, float]:
        """Return ``(xy, normal_angle)`` nearest to front coordinates."""
        s = float(np.mod(s, 1.0))
        tau = float(np.clip(tau, 0.0, 1.0))
        query = [np.cos(2.0 * np.pi * s), np.sin(2.0 * np.pi * s), 2.0 * tau]
        _, index = self.tree.query(query, k=1)
        xy = self.cells[int(index)]
        ix, iy = xy.astype(int)
        normal = self.warp.normal[ix, iy]
        normal_angle = float(np.arctan2(normal[1], normal[0]))
        return xy.copy(), normal_angle

    def decode(self, theta: np.ndarray, n_drones: int) -> np.ndarray:
        """Decode repeating ``[s, tau, delta]`` blocks to ``[x, y, phi]``.

        ``delta`` spans only ``π`` because a retardant rectangle has no head or
        tail.  ``phi`` follows the convention used by
        ``CAFireModel.apply_retardant_cartesian``.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != 3 * int(n_drones):
            raise ValueError(f"expected {3 * int(n_drones)} parameters; got {theta.size}")

        params = []
        for drone in range(int(n_drones)):
            s = theta[3 * drone]
            tau = theta[3 * drone + 1]
            delta = theta[3 * drone + 2] * np.pi
            xy, normal_angle = self.lookup(s, tau)
            long_axis_angle = normal_angle + delta
            phi = np.mod(0.5 * np.pi - long_axis_angle, 2.0 * np.pi)
            params.append((float(xy[0]), float(xy[1]), float(phi)))

        out = np.asarray(params, dtype=float)
        order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
        return out[order]

    def describe(self) -> dict:
        time = self.warp.forward.time_s[self.mask]
        return {
            "setup_rollout_steps": 0,
            "candidate_cells": int(self.cells.shape[0]),
            "min_time_s": self.min_time_s,
            "max_time_s": self.max_time_s,
            "actual_time_range_s": [float(time.min()), float(time.max())],
        }


# ---------------------------------------------------------------------------
# A stationary kernel on the warped space
# ---------------------------------------------------------------------------


class TiedFinslerMatern(Kernel):
    """Matern kernel over repeating 6D Finsler blocks, one block per drone.

    Each block is ``[t_forward, t_reverse, cos s, sin s, cos 2d, sin 2d]`` as
    produced by :meth:`FinslerWarp.features`. Lengthscales are tied across
    drones (drops are exchangeable) but separate per feature role:

    ``l_forward``
        similarity along the arrival-time coordinate -- how far ahead of the
        front two drops are.
    ``l_reverse``
        similarity along the reverse arrival time. This is the asymmetry
        channel: if marginal likelihood pushes ``l_reverse`` to its upper bound
        the data say the upwind/downwind distinction carries no signal, and if
        it settles near ``l_forward`` the directed geometry is doing work. The
        fitted value is therefore a read-out, not just a nuisance parameter.
    ``l_front``
        similarity in which part of the perimeter the fire arrives from.
    ``l_delta``
        similarity in drop orientation relative to the local front normal.

    Positive definiteness is inherited from :class:`~sklearn.gaussian_process.kernels.Matern`:
    the warp is applied to the inputs before the kernel sees them and involves
    no hyperparameters, so this is a Matern kernel composed with a fixed map.
    """

    N_FEATURES = FinslerWarp.N_FEATURES

    def __init__(
        self,
        l_forward: float = 0.3,
        l_reverse: float = 0.6,
        l_front: float = 0.5,
        l_delta: float = 0.6,
        nu: float = 2.5,
        length_scale_bounds: tuple[float, float] = (1e-3, 1e3),
        fd_eps: float = 1e-6,
    ):
        self.l_forward = float(l_forward)
        self.l_reverse = float(l_reverse)
        self.l_front = float(l_front)
        self.l_delta = float(l_delta)
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds
        self.fd_eps = float(fd_eps)
        self._base = Matern(length_scale=1.0, nu=nu)

    @property
    def hyperparameter_l_forward(self):
        return Hyperparameter("l_forward", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_l_reverse(self):
        return Hyperparameter("l_reverse", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_l_front(self):
        return Hyperparameter("l_front", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_l_delta(self):
        return Hyperparameter("l_delta", "numeric", self.length_scale_bounds)

    @property
    def theta(self):
        return np.log([self.l_forward, self.l_reverse, self.l_front, self.l_delta])

    @theta.setter
    def theta(self, theta):
        l_forward, l_reverse, l_front, l_delta = np.exp(theta)
        self.l_forward = float(l_forward)
        self.l_reverse = float(l_reverse)
        self.l_front = float(l_front)
        self.l_delta = float(l_delta)

    @property
    def bounds(self):
        return np.log(np.array([self.length_scale_bounds] * 4, dtype=float))

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        d = X.shape[1]
        if d % self.N_FEATURES != 0:
            raise ValueError(
                f"Expected feature dim multiple of {self.N_FEATURES}, got {d}. "
                "Use FinslerWarp.features(...) to build the inputs."
            )
        role = np.arange(d) % self.N_FEATURES
        scales = np.select(
            [role == 0, role == 1, role <= 3],
            [self.l_forward, self.l_reverse, self.l_front],
            default=self.l_delta,
        )
        return X / scales

    def __call__(self, X, Y=None, eval_gradient=False):
        Xs = self._scale(X)
        Ys = self._scale(Y) if Y is not None else None

        K = self._base(Xs, Ys, eval_gradient=False)
        if not eval_gradient:
            return K

        theta0 = self.theta.copy()
        grad = np.empty(K.shape + (theta0.size,), dtype=float)
        for i in range(theta0.size):
            th = theta0.copy()
            th[i] += self.fd_eps
            Kp = self.clone_with_theta(th)(X, Y, eval_gradient=False)
            grad[..., i] = (Kp - K) / self.fd_eps
        return K, grad

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(l_forward={self.l_forward:.3g}, "
            f"l_reverse={self.l_reverse:.3g}, l_front={self.l_front:.3g}, "
            f"l_delta={self.l_delta:.3g}, nu={self.nu})"
        )


def naive_symmetrised_gram(
    distances: np.ndarray,
    *,
    mode: str = "mean",
    length_scale: float = 1.0,
    nu: float = 2.5,
) -> np.ndarray:
    """Build the kernel matrix you get by *substituting* a Finsler distance.

    ``mode="directed"`` plugs the raw directed distance into a Matern profile,
    which is not even a symmetric matrix. ``mode="mean"`` and ``mode="min"``
    are the tempting repairs; they are symmetric but carry no guarantee of
    positive definiteness, and both discard the upwind/downwind information.

    Used by :mod:`fire_model.finsler_validation` to report the resulting minimum
    eigenvalue next to the warp's. Not for use in a GP.
    """
    d = np.asarray(distances, dtype=float)
    if mode == "directed":
        sym = d
    elif mode == "mean":
        sym = 0.5 * (d + d.T)
    elif mode == "min":
        sym = np.minimum(d, d.T)
    elif mode == "max":
        sym = np.maximum(d, d.T)
    else:
        raise ValueError("mode must be 'directed', 'mean', 'min' or 'max'")

    r = sym / float(length_scale)
    if nu == 2.5:
        k = (1.0 + np.sqrt(5) * r + 5.0 / 3.0 * r ** 2) * np.exp(-np.sqrt(5) * r)
    elif nu == 1.5:
        k = (1.0 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif nu == 0.5:
        k = np.exp(-r)
    else:
        raise ValueError("nu must be 0.5, 1.5 or 2.5")
    return k


__all__ = [
    "RandersField",
    "ArrivalField",
    "FinslerWarp",
    "FinslerSearchMap",
    "TiedFinslerMatern",
    "uniform_directions",
    "fit_randers_profile",
    "zermelo_speed",
    "ca_directional_speed",
    "randers_from_env",
    "neighbour_offsets",
    "arrival_time",
    "geodesic_path",
    "directed_distance_matrix",
    "naive_symmetrised_gram",
]
