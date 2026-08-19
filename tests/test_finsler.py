"""Tests for the Randers-Finsler geometry and the kernel warp built on it.

Three things are worth pinning down here, in increasing order of subtlety:
the Zermelo-to-Randers algebra, the asymmetry of the resulting distance, and
the reason that asymmetry forces a warp rather than a substitution.
"""

import numpy as np
import pytest
from dataclasses import replace

from fire_model.bo_sr import RetardantDropBayesOptSR
from fire_model.ca import CAFireModel, FireEnv, FireState
from fire_model.finsler import (
    FinslerSearchMap,
    FinslerWarp,
    RandersField,
    TiedFinslerMatern,
    arrival_time,
    ca_directional_speed,
    directed_distance_matrix,
    fit_randers_profile,
    geodesic_path,
    naive_symmetrised_gram,
    randers_from_env,
    uniform_directions,
    zermelo_speed,
)


def homogeneous_field(n=41, speed=1.0, drift=(0.3, 0.0), dx_m=100.0):
    return RandersField(
        speed=np.full((n, n), float(speed)),
        drift=np.tile(np.asarray(drift, dtype=float), (n, n, 1)),
        dx_m=float(dx_m),
        burnable=np.ones((n, n), dtype=bool),
    )


def make_env(nx=32, wind=(1.0, 0.0), wind_coeff=0.8, wind_response="clipped", ros=0.7):
    return FireEnv(
        grid_size=(nx, nx),
        domain_km=4.0,
        fuel=np.ones((nx, nx)),
        value=np.ones((nx, nx)),
        wind=np.tile(np.asarray(wind, dtype=float), (nx, nx, 1)),
        dt_s=30.0,
        ros_mps=ros,
        wind_coeff=wind_coeff,
        wind_response=wind_response,
    )


def make_firestate(nx=32, half=2):
    burning = np.zeros((1, nx, nx), dtype=bool)
    c = nx // 2
    burning[0, c - half : c + half, c - half : c + half] = True
    return FireState(
        burning=burning,
        burned=np.zeros_like(burning),
        burn_remaining_s=np.full((1, nx, nx), 600.0),
        retardant=np.zeros((1, nx, nx)),
        t=0,
    )


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def test_indicatrix_is_the_unit_sphere_of_the_randers_metric():
    """F(x, v) = 1 exactly on the reachable-in-unit-time velocity set."""
    field = homogeneous_field(n=5, speed=0.7, drift=(0.3, -0.15))
    for u in uniform_directions(24):
        v = 0.7 * u + np.array([0.3, -0.15])
        assert field.finsler_norm(v, 2, 2) == pytest.approx(1.0, abs=1e-12)


def test_one_form_norm_equals_the_wind_ratio():
    """||b||_a == |W| / s, so strong convexity is exactly the subunit-wind condition."""
    field = homogeneous_field(n=5, speed=0.7, drift=(0.3, -0.15))
    a, b = field.randers_tensors()
    norm_b = np.sqrt(b[2, 2] @ np.linalg.inv(a[2, 2]) @ b[2, 2])

    assert norm_b == pytest.approx(field.wind_ratio()[2, 2], rel=1e-12)
    assert field.is_strongly_convex()


def test_fit_recovers_an_exactly_randers_profile():
    dirs = uniform_directions(32)
    speed, drift = 0.9, np.array([0.25, -0.4])
    sigma = zermelo_speed(np.array(speed), drift, dirs)

    s, W, residual = fit_randers_profile(sigma[:, None, None], dirs)

    assert s.item() == pytest.approx(speed, rel=1e-12)
    assert W.reshape(2) == pytest.approx(drift, rel=1e-12)
    assert residual.item() == pytest.approx(0.0, abs=1e-12)


def test_radial_function_is_not_the_linear_form():
    """sigma(u) is 1 / (sqrt(a(u,u)) + b.u), not s + W.u.

    Those agree only to first order in |W|/s and diverge badly at the flanks --
    67% at |W|/s = 0.8. Fitting the linear form and then calling the result a
    Randers metric silently builds a different geometry, so the distinction is
    pinned here.
    """
    field = homogeneous_field(n=3, speed=0.7, drift=(0.56, 0.0))     # |W|/s = 0.8
    dirs = uniform_directions(4)
    sigma = field.directional_speed(dirs)[:, 1, 1]
    linear = 0.7 + dirs @ np.array([0.56, 0.0])

    assert sigma[0] == pytest.approx(linear[0])                      # agree downwind
    assert sigma[2] == pytest.approx(linear[2])                      # and upwind
    flank = sigma[1]
    assert flank == pytest.approx(np.sqrt(0.7 ** 2 - 0.56 ** 2))     # sqrt(s^2 - |W|^2)
    assert abs(flank - linear[1]) / flank > 0.6                      # but not at the flank

    # ... and the fitted metric's indicatrix really is F = 1.
    for u, speed_u in zip(uniform_directions(16), field.directional_speed(uniform_directions(16))[:, 1, 1]):
        assert field.finsler_norm(speed_u * u, 1, 1) == pytest.approx(1.0, abs=1e-12)


def test_elliptical_ca_spread_law_is_exactly_a_randers_metric():
    """With wind_response='elliptical' the metric restates the CA rather than approximating it."""
    env = make_env(wind_coeff=0.6, wind_response="elliptical")
    field = randers_from_env(env)

    assert np.max(field.fit_residual[field.burnable]) < 1e-12
    # sigma(u) = ros * fuel * (1 + c w.u) => s = ros, W = ros * c * w
    assert field.speed[5, 5] == pytest.approx(0.7, rel=1e-12)
    assert field.drift[5, 5] == pytest.approx([0.7 * 0.6, 0.0], abs=1e-12)
    assert field.clipped_fraction == 0.0


def test_clipped_ca_misfit_grows_with_wind():
    """The default CA clips its wind term, and no ellipse can match a flat upwind half."""
    residuals = [
        float(np.median(randers_from_env(make_env(wind_coeff=c)).fit_residual))
        for c in (0.0, 0.4, 0.8, 1.6)
    ]

    assert residuals[0] == pytest.approx(0.0, abs=1e-10)
    assert residuals == sorted(residuals)
    assert residuals[-1] > 0.05


def test_drift_is_shrunk_to_keep_the_metric_strongly_convex():
    env = make_env(wind_coeff=3.0, wind_response="elliptical")
    field = randers_from_env(env, max_wind_ratio=0.9)

    assert field.is_strongly_convex()
    assert np.max(field.wind_ratio()) <= 0.9 + 1e-12
    assert field.clipped_fraction > 0.0


# ---------------------------------------------------------------------------
# Arrival time
# ---------------------------------------------------------------------------


def test_driftless_arrival_time_is_symmetric_and_metric():
    """Zero drift must give back an isotropic Riemannian distance."""
    n, speed = 41, 0.5
    field = homogeneous_field(n=n, speed=speed, drift=(0.0, 0.0))
    src = np.zeros((n, n), dtype=bool)
    src[n // 2, n // 2] = True

    forward = arrival_time(field, src)
    reverse = arrival_time(field, src, reverse=True)
    assert np.allclose(forward.time_s, reverse.time_s)

    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    exact = np.hypot(xx - n // 2, yy - n // 2) * field.dx_m / speed
    far = exact > 0
    # The 16-neighbour stencil keeps the lattice's own directional bias small.
    assert np.max(np.abs(forward.time_s[far] - exact[far]) / exact[far]) < 0.03


def test_arrival_time_is_asymmetric_under_drift():
    """The point of the exercise: travelling upwind is not travelling downwind."""
    n = 41
    c = n // 2
    field = homogeneous_field(n=n, speed=0.5, drift=(0.3, 0.0))
    src = np.zeros((n, n), dtype=bool)
    src[c, c] = True

    forward = arrival_time(field, src).time_s
    reverse = arrival_time(field, src, reverse=True).time_s
    downwind, upwind = (c + 15, c), (c - 15, c)

    assert forward[downwind] < reverse[downwind]      # easy to reach, slow to come back from
    assert forward[upwind] > reverse[upwind]
    assert forward[downwind] == pytest.approx(reverse[upwind], rel=1e-9)
    # Straight downwind is exactly (speed + drift); no path can beat it.
    assert forward[downwind] == pytest.approx(15 * field.dx_m / 0.8, rel=1e-9)


def test_normalised_asymmetry_recovers_the_one_form_norm():
    """max |(d_F(x,G) - d_F(G,x)) / sum| == ||b||_a for a homogeneous metric.

    A tidy consistency check between the two descriptions: the asymmetry visible
    in the arrival-time field is exactly the size of the Randers one-form.
    """
    n = 51
    for ratio in (0.1, 0.3, 0.5):
        field = homogeneous_field(n=n, speed=1.0, drift=(ratio, 0.0))
        src = np.zeros((n, n), dtype=bool)
        src[n // 2, n // 2] = True
        warp = FinslerWarp.from_source_mask(field, src)

        fwd, rev = warp.forward.time_s, warp.reverse.time_s
        asym = (rev - fwd) / np.maximum(rev + fwd, 1e-12)
        assert np.max(np.abs(asym[np.isfinite(asym)])) == pytest.approx(ratio, abs=1e-6)


def test_unburnable_cells_block_propagation():
    """A firebreak blocks the front only if it is wider than the stencil's reach.

    The 16-neighbour stencil includes knight moves, so it steps clean over a
    one-cell break. That is a property of the discretisation, not of the fire,
    and it is what sets the minimum width a barrier must have to register.
    """
    n = 21
    field = homogeneous_field(n=n, speed=0.5, drift=(0.0, 0.0))
    src = np.zeros((n, n), dtype=bool)
    src[0, n // 2] = True

    def reach(width, neighbourhood):
        burnable = field.burnable.copy()
        burnable[n // 2 : n // 2 + width, :] = False
        blocked = replace(field, burnable=burnable)
        return arrival_time(blocked, src, neighbourhood=neighbourhood).time_s[n - 1, n // 2]

    assert not np.isfinite(reach(width=2, neighbourhood=16))
    assert not np.isfinite(reach(width=1, neighbourhood=8))
    assert np.isfinite(reach(width=1, neighbourhood=16))     # knight moves jump it


def test_a_direction_the_front_cannot_take_costs_infinite_time():
    """A non-convex metric blocks upwind travel; it must not become a free edge.

    When |W| exceeds s the upwind spread rate is non-positive, so the front
    cannot move that way at all. Inverting that rate naively would give a
    zero-cost edge -- an infinitely fast path in the one direction the fire
    cannot go.
    """
    n = 21
    c = n // 2
    field = homogeneous_field(n=n, speed=0.4, drift=(0.9, 0.0))     # |W| > s
    assert not field.is_strongly_convex()

    src = np.zeros((n, n), dtype=bool)
    src[c, c] = True
    forward = arrival_time(field, src, neighbourhood=4).time_s

    assert np.isfinite(forward[c + 5, c])                            # downwind is fine
    assert not np.isfinite(forward[c - 5, c])                        # upwind is impossible
    assert forward[c + 5, c] > 0.0


def test_geodesic_path_connects_source_to_target():
    n = 31
    field = homogeneous_field(n=n, speed=0.5, drift=(0.25, 0.0))
    src = np.zeros((n, n), dtype=bool)
    src[2, n // 2] = True

    forward = arrival_time(field, src)
    path = geodesic_path(forward, n - 3, n // 2)

    assert path.shape[1] == 2
    assert tuple(path[0]) == (2.0, float(n // 2))
    assert tuple(path[-1]) == (float(n - 3), float(n // 2))


# ---------------------------------------------------------------------------
# Why a warp and not a substitution
# ---------------------------------------------------------------------------


def test_substituting_the_finsler_distance_does_not_give_a_covariance():
    """The directed distance is not symmetric, and symmetrising it is not enough.

    This is the reason the warp exists, so it is asserted rather than asserted
    about: a Matern profile of the directed distance is not a symmetric matrix,
    and the two obvious repairs both produce indefinite Gram matrices here.
    """
    field = homogeneous_field(n=41, speed=0.5, drift=(0.35, 0.0))
    rng = np.random.default_rng(0)
    points = rng.integers(2, 39, size=(12, 2))

    D = directed_distance_matrix(field, points)
    assert not np.allclose(D, D.T)

    directed = naive_symmetrised_gram(D, mode="directed", length_scale=np.median(D[D > 0]))
    assert not np.allclose(directed, directed.T)     # not even a symmetric matrix

    for mode in ("mean", "min"):
        K = naive_symmetrised_gram(D, mode=mode, length_scale=np.median(D[D > 0]))
        assert np.allclose(K, K.T)
        assert np.linalg.eigvalsh(K).min() < -1e-6   # symmetric, but indefinite


def test_warped_kernel_is_positive_semidefinite():
    """Composing a stationary kernel with a deterministic warp keeps it PSD."""
    env = make_env(wind_coeff=0.8, wind_response="elliptical")
    warp = FinslerWarp.from_firestate(env, make_firestate())

    rng = np.random.default_rng(1)
    params = np.column_stack(
        [rng.uniform(1, 30, 40), rng.uniform(1, 30, 40), rng.uniform(0, 2 * np.pi, 40)]
    )
    K = TiedFinslerMatern()(warp.features(params))

    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() >= -1e-10
    assert np.allclose(np.diag(K), 1.0)


def test_warp_features_are_bounded_and_deterministic():
    env = make_env(wind_coeff=0.8, wind_response="elliptical")
    warp = FinslerWarp.from_firestate(env, make_firestate())
    params = np.array([[10.0, 12.0, 0.3], [20.0, 8.0, 2.0]])

    features = warp.features(params)
    assert features.shape == (2, FinslerWarp.N_FEATURES)
    assert np.all(np.isfinite(features))
    assert np.allclose(features, warp.features(params))          # deterministic map
    assert np.all(np.abs(features[:, 2:]) <= 1.0 + 1e-12)        # angle embeddings
    assert np.all(features[:, :2] >= 0.0)                        # arrival times


def test_orientation_features_are_pi_periodic():
    """A retardant rectangle has no head or tail: phi and phi + pi are one drop."""
    env = make_env(wind_response="elliptical")
    warp = FinslerWarp.from_firestate(env, make_firestate())

    a = warp.features(np.array([[12.0, 18.0, 0.4]]))
    b = warp.features(np.array([[12.0, 18.0, 0.4 + np.pi]]))
    assert a == pytest.approx(b, abs=1e-9)


def test_asymmetry_diagnostic_is_signed_by_wind_direction():
    env = make_env(wind=(1.0, 0.0), wind_coeff=0.8, wind_response="elliptical")
    warp = FinslerWarp.from_firestate(env, make_firestate())
    c = 16.0

    downwind, upwind = warp.asymmetry(np.array([[c + 8, c, 0.0], [c - 8, c, 0.0]]))
    assert downwind > 0.0 > upwind


# ---------------------------------------------------------------------------
# Integration with the optimiser
# ---------------------------------------------------------------------------


def make_optimizer(n_drones=1, wind_response="elliptical"):
    env = make_env(wind_response=wind_response)
    return RetardantDropBayesOptSR(
        CAFireModel(env, seed=0),
        make_firestate(),
        n_drones=n_drones,
        evolution_time_s=600.0,
        n_sims=8,
    )


def test_finsler_frame_produces_six_features_per_drone():
    optimizer = make_optimizer(n_drones=2)
    optimizer.setup_search_grid_sr(K=40, n_r=20, smooth_iters=10)

    features = optimizer._gp_features(np.full(optimizer.dim, 0.5), "finsler")

    assert features.shape == (2 * FinslerWarp.N_FEATURES,)
    assert optimizer._gp_feature_dim("finsler") == features.size
    assert optimizer._gp_feature_dim("wind") == 4 * 2
    assert np.all(np.isfinite(features))


def test_finsler_frame_is_built_lazily_and_reused():
    optimizer = make_optimizer()
    optimizer.setup_search_grid_sr(K=40, n_r=20, smooth_iters=10)
    assert optimizer.finsler_warp is None

    optimizer._gp_features(np.full(optimizer.dim, 0.5), "finsler")
    warp = optimizer.finsler_warp
    assert warp is not None

    optimizer._gp_features(np.full(optimizer.dim, 0.2), "finsler")
    assert optimizer.finsler_warp is warp


def test_native_finsler_search_decodes_without_an_sr_front():
    optimizer = make_optimizer()
    search = optimizer.setup_search_grid_finsler()
    optimizer.search_frame = "finsler"

    params = optimizer.decode_theta(np.array([0.25, 0.6, 0.4]))

    assert isinstance(search, FinslerSearchMap)
    assert optimizer.sr_grid is None
    assert optimizer.final_search_firestate is None
    assert search.describe()["setup_rollout_steps"] == 0
    assert params.shape == (1, 3)
    assert search.mask[int(params[0, 0]), int(params[0, 1])]


def test_native_finsler_bo_never_builds_the_monte_carlo_front(monkeypatch):
    optimizer = make_optimizer()

    def fail_if_called(**kwargs):
        raise AssertionError("SR setup was charged to a supposedly front-free arm")

    monkeypatch.setattr(optimizer, "setup_search_grid_sr", fail_if_called)
    result = optimizer.run_bayes_opt(
        n_init=2,
        n_iters=1,
        n_candidates=16,
        verbose=False,
        kernel_frame="finsler",
        search_frame="finsler",
        eval_seed=3,
    )

    assert optimizer.sr_grid is None
    assert optimizer.finsler_search_map is not None
    assert np.isfinite(result[2])


def test_corrected_sr_orientation_removes_the_duplicate_half_turn():
    optimizer = make_optimizer()
    optimizer.setup_search_grid_sr(K=40, n_r=20, smooth_iters=10)
    theta_a = np.array([0.3, 0.5, 0.2])
    theta_b = np.array([0.3, 0.5, 1.2])

    # In the corrected parameterisation, adding one normalised period gives the
    # same rectangle and the same GP features. The benchmark enables this for
    # both arms so Finsler cannot win by exploiting an avoidable SR redundancy.
    optimizer.orientation_period_pi = True
    params_a = optimizer.decode_theta(theta_a)
    params_b = optimizer.decode_theta(theta_b)
    assert params_a[:, :2] == pytest.approx(params_b[:, :2], abs=1e-12)
    assert np.mod(params_a[:, 2] - params_b[:, 2], np.pi) == pytest.approx(
        0.0, abs=1e-12
    )
    assert optimizer.theta_to_gp_features(theta_a) == pytest.approx(
        optimizer.theta_to_gp_features(theta_b), abs=1e-12
    )


def test_unknown_kernel_frame_is_rejected():
    optimizer = make_optimizer()
    with pytest.raises(ValueError, match="finsler"):
        optimizer._gp_features(np.zeros(optimizer.dim), "riemannian")


# ---------------------------------------------------------------------------
# The CA's wind response
# ---------------------------------------------------------------------------


def test_clipped_wind_response_remains_the_default():
    """Existing results must not move: the elliptical law is opt-in."""
    assert make_env().wind_response == "clipped"
    assert FireEnv.wind_response == "clipped"


def test_elliptical_response_gives_richards_ordering():
    """head > flank > back, with the flank narrowed -- an ellipse, not a bulge.

    The clipped law leaves flank and back identical at the no-wind rate, so its
    length-to-breadth ratio is capped and its profile is not elliptical. The
    elliptical law narrows the flank to sqrt(1 - (c|w|)^2), which is the ellipse
    being stretched along the wind.
    """
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])       # head, flank, back
    c, ros = 0.6, 0.7
    clipped = ca_directional_speed(make_env(wind_coeff=c), dirs)[:, 5, 5]
    elliptical = ca_directional_speed(
        make_env(wind_coeff=c, wind_response="elliptical"), dirs
    )[:, 5, 5]

    assert clipped[1] == pytest.approx(clipped[2])                # flank == back: not an ellipse
    assert elliptical[0] == pytest.approx(clipped[0])             # head rate preserved
    assert elliptical[2] < elliptical[1] < elliptical[0]          # Richards' ordering
    assert elliptical[1] == pytest.approx(ros * np.sqrt(1.0 - c ** 2))
    assert elliptical[2] == pytest.approx(ros * (1.0 - c))
    assert elliptical[0] / elliptical[1] > clipped[0] / clipped[1]   # higher length-to-breadth


def test_invalid_wind_response_is_rejected():
    with pytest.raises(ValueError, match="wind_response"):
        ca_directional_speed(make_env(wind_response="parabolic"), uniform_directions(4))
