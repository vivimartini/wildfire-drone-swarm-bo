"""Does the Randers metric actually describe this simulator's fire?

Two questions are answered by running this module, and neither is answered by
asserting the geometry is right:

1. **Is the fitted metric predictive?** The CA is a stochastic percolation
   process; the Randers metric is fitted to its *mean-field* directional rate.
   Those need not agree, so simulated first-ignition times are regressed onto
   the geodesic arrival times and the calibration factor and fit quality are
   reported -- alongside the same numbers for an isotropic metric, which is the
   ablation that says whether the one-form earns its place.

2. **Is the warp necessary?** The directed Finsler distance is put into a Matern
   profile, symmetrised the two tempting ways, and the minimum eigenvalue of the
   resulting Gram matrix is reported next to the warp's. Substituting a
   directed distance does not give a symmetric matrix; symmetrising it gives no
   positive-definiteness guarantee and, on this fire, actually fails.

Run with ``python -m fire_model.finsler_validation [--quick]``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

from fire_model.ca import CAFireModel, FireState
from fire_model.finsler import (
    FinslerWarp,
    RandersField,
    arrival_time,
    directed_distance_matrix,
    naive_symmetrised_gram,
    randers_from_env,
    uniform_directions,
)


# ---------------------------------------------------------------------------
# Simulated arrival times
# ---------------------------------------------------------------------------


def simulate_arrival_times(
    model: CAFireModel,
    init_firestate: FireState,
    *,
    T: float,
    n_sims: int,
    seed: int = 0,
) -> np.ndarray:
    """Mean first-ignition time per cell over ``n_sims`` CA realisations.

    Spread parameters are held at their base values (no jitter), because the
    question here is whether the *geometry* matches, not how it degrades under
    parameter uncertainty. Cells that never ignite in a realisation are left out
    of that cell's mean; cells that never ignite in any realisation are ``nan``.
    """
    env = model.env
    nx, ny = env.grid_size
    dt = float(env.dt_s)
    n_steps = int(np.ceil(float(T) / dt))

    burning0 = np.asarray(init_firestate.burning)
    burned0 = np.asarray(init_firestate.burned)
    if burning0.ndim == 2:
        burning0, burned0 = burning0[None], burned0[None]

    state = FireState(
        burning=np.repeat(burning0.astype(bool), n_sims, axis=0),
        burned=np.repeat(burned0.astype(bool), n_sims, axis=0),
        burn_remaining_s=np.full((n_sims, nx, ny), float(env.burn_time_s0)),
        retardant=np.zeros((n_sims, nx, ny)),
        t=int(init_firestate.t),
    )

    model = CAFireModel(env, seed=seed)
    seen = state.burning | state.burned
    arrival = np.full((n_sims, nx, ny), np.nan)
    arrival[seen] = 0.0

    ros = np.full(n_sims, float(np.asarray(env.ros_mps).ravel()[0]))
    wind_c = np.full(n_sims, float(env.wind_coeff))

    for step in range(1, n_steps + 1):
        model.step_batch(state, ros_mps=ros, wind_coeff=wind_c, diag=env.diag)
        now = state.burning | state.burned
        fresh = now & ~seen
        arrival[fresh] = step * dt
        seen |= fresh

    ever = np.any(np.isfinite(arrival), axis=0)
    mean = np.full(arrival.shape[1:], np.nan)
    with np.errstate(invalid="ignore"):
        mean[ever] = np.nanmean(arrival[:, ever], axis=0)
    return mean


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricFit:
    """Regression of simulated arrival time onto a predicted travel-time field."""

    name: str
    calibration: float          # kappa in T_sim ~ kappa * T_pred
    r2: float                   # about that one-parameter model
    spearman: float
    n_cells: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "calibration_factor": self.calibration,
            "r2_through_origin": self.r2,
            "spearman_rho": self.spearman,
            "n_cells": self.n_cells,
        }


def compare_arrival_fields(name: str, predicted: np.ndarray, simulated: np.ndarray) -> MetricFit:
    """Fit ``T_sim ~ kappa * T_pred`` and score it.

    The model is deliberately through the origin and single-parameter: arrival
    time at the ignition set is zero by construction, so an intercept would only
    let a wrong geometry hide behind an offset.
    """
    ok = np.isfinite(predicted) & np.isfinite(simulated) & (predicted > 0.0)
    x = predicted[ok].ravel()
    y = simulated[ok].ravel()
    if x.size < 3:
        return MetricFit(name, float("nan"), float("nan"), float("nan"), int(x.size))

    kappa = float(np.dot(x, y) / np.dot(x, x))
    resid = y - kappa * x
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(x, y).statistic)
    return MetricFit(name, kappa, r2, rho, int(x.size))


def positive_definiteness_report(
    field: RandersField,
    warp: FinslerWarp,
    points: np.ndarray,
    *,
    length_scale: float | None = None,
) -> dict:
    """Minimum eigenvalues of substituted vs warped kernel matrices.

    The claim being tested is not "the warp is nicer" but "the substitution is
    not a covariance function". A negative minimum eigenvalue means the implied
    prior has negative variance somewhere, which makes the GP's predictive
    variance meaningless and the marginal likelihood undefined.
    """
    from fire_model.finsler import TiedFinslerMatern

    pts = np.asarray(points, dtype=int)
    D = directed_distance_matrix(field, pts)
    finite = D[np.isfinite(D) & (D > 0)]
    scale = float(np.median(finite)) if length_scale is None else float(length_scale)

    out = {
        "n_points": int(pts.shape[0]),
        "directed_distance_is_symmetric": bool(np.allclose(D, D.T)),
        "max_relative_asymmetry": float(
            np.max(np.abs(D - D.T) / np.maximum(0.5 * (D + D.T), 1e-12))
        ),
        "substituted": {},
    }
    for mode in ("mean", "min", "max"):
        K = naive_symmetrised_gram(D, mode=mode, length_scale=scale)
        w = np.linalg.eigvalsh(K)
        out["substituted"][mode] = {
            "min_eigenvalue": float(w.min()),
            "is_psd": bool(w.min() >= -1e-10),
        }

    # The warp, evaluated on the same points with an arbitrary orientation.
    params = np.column_stack([pts[:, 0].astype(float), pts[:, 1].astype(float), np.zeros(pts.shape[0])])
    X = warp.features(params)
    Kw = TiedFinslerMatern()(X)
    w = np.linalg.eigvalsh(Kw)
    out["warped"] = {
        "min_eigenvalue": float(w.min()),
        "is_psd": bool(w.min() >= -1e-10),
        "is_symmetric": bool(np.allclose(Kw, Kw.T)),
    }
    return out


# ---------------------------------------------------------------------------
# Sweep: does the one-form earn its place, and where does the fit break?
# ---------------------------------------------------------------------------


def sweep_wind_response(
    base_env,
    init_firestate: FireState,
    *,
    wind_coeffs: tuple[float, ...],
    wind_responses: tuple[str, ...],
    horizon_s: float,
    n_sims: int,
    seed: int = 7,
) -> list[dict]:
    """Score the Randers metric and its isotropic ablation across wind strengths.

    The isotropic ablation keeps the same direction-averaged speed and zeroes the
    drift, so the difference between the two rows is attributable to the
    one-form and nothing else.
    """
    from dataclasses import replace

    source = np.asarray(init_firestate.burning)[0] | np.asarray(init_firestate.burned)[0]
    rows = []
    for response in wind_responses:
        for coeff in wind_coeffs:
            env = replace(base_env, wind_coeff=float(coeff), wind_response=response)
            model = CAFireModel(env, seed=0)
            field = randers_from_env(env)
            isotropic = RandersField(
                speed=field.speed,
                drift=np.zeros_like(field.drift),
                dx_m=field.dx_m,
                burnable=field.burnable,
            )
            simulated = simulate_arrival_times(
                model, init_firestate, T=horizon_s, n_sims=n_sims, seed=seed
            )
            fits = {
                "finsler": compare_arrival_fields("finsler", arrival_time(field, source).time_s, simulated),
                "isotropic": compare_arrival_fields("isotropic", arrival_time(isotropic, source).time_s, simulated),
            }
            ratio = field.wind_ratio()[field.burnable]
            rows.append(
                {
                    "wind_response": response,
                    "wind_coeff": float(coeff),
                    "b_norm_a": float(np.max(ratio, initial=0.0)),
                    "metric_fit_residual": float(np.median(field.fit_residual[field.burnable])),
                    "drift_clipped_fraction": float(field.clipped_fraction),
                    "subunit_wind": bool(field.clipped_fraction == 0.0),
                    "finsler": fits["finsler"].as_dict(),
                    "isotropic": fits["isotropic"].as_dict(),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Does the frame help the optimiser?
# ---------------------------------------------------------------------------


def compare_kernel_frames(
    base_env,
    init_firestate: FireState,
    *,
    frames: tuple[str, ...] = ("wind", "sr", "finsler"),
    wind_responses: tuple[str, ...] = ("clipped", "elliptical"),
    n_seeds: int = 5,
    n_init: int = 6,
    n_iters: int = 12,
    n_sims: int = 24,
    horizon_s: float = 600.0,
    validation_sims: int = 128,
    validation_seed: int = 909,
) -> dict:
    """Run BO under each kernel frame and score the winners independently.

    Only the surrogate's feature map changes between arms: the search space, the
    objective, the acquisition, the initial designs and the evaluation seeds are
    all shared, and each seed indexes the same initial design across frames, so
    the comparison is paired. Every selected plan is then re-scored on a larger
    independent Monte-Carlo batch, so the numbers reported are not the ones BO
    optimised against.

    With a handful of seeds this measures a difference of means only crudely;
    the paired win count is reported alongside because it is the more honest
    summary at this sample size.
    """
    from dataclasses import replace

    from fire_model.bo_sr import RetardantDropBayesOptSR

    results: dict[str, dict] = {}
    for response in wind_responses:
        env = replace(base_env, wind_response=response)
        model = CAFireModel(env, seed=0)

        def validate(drone_params) -> float:
            batch = model.simulate_from_firestate(
                init_firestate,
                T=horizon_s,
                n_sims=validation_sims,
                drone_params=drone_params,
                ros_mps=env.ros_mps,
                wind_coeff=env.wind_coeff,
                diag=env.diag,
                seed=validation_seed,
                avoid_burning_drop=env.avoid_burning_drop,
                burning_prob_threshold=env.avoid_drop_p_threshold,
                return_batch=True,
            )
            scorer = RetardantDropBayesOptSR(
                model, init_firestate, n_drones=1, evolution_time_s=horizon_s, n_sims=n_sims
            )
            return float(scorer.cvar(scorer._losses_from_batch_firestate(batch, env), 0.90))

        arm: dict[str, list[float]] = {}
        for frame in frames:
            losses = []
            for seed in range(n_seeds):
                optimizer = RetardantDropBayesOptSR(
                    model,
                    init_firestate,
                    n_drones=1,
                    evolution_time_s=horizon_s,
                    n_sims=n_sims,
                    risk_measure="cvar",
                    cvar_alpha=0.90,
                    rng=np.random.default_rng(100 + seed),
                )
                _, best_params, *_ = optimizer.run_bayes_opt(
                    n_init=n_init,
                    n_iters=n_iters,
                    n_candidates=200,
                    K_grid=60,
                    n_r=30,
                    smooth_iters=30,
                    verbose=False,
                    kernel_frame=frame,
                    eval_seed=1000 + seed,
                )
                losses.append(validate(best_params))
            arm[frame] = losses

        record = {
            "no_drop_cvar": validate(None),
            "per_frame": {
                frame: {
                    "cvar_by_seed": vals,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "best": float(np.min(vals)),
                }
                for frame, vals in arm.items()
            },
            "paired_vs_finsler": {},
        }
        if "finsler" in arm:
            for frame, vals in arm.items():
                if frame == "finsler":
                    continue
                diff = np.asarray(arm["finsler"]) - np.asarray(vals)
                record["paired_vs_finsler"][frame] = {
                    "mean_difference": float(diff.mean()),
                    "finsler_wins": int(np.sum(diff < 0)),
                    "n_seeds": int(diff.size),
                }
        results[response] = record
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_indicatrix(ax, field: RandersField, ix: int, iy: int, scale: float, colour: str) -> None:
    dirs = uniform_directions(180)
    sigma = np.maximum(field.speed[ix, iy] + dirs @ field.drift[ix, iy], 0.0)
    pts = np.column_stack([ix + scale * sigma * dirs[:, 0], iy + scale * sigma * dirs[:, 1]])
    ax.plot(*np.vstack([pts, pts[:1]]).T, color=colour, lw=1.1)
    ax.plot([ix], [iy], ".", color=colour, ms=3)


def make_figure(
    field: RandersField,
    warp: FinslerWarp,
    simulated: np.ndarray,
    fits: list[MetricFit],
    predictions: dict[str, np.ndarray],
    sweep: list[dict],
    pd_report: dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    nx, ny = field.grid_size
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))

    # (a) the metric itself: indicatrices over the arrival-time field.
    ax = axes[0, 0]
    T = np.where(np.isfinite(warp.forward.time_s), warp.forward.time_s, np.nan) / 60.0
    im = ax.imshow(T.T, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="arrival time (min)")
    ax.contour(T.T, levels=8, colors="w", linewidths=0.5, alpha=0.7)
    step = max(nx // 6, 1)
    denom = 3.4 * max(float(field.speed.max() + np.linalg.norm(field.drift, axis=-1).max()), 1e-9)
    for ix in range(step, nx - step + 1, step):
        for iy in range(step, ny - step + 1, step):
            if field.burnable[ix, iy]:
                _plot_indicatrix(ax, field, ix, iy, scale=step / denom, colour="tab:cyan")
    ax.set(title="(a) Randers indicatrices and Finsler arrival time", xlabel="x cell", ylabel="y cell")

    # (b) the asymmetry that forbids substituting d_F into a kernel.
    ax = axes[0, 1]
    fwd, rev = warp.forward.time_s, warp.reverse.time_s
    with np.errstate(invalid="ignore"):
        asym = (rev - fwd) / np.maximum(rev + fwd, 1e-12)
    asym = np.where(warp.valid, asym, np.nan)
    lim = float(np.nanmax(np.abs(asym))) or 1.0
    im = ax.imshow(asym.T, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, label="normalised asymmetry")
    ax.set(
        title=r"(b) $d_F(x,\Gamma)\neq d_F(\Gamma,x)$; peak $=\|b\|_a$",
        xlabel="x cell",
        ylabel="y cell",
    )

    # (c) geodesic prediction vs the stochastic simulator.
    ax = axes[0, 2]
    ok = np.isfinite(simulated)
    colours = {"finsler": "tab:blue", "isotropic": "tab:orange"}
    for fit in fits:
        pred = predictions[fit.name]
        m = ok & np.isfinite(pred) & (pred > 0)
        ax.scatter(
            pred[m].ravel() / 60.0, simulated[m].ravel() / 60.0, s=4, alpha=0.25,
            color=colours.get(fit.name, "tab:green"),
            label=f"{fit.name}: $\\rho$={fit.spearman:.3f}, $R^2$={fit.r2:.3f}",
        )
    hi = float(np.nanmax(simulated[ok])) / 60.0
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
    ax.set(
        xlabel="predicted geodesic arrival time (min)",
        ylabel="simulated mean arrival time (min)",
        title="(c) Geodesic prediction vs stochastic CA",
    )
    ax.legend(fontsize=8, loc="lower right")

    # (d, e) does the one-form earn its place, as anisotropy grows?
    markers = {"clipped": "o--", "elliptical": "s-"}
    for ax, key, label in (
        (axes[1, 0], "spearman_rho", r"Spearman $\rho$ vs simulated arrival"),
        (axes[1, 1], "r2_through_origin", r"$R^2$ (single global calibration)"),
    ):
        for response in sorted({r["wind_response"] for r in sweep}):
            rows = sorted([r for r in sweep if r["wind_response"] == response], key=lambda r: r["b_norm_a"])
            xs = [r["b_norm_a"] for r in rows]
            for metric, colour in (("finsler", "tab:blue"), ("isotropic", "tab:orange")):
                ax.plot(
                    xs, [r[metric][key] for r in rows], markers[response],
                    color=colour, ms=4, lw=1.3,
                    label=f"{metric}, {response}",
                )
        ax.set(xlabel=r"metric anisotropy $\|b\|_a = |W|/s$", ylabel=label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_title("(d) Ordering quality -- what the warp actually needs")
    axes[1, 1].set_title("(e) Calibrated agreement -- a stricter test")
    axes[1, 0].legend(fontsize=7, loc="lower left")

    # (f) why warp instead of substitute, in eigenvalues.
    ax = axes[1, 2]
    labels = [f"substituted\n({mode}-sym.)" for mode in pd_report["substituted"]]
    vals = [rec["min_eigenvalue"] for rec in pd_report["substituted"].values()]
    labels.append("warped\n(this repo)")
    vals.append(pd_report["warped"]["min_eigenvalue"])
    ax.bar(labels, vals, color=["tab:red" if v < -1e-10 else "tab:green" for v in vals])
    ax.axhline(0.0, color="k", lw=1)
    ax.set(ylabel="min eigenvalue of Gram matrix", title="(f) Substitution breaks positive definiteness")
    ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_validation(output_dir: Path, *, quick: bool = False, frames: bool = False) -> dict:
    from dataclasses import replace

    from fire_model.demo import build_scenario

    nx = 32 if quick else 48
    n_sims = 16 if quick else 64
    horizon = 1800.0 if quick else 3000.0
    wind_coeffs = (0.4, 1.6) if quick else (0.2, 0.4, 0.8, 1.6, 3.0)

    # Headline scenario: the CA's elliptical mode, where the metric is not an
    # approximation of the simulator but a restatement of it.
    base_model, state = build_scenario(nx=nx)
    env = replace(base_model.env, wind_response="elliptical")
    model = CAFireModel(env, seed=0)

    field = randers_from_env(env)
    warp = FinslerWarp.from_firestate(env, state, randers=field)
    source = np.asarray(state.burning)[0] | np.asarray(state.burned)[0]
    isotropic = RandersField(
        speed=field.speed, drift=np.zeros_like(field.drift), dx_m=field.dx_m, burnable=field.burnable
    )
    predictions = {
        "finsler": warp.forward.time_s,
        "isotropic": arrival_time(isotropic, source).time_s,
    }
    simulated = simulate_arrival_times(model, state, T=horizon, n_sims=n_sims, seed=7)
    fits = [compare_arrival_fields(name, pred, simulated) for name, pred in predictions.items()]

    sweep = sweep_wind_response(
        base_model.env,
        state,
        wind_coeffs=wind_coeffs,
        wind_responses=("clipped", "elliptical"),
        horizon_s=horizon,
        n_sims=n_sims,
    )

    rng = np.random.default_rng(0)
    burnable_cells = np.argwhere(field.burnable)
    points = burnable_cells[rng.choice(burnable_cells.shape[0], size=16, replace=False)]
    pd_report = positive_definiteness_report(field, warp, points)

    summary = {
        "scenario": {
            "grid": nx,
            "ca_realisations": n_sims,
            "horizon_s": horizon,
            "wind_response": "elliptical",
        },
        "metric": field.describe(),
        "warp": warp.describe(),
        "arrival_time_fits": [f.as_dict() for f in fits],
        "wind_sweep": sweep,
        "positive_definiteness": pd_report,
    }

    if frames:
        summary["kernel_frame_comparison"] = compare_kernel_frames(
            base_model.env,
            state,
            n_seeds=3 if quick else 5,
            n_iters=6 if quick else 12,
            n_sims=12 if quick else 24,
            validation_sims=64 if quick else 128,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finsler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(field, warp, simulated, fits, predictions, sweep, pd_report, output_dir / "finsler_validation.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the CI-sized validation.")
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Also run the (slower) kernel-frame BO comparison.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/finsler"))
    args = parser.parse_args()
    matplotlib.use("Agg")
    summary = run_validation(args.output_dir, quick=args.quick, frames=args.frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
