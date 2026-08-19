"""Deterministic, one-command reproduction of the repository's core method."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from fire_model.bo_sr import RetardantDropBayesOptSR
from fire_model.ca import CAFireModel, FireEnv, FireState


def build_scenario(nx: int = 36) -> tuple[CAFireModel, FireState]:
    """Construct a small synthetic asset-protection scenario."""
    fuel = np.ones((nx, nx))
    value = np.ones((nx, nx))
    value[int(0.70 * nx) : int(0.88 * nx), int(0.38 * nx) : int(0.62 * nx)] = 12.0
    wind = np.zeros((nx, nx, 2))
    wind[..., 0] = 1.0
    env = FireEnv(
        grid_size=(nx, nx),
        domain_km=4.0,
        fuel=fuel,
        value=value,
        wind=wind,
        dt_s=30.0,
        burn_time_s0=600.0,
        drop_w_km=0.25,
        drop_h_km=1.0,
        drop_amount=2.0,
        ros_mps=0.7,
        wind_coeff=0.8,
        ros_future_jitter_frac=0.30,
        wind_coeff_future_jitter_frac=0.25,
    )
    burning = np.zeros((1, nx, nx), dtype=bool)
    cx, cy = int(0.35 * nx), nx // 2
    burning[0, cx - 1 : cx + 2, cy - 1 : cy + 2] = True
    state = FireState(
        burning=burning,
        burned=np.zeros_like(burning),
        burn_remaining_s=np.full((1, nx, nx), env.burn_time_s0),
        retardant=np.zeros((1, nx, nx)),
        t=0,
    )
    return CAFireModel(env, seed=0), state


def run_reproduction(output_dir: Path, quick: bool = False) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model, state = build_scenario(nx=32 if quick else 40)
    high_sims = 24 if quick else 96
    low_sims = 8 if quick else 24
    high_horizon = 600.0 if quick else 900.0
    low_horizon = high_horizon / 2.0

    optimizer = RetardantDropBayesOptSR(
        model,
        state,
        n_drones=1,
        evolution_time_s=high_horizon,
        n_sims=high_sims,
        risk_measure="cvar",
        cvar_alpha=0.90,
        rng=np.random.default_rng(11),
    )

    result = optimizer.run_bayes_opt_mf(
        n_init_high=3 if quick else 6,
        n_init_low=3 if quick else 8,
        n_iters=4 if quick else 12,
        n_candidates=96 if quick else 750,
        K_grid=60 if quick else 160,
        n_r=30 if quick else 80,
        smooth_iters=30 if quick else 120,
        verbose=False,
        kernel_frame="wind",
        eval_seed=17,
        low_n_sims=low_sims,
        low_evolution_time_s=low_horizon,
        low_scale_params=False,
        mf_warmup_low=4 if quick else 10,
        mf_low_per_high=1,
        mf_max_low=4 if quick else 14,
        mf_return_history=True,
    )
    best_theta, best_params, selection_cvar, _, _, _, trace = result

    # Validate the selected plan and no-drop baseline on a larger, independent
    # Monte-Carlo batch. This keeps the reported result separate from BO's data.
    validation_sims = 64 if quick else 256
    validation_seed = 101
    validation_losses = {}
    for name, params in (("no_drop", None), ("selected_plan", best_params)):
        batch = model.simulate_from_firestate(
            state,
            T=high_horizon,
            n_sims=validation_sims,
            drone_params=params,
            ros_mps=model.env.ros_mps,
            wind_coeff=model.env.wind_coeff,
            diag=model.env.diag,
            seed=validation_seed,
            avoid_burning_drop=model.env.avoid_burning_drop,
            burning_prob_threshold=model.env.avoid_drop_p_threshold,
            return_batch=True,
        )
        validation_losses[name] = optimizer._losses_from_batch_firestate(batch, model.env)
    baseline_cvar = optimizer.cvar(validation_losses["no_drop"], optimizer.cvar_alpha)
    validated_cvar = optimizer.cvar(validation_losses["selected_plan"], optimizer.cvar_alpha)
    improvement = 100.0 * (baseline_cvar - validated_cvar) / max(abs(baseline_cvar), 1e-12)

    summary = {
        "objective": "CVaR_0.90 of asset-value loss",
        "kernel": "anisotropic Matérn on wind-rotated position/orientation features",
        "high_fidelity": {"simulations": high_sims, "horizon_s": high_horizon},
        "low_fidelity": {"simulations": low_sims, "horizon_s": low_horizon},
        "validation": {"simulations": validation_sims, "seed": validation_seed},
        "baseline_cvar": baseline_cvar,
        "selected_plan_cvar": validated_cvar,
        "selection_estimate_cvar": selection_cvar,
        "improvement_percent": improvement,
        "best_theta_sr": np.asarray(best_theta).tolist(),
        "best_drop_xy_phi": np.asarray(best_params).tolist(),
        "fidelity_trace": trace,
        "seed": 17,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    labels = ["No drop", "Selected plan"]
    values = [baseline_cvar, validated_cvar]
    ax.bar(labels, values, color=["tab:gray", "tab:blue"])
    ax.set(ylabel="CVaR$_{0.90}$ asset loss", title="Independent validation of risk-sensitive MFBO")
    fig.tight_layout()
    fig.savefig(output_dir / "cvar_mfbo_reproduction.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the CI-sized reproduction.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reproduction"))
    args = parser.parse_args()
    summary = run_reproduction(args.output_dir, quick=args.quick)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
