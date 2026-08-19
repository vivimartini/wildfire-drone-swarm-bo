"""Equal-budget cold-start benchmark for SR and native Finsler search.

The earlier kernel-frame comparison gave every arm an SR future boundary for
free.  That comparison answers which representation is best *after* a forecast
front exists, but not which planner is best when simulation is scarce.

This benchmark charges every simulator step:

* native Finsler setup: zero rollout steps (deterministic shortest paths);
* SR setup: ``n_sims * ceil(horizon / dt)`` rollout steps;
* one candidate evaluation: the same
  ``n_sims * ceil(horizon / dt)`` rollout steps.

Thus SR spends one evaluation-equivalent before BO starts.  Each arm is run
once to its maximum budget; incumbents at smaller budgets are prefixes of that
same trajectory, avoiding the extra variance caused by rerunning BO separately
for every point on the curve.  Selected plans are scored on an independent,
larger Monte-Carlo batch that is not counted as planning budget because it is
measurement, not information available to either planner.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning

from fire_model.bo_sr import RetardantDropBayesOptSR
from fire_model.ca import CAFireModel, FireState
from fire_model.demo import build_scenario


def build_early_detection_scenarios(nx: int) -> dict[str, tuple[CAFireModel, FireState]]:
    """Three small fires with different asset/wind geometry."""
    base_model, state = build_scenario(nx=nx)
    base = base_model.env
    scenarios: dict[str, tuple[CAFireModel, FireState]] = {}

    # Asset directly in the heading direction.
    scenarios["head_asset"] = (
        CAFireModel(replace(base, wind_response="elliptical"), seed=0),
        state,
    )

    # Asset on the flank: the planner must decide whether to spend scarce
    # evaluations near value or near the fastest-moving part of the front.
    flank_value = np.ones((nx, nx))
    flank_value[
        int(0.38 * nx) : int(0.58 * nx),
        int(0.72 * nx) : int(0.91 * nx),
    ] = 12.0
    scenarios["flank_asset"] = (
        CAFireModel(
            replace(base, value=flank_value, wind_response="elliptical"),
            seed=0,
        ),
        state,
    )

    # A diagonal wind removes any accidental advantage from grid alignment.
    diagonal_wind = np.zeros((nx, nx, 2))
    diagonal_wind[..., 0] = 1.0 / np.sqrt(2.0)
    diagonal_wind[..., 1] = 1.0 / np.sqrt(2.0)
    diagonal_value = np.ones((nx, nx))
    diagonal_value[
        int(0.66 * nx) : int(0.87 * nx),
        int(0.66 * nx) : int(0.87 * nx),
    ] = 12.0
    scenarios["diagonal_asset"] = (
        CAFireModel(
            replace(
                base,
                wind=diagonal_wind,
                value=diagonal_value,
                wind_response="elliptical",
            ),
            seed=0,
        ),
        state,
    )
    return scenarios


def rollout_steps(n_sims: int, horizon_s: float, dt_s: float) -> int:
    """Common planning-cost unit: simulated realisation-steps."""
    return int(n_sims) * int(np.ceil(float(horizon_s) / float(dt_s)))


def _validation_cvar(
    model: CAFireModel,
    state: FireState,
    params: np.ndarray | None,
    *,
    horizon_s: float,
    n_sims: int,
    seed: int,
    alpha: float = 0.90,
) -> float:
    env = model.env
    # CAFireModel.step_batch derives its ignition stream from base_seed, while
    # simulate_from_firestate's seed controls parameter jitter. Re-instantiating
    # here makes validation ensembles genuinely independent across planning
    # seeds; both arms still share each seed as paired common random numbers.
    validation_model = CAFireModel(env, seed=seed)
    batch = validation_model.simulate_from_firestate(
        state,
        T=horizon_s,
        n_sims=n_sims,
        drone_params=params,
        ros_mps=env.ros_mps,
        wind_coeff=env.wind_coeff,
        diag=env.diag,
        seed=seed,
        avoid_burning_drop=env.avoid_burning_drop,
        burning_prob_threshold=env.avoid_drop_p_threshold,
        return_batch=True,
    )
    losses = RetardantDropBayesOptSR._losses_from_batch_firestate(batch, env)
    return RetardantDropBayesOptSR.cvar(losses, alpha)


def _run_arm(
    model: CAFireModel,
    state: FireState,
    *,
    frame: str,
    budgets: tuple[int, ...],
    n_init: int,
    n_sims: int,
    horizon_s: float,
    validation_sims: int,
    seed: int,
    validation_seed: int,
    n_candidates: int,
) -> dict:
    """Run one BO trajectory and independently validate every budget prefix."""
    setup_equivalents = 0 if frame == "finsler" else 1
    # Run SR one step farther than its charged curve needs so the same
    # trajectory can also reproduce the old, unpriced comparison.
    max_evaluations = max(budgets)
    if max_evaluations < n_init:
        raise ValueError("largest budget cannot pay for setup and initial design")

    optimizer = RetardantDropBayesOptSR(
        model,
        state,
        n_drones=1,
        evolution_time_s=horizon_s,
        n_sims=n_sims,
        risk_measure="cvar",
        cvar_alpha=0.90,
        rng=np.random.default_rng(10_000 + seed),
    )
    setup_started = time.perf_counter()
    if frame == "finsler":
        optimizer.setup_search_grid_finsler()
    else:
        optimizer.setup_search_grid_sr(K=60, n_r=30, smooth_iters=30)
    setup_wall_time_s = time.perf_counter() - setup_started

    result = optimizer.run_bayes_opt(
        n_init=n_init,
        n_iters=max_evaluations - n_init,
        n_candidates=n_candidates,
        K_grid=60,
        n_r=30,
        smooth_iters=30,
        verbose=False,
        kernel_frame=frame,
        search_frame=frame,
        orientation_period_pi=True,
        candidate_strategy="qmc",
        eval_seed=20_000 + seed,
        return_theta_history=True,
    )
    history = result[-1]
    theta_history = history["theta"]
    observed = history["objective"]
    per_eval_steps = rollout_steps(n_sims, horizon_s, model.env.dt_s)

    validation_cache: dict[int, tuple[float, np.ndarray]] = {}

    def score_prefix(available: int) -> tuple[float, np.ndarray, int]:
        incumbent = int(np.argmin(observed[:available]))
        if incumbent not in validation_cache:
            params = optimizer.decode_theta(theta_history[incumbent])
            score = _validation_cvar(
                model,
                state,
                params,
                horizon_s=horizon_s,
                n_sims=validation_sims,
                seed=validation_seed,
            )
            validation_cache[incumbent] = (score, params)
        score, params = validation_cache[incumbent]
        return score, params, incumbent

    curve = {}
    free_setup_curve = {}
    for budget in budgets:
        available = int(budget) - setup_equivalents
        if available < n_init:
            curve[str(budget)] = None
            continue
        score, params, incumbent = score_prefix(available)
        curve[str(budget)] = {
            "validated_cvar": score,
            "planning_evaluations": available,
            "setup_rollout_steps": int(history["setup_rollout_steps"]),
            "evaluation_rollout_steps": int(available * per_eval_steps),
            "total_rollout_steps": int(budget * per_eval_steps),
            "incumbent_theta": theta_history[incumbent].tolist(),
            "incumbent_drop_xy_phi": params.tolist(),
        }
        if frame == "sr":
            free_score, _, _ = score_prefix(int(budget))
            free_setup_curve[str(budget)] = {
                "validated_cvar": free_score,
                "planning_evaluations": int(budget),
                "reported_setup_cost_equivalents": 0,
            }
    return {
        "setup_evaluation_equivalents": setup_equivalents,
        "setup_rollout_steps": int(history["setup_rollout_steps"]),
        "setup_wall_time_s": setup_wall_time_s,
        "curve": curve,
        "free_setup_curve": free_setup_curve,
    }


def _aggregate(raw: dict, budgets: tuple[int, ...]) -> dict:
    """Aggregate paired improvements over scenarios and seeds."""
    out: dict[str, dict] = {}
    for budget in budgets:
        rows = []
        for scenario, record in raw.items():
            for seed_record in record["seeds"]:
                baseline = seed_record["no_drop_cvar"]
                f = seed_record["finsler"]["curve"].get(str(budget))
                s = seed_record["sr"]["curve"].get(str(budget))
                free_s = seed_record["sr"]["free_setup_curve"].get(str(budget))
                if f is None or s is None:
                    continue
                f_improvement = 100.0 * (baseline - f["validated_cvar"]) / max(abs(baseline), 1e-12)
                s_improvement = 100.0 * (baseline - s["validated_cvar"]) / max(abs(baseline), 1e-12)
                free_s_improvement = 100.0 * (
                    baseline - free_s["validated_cvar"]
                ) / max(abs(baseline), 1e-12)
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed_record["seed"],
                        "finsler_improvement_percent": f_improvement,
                        "sr_improvement_percent": s_improvement,
                        "free_setup_sr_improvement_percent": free_s_improvement,
                        "paired_advantage_percentage_points": f_improvement - s_improvement,
                        "advantage_vs_free_setup_sr_percentage_points": (
                            f_improvement - free_s_improvement
                        ),
                    }
                )
        advantage = np.asarray([r["paired_advantage_percentage_points"] for r in rows])
        f_values = np.asarray([r["finsler_improvement_percent"] for r in rows])
        s_values = np.asarray([r["sr_improvement_percent"] for r in rows])
        free_s_values = np.asarray(
            [r["free_setup_sr_improvement_percent"] for r in rows]
        )
        free_advantage = np.asarray(
            [r["advantage_vs_free_setup_sr_percentage_points"] for r in rows]
        )
        advantage_se = float(advantage.std(ddof=1) / np.sqrt(advantage.size))
        wins = int(np.sum(advantage > 0.0))
        free_wins = int(np.sum(free_advantage > 0.0))
        ties = int(np.sum(np.isclose(advantage, 0.0)))
        free_ties = int(np.sum(np.isclose(free_advantage, 0.0)))
        sign_n = int(advantage.size - ties)
        free_sign_n = int(free_advantage.size - free_ties)
        out[str(budget)] = {
            "n_pairs": len(rows),
            "finsler_mean_improvement_percent": float(f_values.mean()),
            "sr_mean_improvement_percent": float(s_values.mean()),
            "free_setup_sr_mean_improvement_percent": float(free_s_values.mean()),
            "finsler_mean_advantage_percentage_points": float(advantage.mean()),
            "finsler_mean_advantage_ci95_percentage_points": [
                float(advantage.mean() - 1.96 * advantage_se),
                float(advantage.mean() + 1.96 * advantage_se),
            ],
            "finsler_median_advantage_percentage_points": float(np.median(advantage)),
            "finsler_paired_wins": wins,
            "paired_sign_test_p": float(binomtest(wins, sign_n, 0.5).pvalue),
            "finsler_wins_vs_free_setup_sr": free_wins,
            "free_setup_sr_sign_test_p": float(
                binomtest(free_wins, free_sign_n, 0.5).pvalue
            ),
            "ties": ties,
            "by_scenario": {
                scenario: {
                    "n_pairs": len(group),
                    "mean_advantage_percentage_points": float(
                        np.mean(
                            [
                                row["paired_advantage_percentage_points"]
                                for row in group
                            ]
                        )
                    ),
                    "finsler_wins": int(
                        np.sum(
                            [
                                row["paired_advantage_percentage_points"] > 0.0
                                for row in group
                            ]
                        )
                    ),
                }
                for scenario in sorted({row["scenario"] for row in rows})
                for group in [[row for row in rows if row["scenario"] == scenario]]
            },
            "rows": rows,
        }
    return out


def _plot(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    budgets = [int(b) for b in summary["aggregate"]]
    f = [summary["aggregate"][str(b)]["finsler_mean_improvement_percent"] for b in budgets]
    s = [summary["aggregate"][str(b)]["sr_mean_improvement_percent"] for b in budgets]
    free_s = [
        summary["aggregate"][str(b)]["free_setup_sr_mean_improvement_percent"]
        for b in budgets
    ]
    wins = [
        summary["aggregate"][str(b)]["finsler_paired_wins"]
        / summary["aggregate"][str(b)]["n_pairs"]
        for b in budgets
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(budgets, f, "o-", label="native Finsler (zero-rollout setup)")
    axes[0].plot(budgets, s, "s-", label="SR (front rollout charged)")
    axes[0].plot(
        budgets,
        free_s,
        "^--",
        color="tab:gray",
        label="SR if setup is incorrectly free",
    )
    axes[0].set(
        xlabel="Total planning budget (candidate-evaluation equivalents)",
        ylabel="Independent CVaR improvement vs no drop (%)",
        title="Equal-budget cold-start performance",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(budgets, np.asarray(wins) * 100.0, "o-", color="tab:purple")
    axes[1].axhline(50.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Total planning budget (candidate-evaluation equivalents)",
        ylabel="Paired scenarios/seeds won by Finsler (%)",
        ylim=(0, 100),
        title="How often the front-free prior wins",
    )
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_cold_start(output_dir: Path, *, quick: bool = False) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    nx = 26 if quick else 32
    n_seeds = 3 if quick else 12
    n_sims = 8 if quick else 12
    validation_sims = 48 if quick else 128
    n_candidates = 64 if quick else 128
    horizon_s = 600.0
    budgets = (3, 4, 6, 8, 12)
    n_init = 2

    raw = {}
    for scenario_index, (name, (model, state)) in enumerate(
        build_early_detection_scenarios(nx).items()
    ):
        seeds = []
        for seed in range(n_seeds):
            # Each planning seed gets an independent validation ensemble, while
            # the two arms share it as common random numbers. This yields 36
            # independent paired measurements in the full benchmark rather
            # than repeatedly scoring every plan on the same three ensembles.
            validation_seed = 90_000 + 1_000 * scenario_index + seed
            baseline = _validation_cvar(
                model,
                state,
                None,
                horizon_s=horizon_s,
                n_sims=validation_sims,
                seed=validation_seed,
            )
            seed_record = {
                "seed": seed,
                "validation_seed": validation_seed,
                "no_drop_cvar": baseline,
            }
            for frame in ("finsler", "sr"):
                seed_record[frame] = _run_arm(
                    model,
                    state,
                    frame=frame,
                    budgets=budgets,
                    n_init=n_init,
                    n_sims=n_sims,
                    horizon_s=horizon_s,
                    validation_sims=validation_sims,
                    seed=seed,
                    validation_seed=validation_seed,
                    n_candidates=n_candidates,
                )
            seeds.append(seed_record)
        raw[name] = {
            "mean_no_drop_cvar": float(
                np.mean([seed_record["no_drop_cvar"] for seed_record in seeds])
            ),
            "seeds": seeds,
        }

    summary = {
        "protocol": {
            "scenarios": list(raw),
            "grid_size": nx,
            "planning_seeds_per_scenario": n_seeds,
            "planning_sims_per_evaluation": n_sims,
            "validation_sims": validation_sims,
            "horizon_s": horizon_s,
            "budgets_evaluation_equivalents": list(budgets),
            "sr_setup_cost_equivalents": 1,
            "finsler_setup_cost_equivalents": 0,
            "validation_is_not_planning_information": True,
        },
        "aggregate": _aggregate(raw, budgets),
        "raw": raw,
    }
    for frame in ("finsler", "sr"):
        setup_times = [
            seed_record[frame]["setup_wall_time_s"]
            for scenario in raw.values()
            for seed_record in scenario["seeds"]
        ]
        summary["protocol"][f"{frame}_median_setup_wall_time_s"] = float(
            np.median(setup_times)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cold_start_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(summary, output_dir / "cold_start_budget.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cold_start"))
    args = parser.parse_args()
    matplotlib.use("Agg")
    summary = run_cold_start(args.output_dir, quick=args.quick)
    compact = {
        budget: {
            key: value
            for key, value in record.items()
            if key != "rows"
        }
        for budget, record in summary["aggregate"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
