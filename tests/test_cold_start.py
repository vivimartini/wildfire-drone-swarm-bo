"""Accounting checks for the equal-budget cold-start benchmark."""

import numpy as np
import pytest

from fire_model.cold_start import (
    _aggregate,
    build_early_detection_scenarios,
    crossover_predictions_held,
    interpolated_crossover,
    rollout_steps,
)


def test_rollout_steps_is_the_shared_cost_unit():
    assert rollout_steps(n_sims=8, horizon_s=600.0, dt_s=30.0) == 160
    assert rollout_steps(n_sims=8, horizon_s=601.0, dt_s=30.0) == 168


def test_early_detection_scenarios_change_geometry_not_budget():
    scenarios = build_early_detection_scenarios(24)
    assert set(scenarios) == {"head_asset", "flank_asset", "diagonal_asset"}

    costs = {
        rollout_steps(8, 600.0, model.env.dt_s)
        for model, _ in scenarios.values()
    }
    assert costs == {160}
    assert all(model.env.wind_response == "elliptical" for model, _ in scenarios.values())


def test_env_overrides_change_knobs_not_geometry():
    scenarios = build_early_detection_scenarios(
        24,
        wind_coeff=0.95,
        ros_future_jitter_frac=0.60,
        wind_coeff_future_jitter_frac=0.50,
    )
    for model, _ in scenarios.values():
        assert model.env.wind_coeff == 0.95
        assert model.env.ros_future_jitter_frac == 0.60
        assert model.env.wind_coeff_future_jitter_frac == 0.50
        assert model.env.wind_response == "elliptical"
    costs = {
        rollout_steps(8, 600.0, model.env.dt_s)
        for model, _ in scenarios.values()
    }
    assert costs == {160}


def test_interpolated_crossover_linear_zero_crossing():
    # Advantage +4 at budget 3, -2 at budget 12: zero at 9.
    assert interpolated_crossover((3, 12), [4.0, -2.0]) == 9.0


def test_interpolated_crossover_never_and_already_behind():
    assert interpolated_crossover((3, 4, 12), [2.0, 1.0, 0.5]) == float("inf")
    assert interpolated_crossover((3, 12), [-1.0, -2.0]) == 3.0
    assert interpolated_crossover((3, 6, 12), [1.0, 0.0, -1.0]) == 6.0


def test_interpolated_crossover_rejects_bad_inputs():
    with pytest.raises(ValueError):
        interpolated_crossover((3,), [1.0])
    with pytest.raises(ValueError):
        interpolated_crossover((3, 3), [1.0, -1.0])


def test_crossover_predictions_held_matches_the_stated_mechanism():
    held = crossover_predictions_held(
        {"baseline": 8.0, "high_stochasticity": 5.0, "high_anisotropy": 11.0}
    )
    assert held["predicted_order_held"] is True
    failed = crossover_predictions_held(
        {"baseline": 8.0, "high_stochasticity": 9.0, "high_anisotropy": 11.0}
    )
    assert failed["high_stochasticity_moved_earlier"] is False
    assert failed["predicted_order_held"] is False
    later_failed = crossover_predictions_held(
        {"baseline": 8.0, "high_stochasticity": 5.0, "high_anisotropy": 7.0}
    )
    assert later_failed["high_anisotropy_moved_later"] is False
    never_later = crossover_predictions_held(
        {
            "baseline": 8.0,
            "high_stochasticity": 4.0,
            "high_anisotropy": float("inf"),
        }
    )
    assert never_later["predicted_order_held"] is True


def test_mechanism_plot_accepts_serialised_crossovers(tmp_path):
    from fire_model.cold_start import _plot_mechanism

    summary = {
        "conditions": {
            "high_stochasticity": {
                "label": "higher stochasticity",
                "crossover_budget": 5.0,
                "crossover_never": False,
                "advantage_curve": {"3": 2.0, "12": -1.0},
            },
            "baseline": {
                "label": "baseline",
                "crossover_budget": 8.0,
                "crossover_never": False,
                "advantage_curve": {"3": 3.0, "12": -1.0},
            },
            "high_anisotropy": {
                "label": "higher anisotropy",
                "crossover_budget": None,
                "crossover_never": True,
                "advantage_curve": {"3": 4.0, "12": 1.0},
            },
        },
        "predictions": {"predicted_order_held": True},
    }
    path = tmp_path / "crossover_mechanism.png"
    _plot_mechanism(summary, path)
    assert path.is_file()


def test_aggregate_uses_paired_relative_improvement():
    def finsler_arm(score):
        return {"curve": {"3": {"validated_cvar": score}}}

    def sr_arm(score):
        return {
            "curve": {"3": {"validated_cvar": score}},
            "free_setup_curve": {"3": {"validated_cvar": score}},
        }

    raw = {
        "a": {
            "seeds": [
                {
                    "seed": 0,
                    "no_drop_cvar": 20.0,
                    "finsler": finsler_arm(10.0),
                    "sr": sr_arm(12.0),
                },
                {
                    "seed": 1,
                    "no_drop_cvar": 20.0,
                    "finsler": finsler_arm(15.0),
                    "sr": sr_arm(15.0),
                },
            ],
        }
    }
    result = _aggregate(raw, (3,))["3"]

    assert result["n_pairs"] == 2
    assert result["finsler_paired_wins"] == 1
    assert result["ties"] == 1
    assert result["finsler_mean_advantage_percentage_points"] == 5.0
    assert np.isclose(result["finsler_mean_improvement_percent"], 37.5)
