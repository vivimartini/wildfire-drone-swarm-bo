"""Accounting checks for the equal-budget cold-start benchmark."""

import numpy as np

from fire_model.cold_start import _aggregate, build_early_detection_scenarios, rollout_steps


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
