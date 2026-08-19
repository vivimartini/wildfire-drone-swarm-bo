"""Regression tests for CVaR and the wind-aligned GP representation."""

import numpy as np
import pytest

from fire_model.bo_sr import RetardantDropBayesOptSR
from fire_model.ca import CAFireModel, FireEnv, FireState


def make_optimizer(n_sims=48, nx=32, wind=(1.0, 0.0), risk_measure="cvar"):
    env = FireEnv(
        grid_size=(nx, nx),
        domain_km=4.0,
        fuel=np.ones((nx, nx)),
        value=np.ones((nx, nx)),
        wind=np.tile(np.asarray(wind, dtype=float), (nx, nx, 1)),
        dt_s=30.0,
        ros_mps=0.6,
        wind_coeff=0.6,
        ros_future_jitter_frac=0.35,
        wind_coeff_future_jitter_frac=0.35,
    )
    burning = np.zeros((1, nx, nx), dtype=bool)
    burning[0, nx // 2 - 2 : nx // 2 + 2, nx // 2 - 2 : nx // 2 + 2] = True
    state = FireState(
        burning=burning,
        burned=np.zeros_like(burning),
        burn_remaining_s=np.full((1, nx, nx), 600.0),
        retardant=np.zeros((1, nx, nx)),
        t=0,
    )
    return RetardantDropBayesOptSR(
        CAFireModel(env, seed=0),
        state,
        n_drones=1,
        evolution_time_s=900.0,
        n_sims=n_sims,
        risk_measure=risk_measure,
        cvar_alpha=0.9,
    )


def test_cvar_is_monotone_and_dominates_mean():
    losses = np.arange(1.0, 9.0)
    values = [RetardantDropBayesOptSR.cvar(losses, alpha) for alpha in (0.5, 0.75, 0.9)]

    assert values[0] >= losses.mean()
    assert values == sorted(values)
    assert values[-1] <= losses.max()


def test_default_objective_dispatches_to_cvar(monkeypatch):
    optimizer = make_optimizer()
    monkeypatch.setattr(optimizer, "risk_value_burned_area", lambda theta, **kwargs: 12.5)
    monkeypatch.setattr(optimizer, "expected_value_burned_area", lambda theta, **kwargs: 3.0)

    assert optimizer.risk_measure == "cvar"
    assert optimizer.objective_value(np.zeros(3)) == 12.5


def test_cvar_uses_unaggregated_loss_distribution():
    optimizer = make_optimizer()
    batch = optimizer._simulate_firestate_with_params(
        np.zeros((1, 3)),
        n_sims=optimizer.n_sims,
        evolution_time_s=900.0,
        seed=1,
        return_batch=True,
    )
    losses = optimizer._losses_from_batch_firestate(batch, optimizer.fire_model.env)

    assert losses.shape == (optimizer.n_sims,)
    assert losses.std() > 0.0
    assert optimizer.cvar(losses, 0.9) >= losses.mean()


def test_aggregated_state_is_rejected_for_cvar():
    optimizer = make_optimizer()
    aggregate = optimizer._simulate_firestate_with_params(
        np.zeros((1, 3)),
        n_sims=optimizer.n_sims,
        evolution_time_s=900.0,
        seed=1,
    )
    with pytest.raises(ValueError, match="unaggregated"):
        optimizer._losses_from_batch_firestate(aggregate, optimizer.fire_model.env)


def test_wind_features_rotate_positions_into_wind_frame(monkeypatch):
    optimizer = make_optimizer(wind=(0.0, 1.0))
    # Two cells north of centre, with the drop's long axis pointing north.
    monkeypatch.setattr(
        optimizer,
        "decode_theta",
        lambda theta: np.array([[16.0, 18.0, 0.0]]),
    )

    along, cross, sin_angle, cos_angle = optimizer.theta_to_wind_gp_features(np.zeros(3))

    assert along > 0.0
    assert cross == pytest.approx(0.0, abs=1e-12)
    assert sin_angle == pytest.approx(0.0, abs=1e-12)
    assert cos_angle == pytest.approx(1.0)
