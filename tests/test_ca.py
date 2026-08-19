"""Smoke tests for the cellular-automata simulator."""

import numpy as np

from fire_model.ca import CAFireModel, FireEnv, FireState


def make_env(nx=28, wind=(1.0, 0.0)):
    return FireEnv(
        grid_size=(nx, nx),
        domain_km=4.0,
        fuel=np.ones((nx, nx)),
        value=np.ones((nx, nx)),
        wind=np.tile(np.asarray(wind, dtype=float), (nx, nx, 1)),
        dt_s=30.0,
        ros_mps=0.6,
        wind_coeff=0.6,
    )


def make_state(nx=28):
    burning = np.zeros((1, nx, nx), dtype=bool)
    burning[0, nx // 2 - 1 : nx // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = True
    return FireState(
        burning=burning,
        burned=np.zeros_like(burning),
        burn_remaining_s=np.full((1, nx, nx), 600.0),
        retardant=np.zeros((1, nx, nx)),
        t=0,
    )


def test_simulation_is_reproducible_and_preserves_batch():
    env, state = make_env(), make_state()
    model = CAFireModel(env, seed=0)
    batch = model.simulate_from_firestate(state, T=600.0, n_sims=24, seed=7, return_batch=True)
    aggregate = model.simulate_from_firestate(state, T=600.0, n_sims=24, seed=7)

    assert batch.burning.shape == (24, 28, 28)
    assert np.allclose(batch.burning.mean(axis=0), aggregate.burning[0])
    assert np.allclose(batch.burned.mean(axis=0), aggregate.burned[0])


def test_fire_spreads_further_downwind():
    nx = 28
    model = CAFireModel(make_env(nx), seed=0)
    out = model.simulate_from_firestate(make_state(nx), T=1200.0, n_sims=32, seed=1)
    affected = np.clip(out.burning[0] + out.burned[0], 0.0, 1.0)

    assert affected[nx // 2 :, :].sum() > affected[: nx // 2, :].sum()
