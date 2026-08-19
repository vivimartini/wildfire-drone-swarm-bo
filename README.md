# Wildfire Drone Swarm BO

**A CVaR-targeting multi-fidelity BO planner reduced worst-tail asset loss by
5.7% against no intervention in the reproducible synthetic smoke test.**

The planner combines a stochastic cellular-automata fire simulator,
fire-front-relative interventions and an anisotropic wind-aligned Matérn
surrogate. The headline result is independently evaluated on 64 Monte-Carlo
realisations; it is a software demonstration, not a claim of operational
wildfire effectiveness.

![Independent validation of the CVaR plan](artifacts/reproduction/cvar_mfbo_reproduction.png)

---

## What this repo does

1. **Simulate** wildfire spread on a grid with fuel, wind, slope, and retardant decay.
2. **Parameterise** candidate drops either in Cartesian coordinates `(x, y, φ)` or in an SR fire-front frame `(s, r, δ)` between successive fire boundaries.
3. **Optimise tail risk** using empirical `CVaR_0.90` of per-realisation asset loss rather than averaging the ensemble before optimisation.
4. **Screen cheaply** with short-horizon, low-simulation rollouts before higher-fidelity evaluations in an autoregressive co-kriging BO loop.
5. **Evaluate** on synthetic stress-test scenarios and a Victoria-style semi-realistic environment.

---

## Features

| Area | Details |
|------|---------|
| Fire model | Grid CA with wind, fuel, value maps, time-varying ROS, retardant half-life |
| Drop geometry | Oriented rectangular drops with optional “avoid burning cells” |
| Coordinates | Cartesian BO and fire-front-relative `(s, r, δ)` plans |
| GP kernel | Anisotropic Matérn over positions/orientations rotated into the observed mean-wind frame |
| Optimisation | CVaR objective, expected improvement, heuristic warm-starts, multi-fidelity co-kriging |
| Baselines | Ring / boundary / head–flank / value-blocking heuristics |
| Scenarios | Fuel jumps, wind corridors, asset clusters, realistic Victorian layout |

---

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/vivimartini/wildfire-drone-swarm-bo.git
cd wildfire-drone-swarm-bo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
```

The editable install puts the `fire_model` package on your path so notebooks under `notebooks/` import cleanly. Each notebook also has a short bootstrap cell that locates the repo root if you open it without installing.

---

## Reproduce the headline result

```bash
bash run_all.sh
```

This creates a clean virtual environment, installs pinned dependencies, runs the
tests and reproduces `artifacts/reproduction/summary.json` plus the figure above.
The checked-in quick-run result is:

```text
No-drop CVaR_0.90:       18.5670
Selected-plan CVaR_0.90: 17.5000
Relative reduction:       5.75%
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact fidelity budgets,
seeds, independent validation design and full-sized command. See
[DECISIONS.md](DECISIONS.md) for modelling choices and the evidence boundary.

---

## Repository layout

```
fire_model/                      Core simulation + optimisation library
  ca.py                          Cellular automata fire model (FireEnv, FireState)
  boundary.py                    Fire-front extraction and between-boundary masks
  harmonic.py                    Harmonic strip maps → (s, r) parameterisation
  bo.py                          Bayesian optimisation in (x, y, φ)
  bo_sr.py                       CVaR, wind-aligned SR BO and multi-fidelity BO
  demo.py                        Deterministic command-line reproduction

tests/                           CA, CVaR and wind-frame regression tests
artifacts/reproduction/          Checked-in output from the quick reproduction

notebooks/
  experiments/                   End-to-end BO / MFBO and final experiment runs
  scenarios/                     Stress-test + realistic environment notebooks
  heuristics/                    Heuristic baselines and comparisons
  parameterization/              SR mapping checks and figure notebooks
  archive/                       Older exploratory work

figures/
  report/                        Figures used in the written report
```

---

## Notebook guide

| Goal | Notebook |
|------|----------|
| Realistic end-to-end BO | [`notebooks/experiments/end_to_end_realistic_BO.ipynb`](notebooks/experiments/end_to_end_realistic_BO.ipynb) |
| Small-grid SR BO walkthrough | [`notebooks/experiments/end_to_end_polar_BO_small.ipynb`](notebooks/experiments/end_to_end_polar_BO_small.ipynb) |
| Final experiments | [`notebooks/experiments/FINAL_EXP.ipynb`](notebooks/experiments/FINAL_EXP.ipynb) |
| Iterative / MFBO finals | [`notebooks/experiments/FINAL_EXP_ITERATIVE_MFBO.ipynb`](notebooks/experiments/FINAL_EXP_ITERATIVE_MFBO.ipynb) |
| Arrival delay vs swarm size | [`notebooks/experiments/experiment_early_detection_vs_drones.ipynb`](notebooks/experiments/experiment_early_detection_vs_drones.ipynb) |
| Scenario suite + report figures | [`notebooks/scenarios/scenarios_final_all_BO_report.ipynb`](notebooks/scenarios/scenarios_final_all_BO_report.ipynb) |
| Heuristic baselines | [`notebooks/heuristics/showcasing_heuristics.ipynb`](notebooks/heuristics/showcasing_heuristics.ipynb) |
| SR coordinate system | [`notebooks/parameterization/fire_front_sr_parameterization.ipynb`](notebooks/parameterization/fire_front_sr_parameterization.ipynb) |

---

## Method sketch

**Simulation.** Each cell burns for a finite time; spread rate depends on fuel, wind, and optional slope. Retardant deposited by drones decays over time and slows (or blocks) ignition.

**Search space.** Drops are rectangles with position and orientation. In the SR formulation, candidates live in the strip between the fire front at detection and a predicted later front, which keeps the search focused on actionable ground.

**Risk objective.** Every candidate retains its unaggregated Monte-Carlo loss
vector. BO minimises the mean of the worst 10% of losses (`CVaR_0.90`), so the
tail—not expected damage—drives plan selection.

**Wind-aligned surrogate.** Cartesian drop positions are rotated into along-wind
and cross-wind coordinates. Drop orientation is represented relative to the same
wind direction, and the Matérn kernel learns separate lengthscales on these axes.

**Multi-fidelity optimisation.** An autoregressive low/high-fidelity surrogate
uses 300-second, 8-realisation rollouts to screen candidates before 600-second,
24-realisation evaluations in the reproducible run.

**Baselines.** Hand-designed heuristics (e.g. protect the head/flank, ring the fire, sit on high-value cells) provide interpretable comparisons.

---

## Dependencies

Core dependencies are listed in `pyproject.toml`; exact reproduction versions
are pinned in `requirements-repro.txt`. Notebook-only packages are available via:

```bash
pip install -e ".[notebooks]"
```

---

## License

MIT. See [LICENSE](LICENSE).
