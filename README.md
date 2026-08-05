# Wildfire Drone Swarm BO

**Optimising drone swarms to fight wildfires with Bayesian optimisation**

This project studies how a small swarm of retardant-dropping drones can be scheduled against a spreading wildfire. A cellular-automata fire model provides the physics; Bayesian optimisation (including multi-fidelity BO) searches over drop locations and orientations to protect high-value assets.

---

## What this repo does

1. **Simulate** wildfire spread on a grid with fuel, wind, slope, and retardant decay.
2. **Parameterise** candidate drops either in Cartesian coordinates `(x, y, φ)` or in an SR fire-front frame `(s, r, δ)` between successive fire boundaries.
3. **Optimise** swarm drop plans with Gaussian-process Bayesian optimisation, and compare against simple heuristics (boundary, head/flank, point protection, etc.).
4. **Evaluate** on synthetic stress-test scenarios and a Victoria-style semi-realistic environment.

---

## Features

| Area | Details |
|------|---------|
| Fire model | Grid CA with wind, fuel, value maps, time-varying ROS, retardant half-life |
| Drop geometry | Oriented rectangular drops with optional “avoid burning cells” |
| Coordinates | Cartesian BO and SR-native BO with tied Matérn kernels |
| Optimisation | Expected improvement, heuristic warm-starts, multi-fidelity (MFBO) co-kriging |
| Baselines | Ring / boundary / head–flank / value-blocking heuristics |
| Scenarios | Fuel jumps, wind corridors, asset clusters, realistic Victorian layout |

---

## Setup

Requires Python 3.9+.

```bash
cd wildfire-drone-swarm-bo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
# or: pip install -r requirements.txt && pip install -e .
```

The editable install puts the `fire_model` package on your path so notebooks under `notebooks/` import cleanly. Each notebook also has a short bootstrap cell that locates the repo root if you open it without installing.

---

## Quick start

```python
from fire_model import FireEnv, CAFireModel, FireState
from fire_model import RetardantDropBayesOptSR

# Build an environment (fuel / value / wind arrays + grid metadata)
# env = FireEnv(...)

model = CAFireModel(env, seed=0)

# SR-native Bayesian optimisation over (s, r, δ) drop parameters
# opt = RetardantDropBayesOptSR(...)
# result = opt.run(...)
```

For a full worked example, open:

`notebooks/experiments/end_to_end_realistic_BO.ipynb`

---

## Repository layout

```
fire_model/                      Core simulation + optimisation library
  ca.py                          Cellular automata fire model (FireEnv, FireState)
  boundary.py                    Fire-front extraction and between-boundary masks
  harmonic.py                    Harmonic strip maps → (s, r) parameterisation
  bo.py                          Bayesian optimisation in (x, y, φ)
  bo_sr.py                       SR-native BO and multi-fidelity BO

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

**Optimisation.** A Gaussian process models expected remaining asset value (or related loss). Expected improvement proposes the next swarm configuration. Multi-fidelity BO can use cheaper short-horizon rollouts to accelerate search before expensive full simulations.

**Baselines.** Hand-designed heuristics (e.g. protect the head/flank, ring the fire, sit on high-value cells) provide interpretable comparisons.

---

## Dependencies

Listed in `pyproject.toml` / `requirements.txt`:

`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `GPy`, `botorch`, `torch`, `tqdm`

---

## Licence

No licence file is included yet. Add one before making the repository public if you intend others to reuse the code.
