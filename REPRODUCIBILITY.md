# Reproducing the risk-sensitive BO result

## One command

From a clean clone on macOS or Linux:

```bash
bash run_all.sh
```

The script creates `.venv`, installs the exact versions in
`requirements-repro.txt`, runs the smoke/regression tests, and executes the
CI-sized experiment. Outputs are written to:

- `artifacts/reproduction/summary.json`
- `artifacts/reproduction/cvar_mfbo_reproduction.png`

## What the experiment fixes

- Optimisation objective: empirical upper-tail `CVaR_0.90`, computed from the
  unaggregated Monte-Carlo loss ensemble.
- GP representation: positions and drop orientations rotated into the mean-wind
  frame, with distinct along-wind, cross-wind and orientation lengthscales.
- High fidelity: 24 simulations over 600 seconds.
- Low fidelity: 8 simulations over 300 seconds.
- BO/search seed: 17; independent validation seed: 101.
- Independent validation: 64 simulations for both the no-drop baseline and the
  selected plan.

The full-sized version is:

```bash
MPLBACKEND=Agg python -m fire_model.demo
```

It uses more initial designs, candidates, BO iterations and validation
simulations. Expect numerical differences from the quick run because it spends a
larger simulation budget.

## Interpreting the output

`selection_estimate_cvar` is the noisy value observed while selecting the plan.
The headline comparison uses `baseline_cvar` and `selected_plan_cvar`, both
computed afterward on the same independent Monte-Carlo batch. Lower is better.

This is a synthetic stress test, not an operational wildfire forecast. The model
does not claim calibrated real-world suppression effectiveness.
