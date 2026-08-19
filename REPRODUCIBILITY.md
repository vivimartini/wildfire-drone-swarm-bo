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
- `artifacts/finsler/finsler_summary.json`
- `artifacts/finsler/finsler_validation.png`

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

## Reproducing the Finsler geometry validation

```bash
MPLBACKEND=Agg python -m fire_model.finsler_validation            # full size
MPLBACKEND=Agg python -m fire_model.finsler_validation --quick    # CI size
MPLBACKEND=Agg python -m fire_model.finsler_validation --frames   # + kernel-frame BO comparison
```

This fixes:

- Metric: Randers field fitted to the CA's directional spread law on 64
  directions, drift shrunk to keep `|W| / s <= 0.95`.
- Arrival time: Dijkstra on the 16-neighbour lattice with directed trapezoidal
  slowness weights; the stencil keeps lattice anisotropy under 3%.
- Headline scenario: `wind_response="elliptical"`, where the metric is exact.
- CA arrival times: 64 realisations at base spread parameters (no jitter), seed
  7, so the test isolates geometry from parameter uncertainty.
- Anisotropy sweep: `wind_coeff` in {0.2, 0.4, 0.8, 1.6, 3.0} under both spread
  laws, each against an isotropic ablation.
- Positive-definiteness check: 16 cells drawn with seed 0.
- `--frames`: 5 paired BO seeds per kernel frame, each selected plan re-scored
  on 128 independent realisations with validation seed 909.

The kernel-frame comparison is a small-sample study. Paired win counts are
reported next to the means because at five seeds they are the more honest
summary.

## Interpreting the output

`selection_estimate_cvar` is the noisy value observed while selecting the plan.
The headline comparison uses `baseline_cvar` and `selected_plan_cvar`, both
computed afterward on the same independent Monte-Carlo batch. Lower is better.

This is a synthetic stress test, not an operational wildfire forecast. The model
does not claim calibrated real-world suppression effectiveness.
