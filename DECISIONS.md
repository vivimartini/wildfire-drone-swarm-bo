# Design decisions and limitations

## Why CVaR is the default objective

Expected loss can hide rare but severe asset-loss outcomes. The SR optimiser
therefore defaults to empirical upper-tail `CVaR_0.90`. The simulator preserves
the full Monte-Carlo ensemble until per-realisation losses have been computed;
aggregated cell probabilities are rejected by the CVaR path.

`risk_measure="mean"` remains available for controlled comparisons.

## Why the kernel is wind-aligned

Intervention similarity is directional: moving a drop one kilometre downwind is
not equivalent to moving it one kilometre across the fire front. The default GP
features rotate Cartesian drop positions into along-wind and cross-wind axes and
encode drop orientation relative to the same observed mean-wind direction.
Separate Matérn lengthscales make this anisotropy explicit.

The earlier SR-coordinate kernel remains available with `kernel_frame="sr"`.

## Why two fidelities

The low-fidelity model reduces both rollout horizon and Monte-Carlo count. An
autoregressive co-kriging model combines those observations with full-horizon
evaluations, while the acquisition logic accounts for their relative costs.

## Evidence boundary

The checked-in result is a deterministic synthetic smoke test. It demonstrates
the optimisation and evaluation plumbing; it does not validate wildfire physics,
retardant efficacy, aviation constraints or operational safety. Real deployment
would require calibrated spread models, uncertainty validation, geospatial data,
regulatory review and human incident-command oversight.
