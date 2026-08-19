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

## Why there is a Finsler geometry in a fire model

Fire spread under wind is the standard worked example of a **Randers metric**: a
Riemannian part (the no-wind rate of spread) plus a one-form (the wind drift).
Richards' elliptical growth model, the operational standard, is a Randers metric
in other notation, and fire arrival time is the Finslerian distance from the
ignition set.

The wind-aligned kernel frame is a hand-built approximation of a geometry the
simulator already implies. `fire_model/finsler.py` derives it instead: the CA's
own directional rate law is projected onto the Randers family, arrival time is
solved as a Finsler distance from the perimeter, and the surrogate's features
are built from that field. The anisotropy stops being a modelling choice that
needs defending and becomes a consequence of the spread parameters, and drops
are then placed against where the front will be rather than against grid `x`
and `y`.

## Why the kernel warps the geometry instead of using its distance

Finsler distance is asymmetric. Travelling upwind is slower than downwind, so
`d_F(x, y) != d_F(y, x)` — by up to 140% on the validation scenario. A
covariance function has to be symmetric and positive semi-definite, and a
directed distance is neither, so it cannot be substituted for the distance in a
Matérn kernel.

Symmetrising it (mean or min of the two directions) is the tempting repair. It
restores symmetry, still carries no positive-definiteness guarantee, and throws
away exactly the asymmetry that made the geometry worth having. On this fire it
also simply fails: `artifacts/finsler/finsler_summary.json` records negative
minimum eigenvalues for both symmetrisations, which would leave the GP with
negative predictive variance and an undefined marginal likelihood.

The fix is to **warp rather than substitute**: map each drop through a fixed,
deterministic function of the arrival-time field, then use an ordinary
stationary Matérn in that warped space. Positive definiteness is free, because
`k(phi(x), phi(y))` is PSD for any deterministic `phi` whenever `k` is. The
asymmetry survives, because the forward and reverse arrival times enter the
feature vector as two separate coordinates rather than being averaged into one.
The fitted `l_reverse` lengthscale is then a read-out: at its upper bound, the
data say the upwind/downwind distinction carries no signal.

## Why the CA gained an elliptical wind response

The CA's original spread law multiplies the rate by `1 + wind_coeff * max(0, w·u)`.
The clip means wind never retards spread, so the modelled fire backs into the
wind at its full no-wind rate and the flank and backing rates are identical.
That profile is flat across the entire upwind half-plane, which no ellipse can
reproduce, so the Randers fit degrades as wind strengthens — 5% relative misfit
at `wind_coeff=0.4`, 20% at `3.0` — and past `‖b‖_a ≈ 0.5` the fitted metric
predicts arrival worse than an isotropic one.

`FireEnv.wind_response="elliptical"` drops the clip. It is opt-in and the
default is unchanged, so every existing result reproduces bit-for-bit. In that
mode the CA's law *is* `ros * fuel * (1 + wind_coeff * w·u)` — a Randers
indicatrix written out — the metric fit residual is zero to machine precision,
and the backing rate falls below the flank rate as Richards' model says it
should. It is both the more physical law and the one that makes the geometry an
identity rather than an approximation.

## What the Finsler validation does and does not show

`python -m fire_model.finsler_validation` regresses simulated first-ignition
times onto geodesic arrival times, against an isotropic ablation that keeps the
same mean speed and zeroes the drift.

The geometry recovers the *ordering* of arrival well (Spearman `rho = 0.90` on
the headline scenario, against `0.73` for the isotropic ablation), which is what
a monotone warp coordinate needs, and the gap widens with anisotropy. On the
stricter single-calibration test the isotropic ablation scores better
(`R^2 = 0.41` against `0.34`) and the constant is ≈ 0.35 rather than 1: the
stochastic percolation front outruns the mean-field rate the metric is fitted
to, because the front advances by the earliest of many competing ignition
attempts, and that speedup depends on the local hazard. A metric with a wide
speed range is penalised most by a single constant. The metric is an accurate
description of the CA's *rate law*, not of its stochastic front, and the warp
depends only on the former.

At low anisotropy the two metrics are indistinguishable, as they must be. The
one-form earns its place as `‖b‖_a` grows — but only under the elliptical
spread law. Under the default clipped law the isotropic metric predicts arrival
better at every wind strength tested, which is the honest cost of fitting an
ellipse to a profile that is flat over half the circle.

## What the Finsler kernel frame does not buy

`--frames` runs BO under each kernel frame with everything else held fixed, and
re-scores each selected plan on an independent Monte-Carlo batch. The Finsler
frame comes out level with the wind-aligned default and behind the SR frame,
under both spread laws. On a smaller variant of the same scenario the ranking
against the wind frame reversed completely, which says the differences are
scenario-dependent noise at five seeds rather than a real effect either way.

No optimisation gain is claimed, and the frame is not the default. The SR frame
likely retains an edge because its coordinates come from simulated fire
boundaries and so already encode the stochastic front's shape. The Finsler
frame's case is structural rather than empirical: it is derived from the spread
parameters instead of requiring a Monte-Carlo front to exist first, it needs no
boundary extraction or strip smoothing, and it represents the upwind/downwind
asymmetry explicitly instead of discarding it. Establishing whether that
translates into better optimisation would need far more seeds and more than one
scenario.
