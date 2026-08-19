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
`d_F(x, y) != d_F(y, x)` — by up to 160% on the validation scenario. A
covariance function has to be symmetric and positive semi-definite, and a
directed distance is neither, so it cannot be substituted for the distance in a
Matérn kernel.

Symmetrising it (mean or min of the two directions) is the tempting repair. It
restores symmetry, still carries no positive-definiteness guarantee, and throws
away exactly the asymmetry that made the geometry worth having. Measured on this
fire, min-symmetrisation gives a minimum eigenvalue of `-0.25`; mean-symmetrisation
happens to stay PSD on that particular point set. That is the point — neither
outcome is guaranteed, and a GP with a negative eigenvalue has negative
predictive variance and an undefined marginal likelihood.

The fix is to **warp rather than substitute**: map each drop through a fixed,
deterministic function of the arrival-time field, then use an ordinary
stationary Matérn in that warped space. Positive definiteness is free, because
`k(phi(x), phi(y))` is PSD for any deterministic `phi` whenever `k` is. The
asymmetry survives, because the forward and reverse arrival times enter the
feature vector as two separate coordinates rather than being averaged into one.
The fitted `l_reverse` lengthscale is then a read-out: at its upper bound, the
data say the upwind/downwind distinction carries no signal.

## The radial function is not the linear form

Worth recording because it is an easy and silent mistake. A Randers metric is
`F(v) = sqrt(a(v,v)) + b.v`, so the rate of spread in direction `u` — the radial
function of the indicatrix — is `1 / (sqrt(a(u,u)) + b.u)`, the *reciprocal* of
a linear form. Writing the spread law as `s + W.u` instead builds a different
curve entirely: the two agree to first order in `|W|/s` and along the drift
axis, but at `|W|/s = 0.8` the linear version overstates the flank rate by 67%.

The correct radial function, for a base spread `s` advected by drift `W`, is

    sigma(u) = W.u + sqrt(s^2 - |W|^2 + (W.u)^2),

the circle of radius `s` centred on `W`. Squaring rearranges it to
`sigma^2 = 2 sigma (W.u) + q` with `q = s^2 - |W|^2`, which is linear in
`(W, q)` — so fitting it is a closed-form 3x3 least-squares solve per cell, and
the strong-convexity condition is just `q > 0`. Both the metric fit and the
arrival-time solver use this form, and a regression test pins the flank rate at
`sqrt(s^2 - |W|^2)` so the linear approximation cannot creep back in.

## Why the CA gained an elliptical wind response

The CA's original spread law multiplies the rate by `1 + wind_coeff * max(0, w.u)`.
The clip means wind never retards spread, so the modelled fire backs into the
wind at its full no-wind rate and the flank and backing rates are identical.
That profile is flat across the entire upwind half-plane; its reachable set has
corners of 22 to 58 degrees and is not an ellipse, so the Randers fit degrades
as wind strengthens — 4.9% relative misfit at `wind_coeff=0.4`, 11.4% at `3.0` —
and past `‖b‖_a ~ 0.3` the fitted metric predicts arrival worse than an
isotropic one.

`FireEnv.wind_response="elliptical"` replaces it with Richards' law: the set of
displacements reachable in unit time is the no-wind circle translated by the
wind drift. It is opt-in and the default is unchanged, so every existing result
reproduces bit-for-bit. In that mode, for sub-critical wind, the reachable set
is the circle of radius `s` centred on `W` to within `1e-14`, the metric fit
residual is `9e-15`, and the rates order as `back < flank < head` with the flank
narrowed to `sqrt(1 - (wind_coeff |w|)^2)` — the ellipse stretching along the
wind, with a length-to-breadth ratio of 3.0 at `wind_coeff=0.8` against 1.8 for
the clipped law. It is both the more physical law and the one that makes the
geometry an identity rather than an approximation.

## What the Finsler validation does and does not show

`python -m fire_model.finsler_validation` regresses simulated first-ignition
times onto geodesic arrival times, against an isotropic ablation that keeps the
same mean speed and zeroes the drift.

The geometry recovers the *ordering* of arrival well (Spearman `rho = 0.91` on
the headline scenario, against `0.57` for the isotropic ablation), which is what
a monotone warp coordinate needs, and the gap opens up with anisotropy. It also
wins the stricter single-calibration test (`R^2 = 0.28` against `0.13`), though
both are low in absolute terms and the constant is ≈ 0.27 rather than 1: the
stochastic percolation front outruns the mean-field rate the metric is fitted
to, because the front advances by the earliest of many competing ignition
attempts, and that speedup depends on the local hazard. The metric is an
accurate description of the CA's *rate law*, not of its stochastic front, and
the warp depends only on the former.

At low anisotropy the two metrics are indistinguishable, as they must be
(`0.98` vs `0.99` at `‖b‖_a = 0.2`). The one-form earns its place as `‖b‖_a`
grows — but only under the elliptical spread law. Under the default clipped law
the isotropic metric predicts arrival better at every wind strength tested,
which is the honest cost of fitting an ellipse to a profile that is flat over
half the circle.

## What the Finsler kernel frame alone does not buy

`--frames` runs BO under each kernel frame with everything else held fixed, and
re-scores each selected plan on an independent Monte-Carlo batch. All arms in
that experiment still decode through the SR search grid, so all receive its
simulated future boundary for free. Under that feature-only comparison the
Finsler kernel comes out level with the wind-aligned default and does not
establish an optimisation gain.

That negative result remains useful: changing a kernel is not the same as
realising the operational advantage of a front-free representation.

## Why setup simulations are part of the optimisation budget

`setup_search_grid_sr()` calls `generate_search_domain()` with the full
`n_sims` and horizon. It therefore costs exactly as many simulated
realisation-steps as one candidate evaluation. Treating the resulting boundary
as free gives SR information that a cold-start planner has not paid for.

`FinslerSearchMap` removes that hidden dependency. It inverts current-perimeter
label and deterministic arrival time directly, so its setup consumes zero
future-fire rollout steps. The equal-budget benchmark charges SR one
evaluation-equivalent before BO begins and charges Finsler zero.

The benchmark also corrects two ways a positive result could have been
manufactured accidentally:

- both arms use the rectangle's true π-periodic orientation, so SR does not
  waste half its search domain on duplicate drops;
- each planning seed is scored on a genuinely independent validation stream by
  re-instantiating `CAFireModel` with the validation seed. Passing only
  `simulate_from_firestate(seed=...)` changes spread-parameter jitter but not
  the model's base-seeded ignition stream.

## The cold-start crossover is the optimisation finding

Across 3 early-detection geometries and 12 seeds each, at a total budget of
three candidate-evaluation equivalents:

- native Finsler improves independent `CVaR_0.90` by 12.73% over no drop;
- charged SR improves it by 8.43%;
- the paired advantage is +4.30 percentage points, 95% interval
  `[+1.53, +7.06]`;
- Finsler wins 25 of 36 pairs (exact sign-test `p=0.0288`);
- all three scenario-level means are positive.

At budget four the point advantage is +2.98 points. At six and eight it remains
positive but uncertain. At twelve SR leads by 1.63 points, also uncertain. The
claim is therefore deliberately narrow: the physics-derived prior improves
sample efficiency when simulations are extremely scarce; the
simulation-derived SR coordinates catch up as data accumulate.

Giving SR its setup for free reduces the budget-three gap to 2.49 points and
removes statistical establishment. Setup accounting explains a material part
of the finding, but the experiment does not establish how much of the remaining
gap comes from arrival-time coordinates versus other representation choices.

See `COLD_START_FINDING.md` and `artifacts/cold_start/cold_start_summary.json`
for the full protocol, caveats and raw selected plans.
