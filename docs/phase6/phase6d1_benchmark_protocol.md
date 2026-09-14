# Phase 6D1 ? Continuous Optimizer Benchmark Protocol

## Status

This gate freezes the continuous-method benchmark contract before either
continuous optimizer is implemented or executed.

Protocol:

`configs/phase6/continuous_optimizer_benchmark.json`

SHA-256:

`0e618bf046a0d3749393003f15d2015ba2b607b79bc7490c24ac861ca03e9ad9`

Frozen deterministic-grid reference:

`results/phase6/deterministic_grid_baseline.json`

Grid result SHA-256:

`cb3cbad2345d5a83286e7940fdd7f1960e96b1a3fdd7eea6f0492caa058a7f59`

## Scenario eligibility

Continuous benchmarking is restricted to frozen Phase 6C1 scenarios having
exactly one declared objective.

This rule is structural and does not depend on the observed grid result.

The five qualification scenarios are:

1. `soft_density_5e16`
2. `hard_density_1e17_min_power`
3. `temperature_window_min_power`
4. `boundary_margin_reference`
5. `nominal_centre_reference`

The other five Phase 6C1 scenarios are multiobjective and remain deferred.

No weighted-sum or other continuous multiobjective scalarisation is introduced.

## Methods

Two methods are benchmarked.

Differential Evolution is the primary continuous candidate, with the already
frozen Phase 6A settings:

- strategy `best1bin`;
- `maxiter=300`;
- `popsize=20`;
- `tol=1e-9`;
- `atol=1e-12`;
- mutation `(0.5, 1.0)`;
- recombination `0.7`;
- no polishing;
- one worker;
- immediate updating;
- RNG `numpy.random.default_rng(20260914)`.

SHGO is the deterministic global comparator:

- `n=256`;
- `iters=3`;
- simplicial sampling;
- one worker;
- default SciPy 1.18.1 local-minimizer configuration.

A standalone local `minimize` method is not executed in this benchmark because
its exact configuration was not previously frozen and the production
surrogates are tree-based.

## Hard constraints

Continuous methods must use actual nonlinear inequality constraints.

No hard constraint may be replaced with an objective penalty.

Every active constraint follows `g(x) >= 0`.

The frozen mappings are:

- density tolerance minus density relative error;
- predicted Te minus minimum Te;
- maximum Te minus predicted Te;
- normalized boundary margin minus required minimum margin.

Qualified-domain limits are enforced through optimizer bounds.

## Final feasibility

A SciPy success flag is recorded but is not sufficient to establish a valid
Phase 6 result.

Every returned candidate must be independently re-evaluated through the Phase
6B feasibility primitives.

Where the deterministic grid demonstrated feasibility, a qualifying continuous
run must also return a scenario-feasible candidate.

Solver failure never establishes global physical infeasibility.

## Repeatability

Each method is executed twice for each qualification scenario.

Differential Evolution receives the same frozen RNG seed on both repeats.

SHGO is repeated under the same deterministic configuration.

The two runs must agree exactly in:

- returned decision coordinates;
- final objective;
- final physical predictions;
- final feasibility.

## Grid comparison

The grid remains the mandatory baseline.

For each continuous result the benchmark records the primary-objective gap:

`continuous objective - grid objective`

A negative value means the continuous candidate improved the frozen objective
relative to the discrete 0.5 W x 0.5 mTorr grid point.

Beating the grid is not required for method qualification.

A continuous result may demonstrate that a discrete grid point was not optimal
over the continuous search domain, but it may not be described as proof of the
global continuous optimum.

## Method selection

No optimizer is selected in the benchmark-execution gate.

Selection requires a separate review after both methods have produced frozen
evidence.

The review must consider:

- feasibility;
- reproducibility;
- objective quality;
- solver status;
- prediction evaluations;
- boundary behaviour;
- agreement with the deterministic baseline;
- methodological simplicity.

## Evidence separation

This protocol freeze performs no surrogate inference or continuous
optimisation.

It does not access Phase 4 TEST targets, the Phase 5 TEST dataset or the source
plasma simulator.

The frozen Phase 6C grid result is used only as the benchmark reference and is
not modified.
