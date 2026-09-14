# Phase 6D2A — Continuous Benchmark Engine and Synthetic Qualification

## Status

Phase 6D2A implements the frozen continuous benchmark contract together with
the controlled Phase 6D1.1 SHGO compatibility amendment.

This remains optimisation-engineering evidence only.

No production plasma-surrogate benchmark is executed in this gate.

## Effective benchmark identity

Base benchmark protocol SHA-256:

`0e618bf046a0d3749393003f15d2015ba2b607b79bc7490c24ac861ca03e9ad9`

Compatibility amendment SHA-256:

`e54e2e3a6d947f274171afc4e01bd203ea5e617b2f27730add1109321431d62f`

Effective continuous-benchmark SHA-256:

`13ce4e5a488d4df6ba5b1e5370a93099f560629e49600ded77be658de8e3ce2a`

## Mathematical constraint source of truth

Each active hard constraint is created once as a mathematical `g(x)` function.

Feasibility remains:

`g(x) >= 0`

The same function objects are then wrapped differently only because the two
SciPy solvers expose different practical interfaces.

### Differential Evolution

Differential Evolution receives:

`NonlinearConstraint(g, 0.0, numpy.inf)`

### SHGO

SHGO receives:

`{"type": "ineq", "fun": g}`

No constraint equation or sign changes between these representations.

## Why the SHGO representation differs

The first synthetic implementation attempt demonstrated that SciPy SHGO
internally converts a `NonlinearConstraint` and probes it at a placeholder
coordinate outside the qualified Phase 4 domain.

Phase 6 correctly failed closed.

The controlled amendment therefore changed only SHGO's SciPy interface
representation. Domain checks were not weakened.

## Domain policy

All prediction requests continue to pass through explicit Phase 6 decision
validation.

No clipping, extrapolation or coordinate rounding is permitted.

The prediction cache key is based on exact IEEE-754 float64 coordinate bits.

## Synthetic qualification

The development-only predictor is:

- `electron_density_m3 = 2.5e15 * absorbed_power_W`
- `electron_temperature_eV = 1.55 + 0.005 * pressure_mTorr`

This predictor is not a plasma model and creates no scientific plasma evidence.

The tests verify:

- base/amendment/effective benchmark identities;
- exact five-scenario qualification order;
- exact-coordinate caching;
- no coordinate rounding;
- fail-closed out-of-domain behavior;
- DE and SHGO wrappers return identical `g(x)` values;
- density-constraint sign;
- temperature-constraint signs;
- constrained Differential Evolution execution;
- constrained SHGO execution without the placeholder failure;
- exact DE repeatability;
- exact SHGO repeatability;
- multiobjective scenario rejection;
- unsupported-method rejection.

## Final candidate validation

A solver result is independently re-evaluated using the frozen Phase 6
candidate evaluator.

`run_pass` requires:

- in-domain decision coordinates;
- finite strictly positive physical predictions;
- explicit scenario feasibility;
- all active `g(x) >= 0` constraints;
- a finite frozen objective.

SciPy's `success` flag is recorded separately and is not treated as the sole
scientific validity criterion.

## Explicit exclusions

This gate performs no:

- production Phase 4G surrogate loading;
- real plasma-surrogate inference;
- real continuous benchmark;
- method selection;
- source-simulator execution;
- Phase 4 TEST access;
- Phase 5 TEST access.

## Next step

After this implementation is committed, Phase 6D2B may execute the real
continuous benchmark against the frozen Phase 4G production surrogate.

The real benchmark remains:

- five qualification scenarios;
- Differential Evolution and SHGO;
- two repeats per method/scenario;
- twenty controlled runs in total.

Those results must be frozen before method selection.
