# Phase 6D1.1 ? SHGO Constraint-Compatibility Amendment

## Status

This is a controlled compatibility amendment to the frozen Phase 6D1
continuous-optimizer benchmark protocol.

Base benchmark protocol:

`configs/phase6/continuous_optimizer_benchmark.json`

Base SHA-256:

`0e618bf046a0d3749393003f15d2015ba2b607b79bc7490c24ac861ca03e9ad9`

Amendment:

`configs/phase6/continuous_optimizer_benchmark_amendment_001.json`

Amendment SHA-256:

`e54e2e3a6d947f274171afc4e01bd203ea5e617b2f27730add1109321431d62f`

Effective continuous-benchmark identity:

`13ce4e5a488d4df6ba5b1e5370a93099f560629e49600ded77be658de8e3ce2a`

## Trigger

The first Phase 6D2A synthetic implementation attempt stopped before any
scientific benchmark was executed.

SciPy SHGO internally converted a `NonlinearConstraint` and evaluated the
constraint function at an internal placeholder point approximately:

- absorbed power: `0.013333333333333334`
- pressure: `0.02`

That point lies outside the frozen qualified Phase 4 operating domain.

The existing fail-closed Phase 6 decision validation correctly rejected it.

No production surrogate was loaded and no scientific continuous-optimization
evidence was generated.

## Controlled change

The mathematical hard constraints remain exactly unchanged.

Every hard constraint continues to use:

`g(x) >= 0`

Differential Evolution continues to receive:

`NonlinearConstraint(g, 0.0, numpy.inf)`

SHGO will instead receive the same `g` function through SciPy's old-style
inequality interface:

`{"type": "ineq", "fun": g}`

SciPy's old-style inequality convention is also `fun(x) >= 0`.

Therefore this amendment changes only the solver-interface representation.

## Explicitly unchanged

This amendment does not change:

- qualified decision-space bounds;
- any density target;
- any electron-temperature bound;
- any boundary-margin rule;
- any objective definition;
- any hard-constraint formula;
- hard-constraint signs;
- Differential Evolution settings;
- SHGO sampling settings;
- scenario eligibility;
- repeatability requirements;
- deterministic-grid evidence;
- method-selection policy;
- final Phase 6 feasibility recheck;
- TEST separation;
- scientific scope.

## Domain safety

The amendment does not permit clipping or extrapolation.

Phase 6 fail-closed domain validation remains active.

Production-surrogate prediction remains forbidden outside:

- 15?90 W absorbed power;
- 10?60 mTorr pressure.

## Implementation requirement

The next Phase 6D2A implementation must create each mathematical `g(x)`
function once and wrap the same function differently for each solver:

- DE -> `NonlinearConstraint`
- SHGO -> old-style inequality dictionary

Synthetic tests must prove that both wrappers return the same constraint value
at the same in-domain point before real surrogate benchmarking is unlocked.

## Evidence separation

This amendment gate performs no:

- Differential Evolution execution;
- SHGO execution;
- production-surrogate inference;
- source-simulator execution;
- Phase 4 TEST access;
- Phase 5 TEST access;
- continuous benchmark result generation;
- optimizer selection.
