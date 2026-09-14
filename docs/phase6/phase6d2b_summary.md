# Phase 6D2B ? Real Continuous Optimizer Benchmark

## Status

Phase 6D2B executed the frozen continuous benchmark against the frozen
production Phase 4G surrogate.

Result:

`results/phase6/continuous_optimizer_benchmark.json`

SHA-256:

`a4c8a55fd2ecfbf666479c6ee054a196c1a093b7640907f071fce0a04e7657d9`

Execution source commit:

`a49791461cd07418f94b5eb07c77c84d50f51d8d`

Effective continuous-benchmark identity:

`13ce4e5a488d4df6ba5b1e5370a93099f560629e49600ded77be658de8e3ce2a`

## Benchmark size

The benchmark executed exactly:

- 5 frozen single-objective qualification scenarios;
- 2 methods;
- 2 repeats per method/scenario;
- 20 controlled optimizer runs.

The production Phase 4G surrogate was loaded once for the benchmark process.

## Qualification results

Differential Evolution qualified across all five scenarios:

`True`

SHGO qualified across all five scenarios:

`False`

Qualified-method count:

`1`

Qualified methods:

`['differential_evolution']`

No production optimizer has been selected in this gate.

## Scenario-level results

The objective gap is:

`continuous primary objective - deterministic-grid primary objective`

A negative value means the continuous candidate achieved a lower frozen
objective than the discrete 0.5 W x 0.5 mTorr grid reference.

| Method | Scenario | Both runs pass | Exact repeat | Pair qualifies | Run-0 power W | Run-0 pressure mTorr | Run-0 objective | Run-0 objective gap vs grid | SciPy success |
|---|---|---|---|---|---:|---:|---:|---:|---|
| differential_evolution | soft_density_5e16 | True | True | True | 23.141221 | 42.023134 | 0.0001443171758 | 0 | True |
| differential_evolution | hard_density_1e17_min_power | True | True | True | 38.533994 | 57.559284 | 0.3137865802 | -0.00621341977 | True |
| differential_evolution | temperature_window_min_power | True | True | True | 15 | 28.787295 | 8.526512829e-16 | 8.526512829e-16 | True |
| differential_evolution | boundary_margin_reference | True | True | True | 52.5 | 35 | 1.793565296e-13 | 1.793565296e-13 | True |
| differential_evolution | nominal_centre_reference | True | True | True | 52.5 | 35 | 8.980080861e-14 | 8.980080861e-14 | True |
| shgo | soft_density_5e16 | True | True | True | 26.71875 | 30.3125 | 0.0007218146406 | 0.0005774974648 | True |
| shgo | hard_density_1e17_min_power | False | True | False | 38.4375 | 58.4375 | 0.3125 | -0.0075 | True |
| shgo | temperature_window_min_power | True | True | True | 15 | 35 | 0 | 0 | False |
| shgo | boundary_margin_reference | True | True | True | 52.5 | 35 | 4.962143474e-10 | 4.962143474e-10 | True |
| shgo | nominal_centre_reference | True | True | True | 52.5 | 35 | 8.303776201e-11 | 8.303776201e-11 | True |

## Qualification semantics

A continuous method/scenario pair qualifies only when:

1. both repeats pass explicit Phase 6 final validation; and
2. the two repeats satisfy the frozen exact-repeatability contract.

Method-level qualification requires this to hold for all five qualification
scenarios.

SciPy's `success` flag is recorded independently and does not by itself define
scientific validity.

## Grid reference

The frozen deterministic-grid result remains the mandatory baseline:

`results/phase6/deterministic_grid_baseline.json`

SHA-256:

`cb3cbad2345d5a83286e7940fdd7f1960e96b1a3fdd7eea6f0492caa058a7f59`

The grid was not modified during continuous benchmarking.

Beating the grid was not required for qualification.

## Scientific interpretation

These results compare numerical optimization methods operating on the frozen
reduced-order surrogate.

They do not establish:

- a mathematically proven global continuous optimum;
- experimental validation;
- industrial validation;
- OIPT hardware performance;
- source-plasma-model revalidation.

## Evidence separation

The benchmark did not use:

- Phase 4 TEST targets;
- the Phase 5 TEST dataset;
- the source plasma simulator;
- agentic orchestration.

The frozen scenarios were not changed after observing optimizer outcomes.

## Next step

Phase 6D3 must review the frozen evidence and select the production
optimization method, if any.

That decision should consider:

- method qualification;
- exact repeatability;
- final Phase 6 feasibility;
- objective quality relative to the grid;
- solver-status behavior;
- surrogate-evaluation cost;
- boundary behavior;
- methodological simplicity.

The selection must be documented separately and must not modify this benchmark
result.
