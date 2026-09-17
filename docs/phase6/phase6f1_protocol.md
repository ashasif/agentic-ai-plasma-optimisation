# Phase 6F1 ? Production Optimisation Runtime Protocol

## Status

Phase 6F1 freezes the production-runtime, fallback, persistence and
reproducibility contract for the qualified Phase 6 optimisation capability.

Protocol:

`configs/phase6/runtime_protocol.json`

SHA-256:

`ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a`

No optimization or surrogate inference is executed by this gate.

## Qualified production method

Primary continuous optimizer:

**Differential Evolution**

Production RNG seed:

`20260914`

Mandatory reference and fallback:

**Deterministic 15,251-point grid**

SHGO is not eligible for the runtime.

## Supported requests

The runtime supports validated single-objective Phase 6 scenarios.

Multiobjective runtime requests are rejected because no continuous
multiobjective scalarization or production preference rule has been qualified.

No hidden scalarization is permitted.

Scenario validation occurs before surrogate inference.

## Runtime initialization

A runtime instance must:

1. validate the Phase 6 runtime manifest and all referenced hashes;
2. load the frozen Phase 4G surrogate once;
3. construct the frozen 15,251-point grid once;
4. execute one complete grid prediction batch;
5. verify exact frozen input, density and temperature array SHA-256 identities;
6. retain those grid predictions as read-only runtime reference data.

Expected array identities are:

Input:

`f77c780d9a7c4e4c6db1dc921317c0b48e6d2e397d14075abbe34d38a2472cfd`

Density:

`70e60a5659f9ea5ac4f0c768d056119db7aacccf7c90d85e06996ab092008b90`

Temperature:

`4f3d9e087554e378478bec2405d624a061e46ad49facfbcd1058c2b9a7ba7dd0`

A mismatch is a fail-closed runtime-initialization error.

## Request execution

For every valid single-objective request:

1. evaluate the deterministic grid reference using cached grid predictions;
2. execute Differential Evolution with the frozen Phase 6D parameters;
3. use the fixed production seed `20260914`;
4. independently recheck the final Phase 6 feasibility contract;
5. compare the selected-method primary objective with the grid reference.

Production seed and optimizer parameters cannot be overridden.

## Explicit fallback policy

Fallback is never silent.

If Differential Evolution is feasible and no worse than the grid by more than
`1e-6`, the status is:

`selected_method_accepted`

If Differential Evolution is infeasible but the grid has a feasible point:

`grid_fallback_selected_method_infeasible`

If Differential Evolution is feasible but its objective is worse than the
grid by more than `1e-6`:

`grid_fallback_selected_method_objective_regression`

If both methods fail to produce a feasible point:

`no_feasible_point_found_under_search_protocol`

That final status is not a claim of global physical infeasibility.

## Persistence

A serialized optimizer object is not required.

Differential Evolution is stateless between requests. Persistence is therefore
defined through:

- frozen optimizer configuration;
- fixed production seed;
- frozen runtime implementation;
- validated surrogate artifacts;
- validated upstream evidence;
- a Phase 6 runtime manifest.

Planned manifest:

`artifacts/phase6/optimization_runtime_manifest.json`

The loader must fail closed on any manifest/hash mismatch.

## Reproducibility

The frozen persistence-equivalence probe set contains three requests:

1. `soft_density_5e16`;
2. `hard_density_1e17_min_power`;
3. `boundary_margin_constraint_edge`.

Two independent runtime instances will execute all three requests.

That gives six real probe optimizations.

Exact non-timing equality is required between corresponding outputs.

No numerical tolerance is permitted for persistence-equivalence comparison.

Wall-clock timing is explicitly excluded.

## Negative runtime qualification

The implementation must also reject, without real production-surrogate
inference where applicable:

- out-of-domain nominal points;
- multiobjective runtime requests;
- invalid density targets;
- invalid temperature intervals;
- invalid boundary margins;
- production-seed overrides;
- optimizer-parameter overrides;
- manifest drift;
- surrogate-artifact drift.

## Scientific boundaries

The runtime remains a reduced-order argon ICP optimisation capability.

It does not establish:

- experimental validation;
- industrial validation;
- OIPT hardware performance;
- wafer-scale spatial behaviour;
- reactive-chemistry behaviour;
- a mathematically proven global continuous optimum.

## Next steps

Phase 6F2 may implement and synthetically qualify the production runtime and
manifest loader.

Phase 6F3 remains locked until Phase 6F2 passes.

Phase 6G final audit remains locked until Phase 6F is complete.

Phase 7 remains locked until Phase 6 is formally closed.
