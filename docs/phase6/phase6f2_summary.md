# Phase 6F2 — Production Runtime and Manifest Loader

## Status

Phase 6F2 implements the production optimisation runtime contract frozen in
Phase 6F1 and Phase 6F1.1.

This gate produces implementation and synthetic qualification evidence only.

The official production runtime manifest is not created in this gate.

## Effective runtime protocol

Base:

`configs/phase6/runtime_protocol.json`

Base SHA-256:

`ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a`

Amendment:

`configs/phase6/runtime_protocol_amendment_001.json`

Amendment SHA-256:

`6294323543b9c4d121e4f4a8dcba34f8aad133924b12d63add1b37494232b87e`

Effective identity:

`7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83`

## Runtime implementation

The runtime implements:

- validated single-objective requests only;
- deterministic-grid reference evaluation for every request;
- fixed-seed Differential Evolution execution;
- independent final Phase 6 feasibility;
- explicit grid comparison;
- five frozen runtime statuses;
- canonical non-timing response serialization;
- read-only cached grid prediction vectors.

## Five-state outcome model

The runtime implements exactly:

1. `selected_method_accepted`
2. `selected_method_accepted_grid_infeasible`
3. `grid_fallback_selected_method_infeasible`
4. `grid_fallback_selected_method_objective_regression`
5. `no_feasible_point_found_under_search_protocol`

Fallback is explicit and never silent.

## Manifest loader

The manifest subsystem validates:

- runtime base protocol hash;
- runtime amendment hash;
- effective runtime identity;
- runtime implementation source hash;
- Phase 4G surrogate manifest hash;
- density model hash;
- temperature model hash;
- optimizer-selection hash;
- Phase 6E result hash;
- continuous benchmark result hash;
- deterministic-grid result hash;
- production seed;
- selected method;
- full five-state status vocabulary.

A mismatch fails closed.

## Production loader

The real production loader is implemented but not executed by this gate.

When Phase 6F3 invokes it, the loader will:

1. validate the official runtime manifest;
2. load the trusted Phase 6 surrogate adapter;
3. build the frozen 15,251-point grid;
4. predict exactly one complete grid batch;
5. verify exact frozen grid-array hashes;
6. freeze those arrays read-only;
7. configure selected Differential Evolution with seed `20260914`.

## Synthetic qualification

Synthetic tests cover:

- effective runtime identity;
- exact five-state vocabulary;
- runtime-manifest construction;
- valid temporary manifest loading;
- fail-closed seed drift;
- fail-closed status-vocabulary drift;
- fail-closed runtime-source hash drift;
- multiobjective rejection;
- selected-method acceptance;
- grid fallback on selected-method infeasibility;
- grid fallback on objective regression;
- no-feasible result;
- continuous-feasible/grid-infeasible acceptance;
- exact repeated non-timing response equality;
- read-only runtime grid vectors.

## Explicit exclusions

Phase 6F2 performs no:

- official runtime-manifest creation;
- production Phase 4G surrogate loading;
- production surrogate inference;
- real persistence-equivalence probe;
- SHGO execution;
- optimizer retuning;
- source-simulator execution;
- Phase 4 TEST access;
- Phase 5 TEST access.

## Next step

Phase 6F3 may now:

- freeze the official runtime manifest;
- load two independent production runtime instances;
- execute the three frozen persistence probes in each instance;
- compare six real optimization outputs;
- require exact non-timing equality;
- freeze persistence-equivalence evidence.

Phase 6G and Phase 7 remain locked until Phase 6F3 passes.
