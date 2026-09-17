# Phase 6F1.1 ? Runtime Fallback-Completeness Amendment

## Status

This amendment closes one runtime-state gap identified before Phase 6F2
implementation.

Base runtime protocol:

`configs/phase6/runtime_protocol.json`

Base SHA-256:

`ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a`

Amendment:

`configs/phase6/runtime_protocol_amendment_001.json`

Amendment SHA-256:

`6294323543b9c4d121e4f4a8dcba34f8aad133924b12d63add1b37494232b87e`

Effective runtime-protocol identity:

`7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83`

## Missing state

The base protocol did not explicitly define the case where:

- Differential Evolution returns a valid Phase 6-feasible continuous point;
- the mandatory deterministic grid contains zero feasible candidates.

This state is possible because the continuous search can identify a feasible
point between the frozen 0.5 W ? 0.5 mTorr grid locations.

## Frozen behavior

The status is:

`selected_method_accepted_grid_infeasible`

Chosen source:

`differential_evolution`

The grid reference remains explicitly recorded with zero feasible candidates.

Because no grid objective exists:

`selected_minus_grid_primary_objective = null`

The continuous point is accepted only when:

- the selected-method run passes;
- final Phase 6 feasibility is true;
- the primary objective is finite.

## Effective runtime statuses

The complete status vocabulary is now:

1. `selected_method_accepted`
2. `selected_method_accepted_grid_infeasible`
3. `grid_fallback_selected_method_infeasible`
4. `grid_fallback_selected_method_objective_regression`
5. `no_feasible_point_found_under_search_protocol`

## Unchanged science and optimization

This amendment changes no:

- optimizer selection;
- Differential Evolution parameter;
- production seed;
- deterministic-grid contract;
- Phase 6 qualification result;
- surrogate model;
- scientific objective or constraint definition.

No optimization was executed.

## Interpretation

Acceptance of a continuous feasible point when the discrete grid is infeasible
does not prove a global optimum.

Likewise, a no-feasible result remains only:

`no_feasible_point_found_under_search_protocol`

and is not a claim of physical impossibility.

## Next step

Phase 6F2 must implement and synthetically test all five effective runtime
states.

Phase 6F3, Phase 6G and Phase 7 remain locked.
