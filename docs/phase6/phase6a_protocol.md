# Phase 6A ? Optimisation Protocol Freeze

## Status

Phase 6A freezes the scientific and computational contract for constrained
operating-point optimisation.

No optimiser is implemented in this subphase.

Frozen protocol:

`configs/phase6/optimisation_protocol.json`

SHA-256:

`57fee45c69c5f42f9e94b786e895a5d89b163d141367326706eac2d3d2ed1cb1`

## Trusted surrogate interface

Phase 6 optimisation must use the frozen Phase 4G trusted inference layer:

- module: `plasma_ai.surrogate.phase4g_persistence`
- loader: `load_phase4g_surrogate`
- loaded runtime type: `LoadedPhase4GSurrogate`
- prediction method: `predict_physical`
- prediction container: `Phase4GPredictions`
- prediction fields: `electron_density_m3`, `electron_temperature_eV`

Direct loading of the density or temperature pickle is prohibited.

The trusted Phase 4G loader remains responsible for validating persisted
metadata, artifact hashes, estimator contracts and runtime compatibility.

## Decision variables

The decision vector is fixed as:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

The qualified surrogate domain is:

- absorbed power: 15?90 W inclusive
- target pressure: 10?60 mTorr inclusive

No Phase 6 optimiser may silently extrapolate or clip an out-of-domain
candidate back into the domain.

## Scientific scope

Phase 6 remains optimisation of a reduced-order synthetic argon ICP model.

It does not establish experimental validation, industrial validation, OIPT
hardware performance, an OIPT operating range, wafer-scale spatial plasma
behaviour or reactive process chemistry.

Absorbed power is not treated as equivalent to generator RF power.

## Feasibility

Feasibility is separated into:

- domain feasibility;
- surrogate-output validity;
- satisfaction of explicitly active scenario constraints;
- target achievement where a target tolerance is explicitly requested.

A numerical search that does not find a feasible candidate may report:

`no_feasible_point_found_under_search_protocol`

It may not claim proof of global physical infeasibility.

## Objectives and trade-offs

The protocol defines explicit components for:

- electron-density target error;
- normalized deviation from a nominal operating point;
- normalized absorbed-power cost;
- normalized margin from the qualified-domain boundary.

No hidden scalarisation is permitted.

Every active objective must be exposed, and objective priority must be
explicit. Multiobjective grid studies must preserve non-dominated candidates
rather than hiding the trade-off behind one undocumented score.

## Boundary policy

The qualified-domain boundary itself is permitted.

There is no arbitrary implicit safety margin.

A minimum boundary margin becomes a hard constraint only when a scenario
explicitly requests it. Boundary margin must nevertheless always be reported.

## Mandatory baseline

The Phase 6 baseline is a deterministic Cartesian grid:

- 151 absorbed-power values;
- 101 pressure values;
- 15,251 operating points;
- 0.5 W power spacing;
- 0.5 mTorr pressure spacing.

The grid is a discrete optimisation baseline and does not claim to locate the
continuous optimum.

## Continuous optimisation candidates

The frozen benchmark candidates are:

- SciPy `differential_evolution` as the primary bounded continuous candidate;
- SciPy `shgo` as a deterministic global comparator.

Local `minimize` methods may be used diagnostically, but are not eligible for
automatic primary selection because the frozen surrogate is tree-based and is
not assumed to provide a smooth differentiable response surface.

Reinforcement learning is excluded from Phase 6.

## Method-selection principle

A more sophisticated method is not automatically preferred.

Eligibility requires domain compliance, reproducibility, correct feasibility
and infeasibility handling, valid surrogate predictions, and consistency with
the mandatory grid baseline.

Selection will occur only against a later frozen qualification-scenario set.

Where methods are scientifically adequate and materially equivalent, the
simpler method is preferred.

## Source-model policy

The expensive source simulator is not permitted during routine optimisation
or Phase 6 optimisation development.

Any independent post-selection source-model spot check would require a
separate frozen protocol and could not be used to retune the optimiser.

## Phase structure

The frozen Phase 6 structure is:

- **6A:** optimisation protocol freeze
- **6B:** trusted surrogate adapter and optimisation primitives
- **6C:** deterministic grid baseline and feasible-region characterisation
- **6D:** classical global optimisation benchmark and controlled selection
- **6E:** robustness, boundary, trade-off and infeasibility qualification
- **6F:** final optimiser runtime, persistence and reproducibility
- **6G:** final audit, formal Phase 6 closure and Phase 7 unlock

## Locked upstream evidence

Phase 4 and Phase 5 TEST evidence remains frozen.

Neither TEST dataset may be reopened for Phase 6 development or tuning.

Phase 5 monitoring models are not inputs to the Phase 6 optimisation
decision. Integration of monitoring, optimisation and agentic orchestration
belongs to the later project architecture and must not be introduced
silently during Phase 6.
