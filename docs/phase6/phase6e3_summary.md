# Phase 6E3 ? Real Robustness Qualification

## Status

Phase 6E3 executed the frozen robustness-qualification protocol against the
frozen production Phase 4G surrogate.

Result:

`results/phase6/robustness_qualification.json`

SHA-256:

`0b5cfffe64905499bb14a18b4f84824ccc0d071854669941ade4be9d5537a17d`

Execution source commit:

`dabe4354e285de549a8276d582fdae2ba30cfdc8`

Overall Phase 6E qualification:

**PASS**

## Selected optimizer

The selected continuous optimizer remained:

**Differential Evolution**

No optimizer parameter was retuned.

The three predeclared seeds were:

- `20260914`
- `20260915`
- `20260916`

Nine selected-method runs were executed in total.

## Challenge qualification

| Challenge | Expected outcome | Objective spread | Pass |
|---|---|---:|---|
| selected_method_replay | feasible | 7.52820028538e-12 | True |
| boundary_margin_constraint_edge | feasible | 5.29729038412e-13 | True |
| deliberate_infeasible_high_density | no_feasible_point_found_under_search_protocol | 0.00340094165711 | True |

The two expected-feasible challenges passed the frozen Phase 6 final-feasibility
checks, objective-spread requirement and deterministic-grid comparison rule.

The deliberately infeasible high-density challenge produced zero feasible
deterministic-grid candidates and no false feasible Differential Evolution
candidate.

This is evidence of infeasibility under the frozen surrogate/search protocol
only. It is not a claim of global physical impossibility.

## Shared deterministic-grid prediction batch

Exactly one complete 15,251-point production-surrogate grid prediction batch
was executed.

Input-array SHA-256:

`f77c780d9a7c4e4c6db1dc921317c0b48e6d2e397d14075abbe34d38a2472cfd`

Density-array SHA-256:

`70e60a5659f9ea5ac4f0c768d056119db7aacccf7c90d85e06996ab092008b90`

Temperature-array SHA-256:

`4f3d9e087554e378478bec2405d624a061e46ad49facfbcd1058c2b9a7ba7dd0`

That same prediction batch was reused for:

- all three Phase 6E grid challenge references;
- all five multiobjective tradeoff diagnostics.

## Perturbation diagnostics

Five frozen Phase 6D Differential Evolution operating points were examined.

Each source point accounted for all eight predeclared ?0.25 W / ?0.25 mTorr
perturbations.

No coordinate clipping occurred.

No out-of-domain surrogate inference occurred.

Out-of-domain perturbations, where present, were explicitly recorded as not
evaluated.

The perturbation diagnostics are sensitivity evidence. A locally infeasible
neighbour around a constraint-edge optimum is not by itself a failure.

## Multiobjective tradeoff diagnostics

Five deferred multiobjective scenarios were processed using exact
deterministic-grid Pareto semantics.

| Scenario | Feasible candidates | Pareto candidates | Frozen lexicographic point is Pareto |
|---|---:|---:|---|
| soft_density_1e17_low_power | 15251 | 53 | True |
| soft_density_2e17_low_power | 15251 | 118 | True |
| hard_density_1p5e17_nominal | 1061 | 11 | True |
| density_1p5e17_temperature_window | 5889 | 120 | True |
| hard_density_interior_margin | 1763 | 193 | True |

No:

- weighted scalarization;
- continuous multiobjective optimizer;
- knee-point selection;
- new production operating-point selection

was introduced.

## Qualification result

Selected-method robustness:

`True`

Infeasibility handling:

`True`

Perturbation diagnostics:

`True`

Tradeoff diagnostics:

`True`

Overall Phase 6E:

`True`

## Scientific boundaries

This qualification concerns optimization of the frozen reduced-order argon ICP
surrogate.

It does not establish:

- experimental validation;
- industrial validation;
- OIPT hardware performance;
- wafer-scale spatial behaviour;
- reactive chemistry behaviour;
- a mathematically proven global continuous optimum.

## Evidence separation

Phase 6E did not:

- access Phase 4 TEST targets;
- access the Phase 5 TEST dataset;
- execute the source plasma simulator;
- execute SHGO;
- retune Differential Evolution;
- use agentic orchestration.

## Next step

Phase 6F may now freeze the production optimization runtime contract using:

- Differential Evolution as the selected continuous method;
- deterministic grid as mandatory reference/fallback;
- the frozen Phase 6 objective and feasibility definitions;
- fail-closed domain and infeasibility handling.

Phase 6G and Phase 7 remain locked until the Phase 6 runtime is frozen and
audited.
