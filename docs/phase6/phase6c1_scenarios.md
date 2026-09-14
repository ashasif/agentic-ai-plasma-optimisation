# Phase 6C1 ? Deterministic-Grid Qualification Scenario Freeze

## Status

Phase 6C1 freezes the qualification scenarios that will be used by the
mandatory deterministic Phase 6 grid baseline.

No Phase 6C search has been executed in this gate.

Frozen scenario set:

`configs/phase6/grid_qualification_scenarios.json`

SHA-256:

`6af65defab31691ffbaef1fc91c7a685267d6169af45f277c097488252d0fc1a`

Effective Phase 6 contract:

`91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7`

## Why scenarios are frozen first

Qualification cases must be fixed before examining grid-search outcomes.

This prevents later adjustment of targets, constraints or objective ordering
to favour a particular optimiser or produce a preferred result.

Expected feasibility is deliberately not encoded.

A scenario may therefore prove feasible or infeasible under the frozen
15,251-point grid without the scenario definition being changed afterward.

## Frozen qualification cases

1. `soft_density_5e16`
2. `soft_density_1e17_low_power`
3. `soft_density_2e17_low_power`
4. `hard_density_1e17_min_power`
5. `hard_density_1p5e17_nominal`
6. `temperature_window_min_power`
7. `density_1p5e17_temperature_window`
8. `hard_density_interior_margin`
9. `boundary_margin_reference`
10. `nominal_centre_reference`

The cases collectively exercise:

- soft density targeting;
- lower, middle and higher density targets;
- hard density tolerances;
- absorbed-power minimisation;
- nominal-point deviation;
- one-sided and two-sided feasibility logic through the shared primitives;
- a two-sided electron-temperature range;
- an explicit normalized boundary-margin constraint;
- boundary-margin maximisation;
- structural nominal-point recovery.

## Frozen deterministic grid

The grid contains:

- 151 absorbed-power coordinates from 15 to 90 W inclusive;
- 101 pressure coordinates from 10 to 60 mTorr inclusive;
- 0.5 W absorbed-power spacing;
- 0.5 mTorr pressure spacing;
- 15,251 total candidate operating points.

Enumeration is frozen as:

1. absorbed power ascending;
2. for each power value, pressure ascending.

The candidate index is:

`power_index * 101 + pressure_index`

If every declared objective component is exactly equal, the lowest candidate
index wins. This final tie rule is deterministic bookkeeping and is not an
additional scientific objective.

## Multiobjective behaviour

Scenarios with more than one active objective use the already frozen
lexicographic ranking contract.

The grid baseline must additionally retain the non-dominated candidate set.

Dominance is evaluated in minimization-form objective space. Candidate A
dominates B only when A is no worse on every active component and strictly
better on at least one.

No hidden weighting and no numerical dominance tolerance are introduced.

## Evidence separation

This scenario freeze uses no:

- Phase 6C search result;
- Phase 4 TEST evidence;
- Phase 5 TEST evidence;
- source-simulator evaluation;
- experimental validation evidence;
- industrial validation evidence.

## Next step

Phase 6C2 may now implement and execute the mandatory deterministic grid
baseline against this exact frozen scenario set.

All 15,251 surrogate predictions should be generated once as a trusted batch
and then reused across every frozen scenario.
