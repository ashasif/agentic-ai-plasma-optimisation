# Phase 6A.1 ? Controlled Scenario-Contract Amendment

## Status

This is a controlled pre-implementation clarification of the frozen Phase 6A
optimisation protocol.

The original base protocol remains unchanged.

Base protocol:

`configs/phase6/optimisation_protocol.json`

Base SHA-256:

`57fee45c69c5f42f9e94b786e895a5d89b163d141367326706eac2d3d2ed1cb1`

Amendment:

`configs/phase6/optimisation_protocol_amendment_001.json`

Amendment SHA-256:

`9cf31578c6adddb195766fefa38ada89c6678e2df60bea708ddded0383c53912`

Effective Phase 6 contract identity:

`91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7`

The effective identity is SHA-256 of the ASCII sequence:

`<base hash>\n<amendment hash>\n`

## Reason for amendment

The Phase 6A base protocol correctly distinguished density targeting from
density-target tolerance, but it did not explicitly serialize whether a
density target is disabled, a soft optimisation objective, or a hard
constraint.

Implementing that distinction directly in source code would introduce a new
scientific decision after protocol freeze.

This amendment therefore resolves the ambiguity before optimiser
implementation and without using surrogate predictions, optimisation results,
TEST evidence, or source-model evidence.

## Density target modes

Every scenario must explicitly use exactly one of:

- `disabled`
- `soft_objective`
- `hard_constraint`

For `disabled`, both the density target and density tolerance are null.

For `soft_objective`, a positive density target is required and density
relative error must appear in the declared objective priority. No density
tolerance is applied as a hard constraint.

For `hard_constraint`, both a positive target and explicit positive relative
tolerance are required. The hard condition is:

`abs(predicted_density - target_density) / target_density <= tolerance`

Density error may additionally be ranked as an objective, but it is not
implicitly added.

## Temperature constraints

Minimum-only, maximum-only and two-sided temperature constraints are permitted.

When both bounds are present, minimum must not exceed maximum.

All active temperature constraints are inclusive hard constraints.

## Objective contract

The only allowed objective names are:

- `density_target_error`
- `nominal_distance`
- `absorbed_power`
- `boundary_margin`

`objective_priority` is an ordered unique list and must never be empty.

The ranking policy is lexicographic.

No hidden objective or hidden weighting is allowed.

`boundary_margin` is maximised, while its previously frozen
`boundary_margin_cost` representation may be used where a minimisation-form
component is required.

## Hard-constraint precedence

Hard feasibility is evaluated before objective ranking.

An infeasible candidate cannot outrank a feasible candidate because of a
better objective value.

Any candidate returned by a later search method must undergo an explicit final
feasibility recheck.

## Multiobjective restriction

The deterministic grid baseline may perform lexicographic ranking and must
retain the non-dominated set for multiobjective scenarios.

Phase 6B must not invent weighted continuous multiobjective optimisation.

Continuous Phase 6 benchmarking may initially use single-objective scenarios.
Any later continuous multiobjective technique requires another explicit
protocol freeze.

## Phase 6B boundary

Phase 6B may now implement:

- effective protocol loading;
- scenario validation;
- decision-point validation;
- objective primitives;
- hard-constraint evaluation;
- the trusted Phase 4G surrogate adapter.

Phase 6B may not yet implement:

- grid optimisation search;
- differential evolution;
- SHGO;
- optimiser-method selection;
- qualification-scenario tuning;
- agentic orchestration.

## Evidence separation

This amendment used no:

- Phase 4 TEST targets;
- Phase 5 TEST dataset;
- surrogate predictions;
- optimisation results;
- source-simulator evaluations;
- scientific-performance evidence.

Its purpose is solely to make the already intended optimisation semantics
machine-explicit before executable Phase 6 code is written.
