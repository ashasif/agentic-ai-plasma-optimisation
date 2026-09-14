# Phase 6C2A — Deterministic Grid Engine Implementation and Qualification

## Purpose

Phase 6C2A implements the deterministic search machinery required for the
mandatory Phase 6 grid baseline.

This gate qualifies the implementation using synthetic prediction vectors only.

It does not generate Phase 6 scientific optimisation evidence.

## Frozen inputs

The implementation depends on:

- Phase 6A base protocol SHA-256:
  `57fee45c69c5f42f9e94b786e895a5d89b163d141367326706eac2d3d2ed1cb1`
- Phase 6A.1 amendment SHA-256:
  `9cf31578c6adddb195766fefa38ada89c6678e2df60bea708ddded0383c53912`
- effective Phase 6 contract SHA-256:
  `91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7`
- frozen Phase 6C1 scenario set SHA-256:
  `6af65defab31691ffbaef1fc91c7a685267d6169af45f277c097488252d0fc1a`

## Implemented deterministic grid

The engine constructs exactly:

- 151 absorbed-power coordinates;
- 101 pressure coordinates;
- 15,251 total candidates;
- 0.5 W absorbed-power spacing;
- 0.5 mTorr pressure spacing.

Enumeration is:

1. absorbed power ascending;
2. pressure ascending within each absorbed-power value.

Candidate indices are contiguous integers from 0 through 15,250.

The centre point `(52.5 W, 35 mTorr)` is candidate index 7,625.

## Candidate evaluation

The engine reuses the Phase 6B1 pure feasibility/objective primitives rather
than reimplementing constraint mathematics.

For each scenario it records in memory:

- total candidate count;
- feasible candidate count;
- infeasible candidate count;
- selected candidate index;
- selected candidate evaluation;
- selected lexicographic objective key;
- Pareto candidate indices for multiobjective scenarios;
- constraint-failure counts.

## Selection

Only feasible candidates are eligible for ranking.

Declared objectives are compared lexicographically.

If all declared objective components compare exactly equal, the lower frozen
candidate index is selected.

That final index rule is deterministic bookkeeping, not an extra scientific
objective.

## Pareto extraction

Pareto comparison uses minimization-form objective components and no numerical
dominance tolerance.

For the frozen one- and two-objective Phase 6C scenarios, deterministic exact
implementations are provided.

An exact generic fallback exists for higher-dimensional objective vectors, but
the current frozen Phase 6C1 set contains at most two objectives.

Returned Pareto indices preserve original candidate order.

## Evidence separation

Phase 6C2A uses synthetic prediction vectors in unit tests solely to qualify
search mechanics.

It performs no:

- trusted surrogate loading;
- real surrogate inference;
- real 15,251-point scientific optimisation search;
- Phase 6 result writing;
- SciPy optimisation;
- source-model execution;
- Phase 4 TEST access;
- Phase 5 TEST access.

## Next step

Once this implementation is committed, Phase 6C2B may execute the real
deterministic baseline from the committed code.

That gate will:

1. load the frozen 15,251-point grid;
2. load the trusted Phase 4G surrogate once;
3. issue one complete-grid batch prediction;
4. reuse those predictions across all ten frozen scenarios;
5. freeze the resulting optimisation evidence with full provenance.
