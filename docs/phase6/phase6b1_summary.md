# Phase 6B1 — Effective Contract Loader, Scenario Validation and Pure Primitives

## Purpose

Phase 6B1 introduces the first executable Phase 6 optimisation package while
remaining below the search-algorithm and surrogate-inference layers.

The implementation is governed by the frozen effective Phase 6 contract:

- Phase 6A base protocol SHA-256:
  `57fee45c69c5f42f9e94b786e895a5d89b163d141367326706eac2d3d2ed1cb1`
- Phase 6A.1 amendment SHA-256:
  `9cf31578c6adddb195766fefa38ada89c6678e2df60bea708ddded0383c53912`
- effective contract SHA-256:
  `91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7`

## Implemented

Phase 6B1 adds:

- cryptographic loading and validation of the frozen effective protocol;
- exact scenario-field validation;
- explicit hard/soft/disabled density-target semantics;
- temperature-constraint validation;
- nominal operating-point validation;
- explicit objective dependency validation;
- strict two-variable qualified-domain checking;
- density relative-error calculation;
- normalized absorbed-power cost;
- normalized nominal-point distance;
- normalized boundary-margin calculation;
- boundary-margin minimization cost;
- pure candidate feasibility evaluation;
- hard-constraint precedence;
- lexicographic feasible-candidate objective keys.

## Domain behaviour

The qualified domain remains exactly:

- absorbed power: 15–90 W inclusive;
- target pressure: 10–60 mTorr inclusive.

No clipping is implemented.

Out-of-domain decision points are rejected before they can become valid
optimisation candidates.

## Scientific separation

Phase 6B1 performs no:

- persisted surrogate loading;
- surrogate inference;
- grid search;
- differential evolution;
- SHGO;
- source-model execution;
- optimiser-method selection;
- Phase 4 TEST access;
- Phase 5 TEST access;
- Phase 6 scientific-performance evaluation.

All unit tests use synthetic scalar predictions solely to test deterministic
mathematics and scenario semantics.

## Next step

The next controlled sub-gate is Phase 6B2: trusted Phase 4G surrogate adapter
and prediction-contract validation.

That adapter must use only the existing trusted Phase 4G persistence interface
and must not directly deserialize either model artifact.
