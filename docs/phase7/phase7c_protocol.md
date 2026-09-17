# Phase 7C — Deterministic Decision Policy and Orchestration

State: FROZEN ON COMMIT.

## 1. Purpose

Phase 7C defines deterministic orchestration over the frozen Phase 7B trusted monitoring and optimisation boundaries.

It does not create new scientific inference, retrain or retune upstream components, actuate hardware, change setpoints autonomously, or give an LLM decision authority.

## 2. Frozen parent evidence

- Phase 7A protocol SHA-256: b1978e0b77cfe7cff30fa91f0b2d8fba943d99a518a2d9b43e72543026456f31
- Phase 7A document SHA-256: 70c12235f423da839e662d6a4bffd21acb0d3797a807b4354eaae5061fdb315d
- Phase 7B protocol SHA-256: 732cc78f9bd0a72eb9d2d61af83baae7da0ff9e5bac58b526f3d079580a6e8a7
- Phase 7B document SHA-256: d764c2f0f67b3e2da7ecf1121fcadf7e5f831dc4560290174b5aefa387cc4e2d
- Phase 7B monitoring adapter SHA-256: 99198f9701e9b0fc73374de654585ed1e95de924c14f762e00fbfffbf6df4718
- Phase 7B optimisation adapter SHA-256: ead017c8a0abc8b0b0437de51dce59487bf4e9875a83c863d1dd274df115546b
- Phase 5 monitoring manifest SHA-256: a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259
- Phase 6 runtime manifest SHA-256: 5f4d7446d5c5a504c793d38cecda854b5b36b9dd65eeba6dcfe5a0c64b8d7395

## 3. Decision modes

Exactly two modes are permitted:

1. monitoring_driven
2. operator_initiated_optimisation

Every decision unit contains exactly one current monitoring observation, at most one Phase 6 request, and at most one human-approval target.

## 4. Decision outcomes

The authoritative outcome vocabulary is:

- no_action
- request_human_input
- recommend_change_pending_human_approval
- escalate_without_recommendation

## 5. Monitoring-driven policy

A monitoring-driven decision always evaluates the current observation through TrustedMonitoringAdapter.

If no fault is active, Phase 6 is not invoked and the outcome is no_action.

No implicit mapping from diagnostic state to optimisation objective, constraint or target is permitted.

Because no separately frozen fault-to-optimisation template currently exists, an active monitoring fault must not invoke Phase 6 in Phase 7C. The outcome is request_human_input and no operating-point recommendation may be invented.

A trusted monitoring-boundary failure escalates without recommendation.

## 6. Operator-initiated optimisation policy

An explicit operator optimisation request is required.

The current Phase 5 monitoring observation remains mandatory, but neither Phase 5 nor Phase 7C may rewrite the operator request.

Phase 6 may be invoked when the monitoring fault flag is inactive or active, provided the current Phase 5 observation is trusted and an explicit operator optimisation request is supplied. An active fault does not convert the explicit operator request into a fault-derived request and does not permit Phase 7C to invent or rewrite an objective, constraint or target.

A valid Phase 6 response with a selected or grid-fallback chosen operating point produces recommend_change_pending_human_approval.

The no-feasible status produces escalate_without_recommendation. It must not be represented as global physical infeasibility.

A missing operator optimisation request produces request_human_input. A supplied operator request is not semantically re-parsed, repaired or rewritten by Phase 7C; it is passed unchanged to TrustedOptimisationAdapter. If the trusted adapter rejects the supplied request during parsing, validation, integrity checking or execution, the fail-closed outcome is escalate_without_recommendation.

Trusted monitoring or optimisation boundary failure produces escalate_without_recommendation.

## 7. Recommendation authority

Every recommendation must come exactly from the Phase 6 chosen operating point. Phase 7C may not modify those coordinates, substitute another point, override chosen_source, or recompute Phase 6 status.

Every recommendation begins with human approval state pending.

Phase 7C has no hardware-actuation authority.

## 8. Immutable evidence handoff

Phase 7B intentionally returns immutable MappingProxyType mappings and tuples.

Phase 7C must construct a fresh JSON-native copy for authoritative recording: mappings recursively become plain dictionaries, tuples recursively become lists, immutable JSON scalar values are preserved, and non-finite floats are rejected.

The immutable Phase 7B evidence itself must never be mutated.

## 9. Canonical record identity

Canonical JSON uses Python standard-library json.dumps with sort_keys=True, separators=(comma, colon), ensure_ascii=True and allow_nan=False.

No trailing newline is included in hashed canonical payloads.

decision_id is SHA-256 over the frozen Phase 7A identity payload fields.

replay_fingerprint is SHA-256 over the canonical authoritative record excluding replay_fingerprint, wall-clock timestamp, timing metadata, optional natural-language explanation and approval events.

## 10. Human approval

The authoritative scientific decision record is immutable.

Approval or rejection is represented by a separate immutable event referencing decision_id. An approval event never mutates the original scientific decision and never grants Phase 7 hardware-actuation authority.

## 11. Failure policy

TrustedBoundaryError is not itself a scientific decision outcome.

There is no silent recovery, scientific-path substitution, retry with modified request, retry with different seed, or invented fallback.

At the Phase 7C orchestration boundary, TrustedBoundaryError may be caught only to produce the deterministic fail-closed escalation record defined by this protocol while preserving its boundary, stage and cause_type metadata. This translation is not scientific recovery, does not make the exception itself a scientific decision outcome, does not retry the scientific path and does not invent substitute evidence.

## 12. Scientific claim governance

Claims remain restricted to the frozen synthetic reduced-order argon ICP decision-support scope.

Phase 7C does not establish experimental, industrial, production-fab, wafer-scale or reactive-chemistry validity; global physical optimality or infeasibility; or autonomous industrial plasma control.

## 13. Implementation lock

No Phase 7C implementation is permitted until this protocol and its machine-readable counterpart are formally audited, frozen in Git, and independently verified.
