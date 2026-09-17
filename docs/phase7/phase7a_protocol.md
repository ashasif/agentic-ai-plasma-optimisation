# Phase 7A - Agentic System Architecture, Decision Policy and Safety Protocol

## Status

Protocol state: frozen_on_commit.

The normative machine-readable protocol is configs/phase7/phase7a_protocol.json.

No Phase 7 implementation is authorised until this protocol passes formal freeze verification and is committed.

## 1. Purpose

Phase 7 integrates the already-qualified Phase 5 monitoring and diagnosis runtime with the frozen Phase 6 constrained optimisation runtime.

It does not replace, retrain, retune or reinterpret the scientific logic of either upstream system.

The Phase 7 layer validates trusted evidence, applies deterministic policy, controls which permitted runtime operations may occur, and emits an auditable decision-support record.

## 2. Definition of the agent

The authoritative agent is a constrained deterministic decision orchestrator operating only over frozen trusted scientific runtimes and an explicit versioned safety policy.

The authoritative decision path is not an LLM or free-form reasoning system.

An optional natural-language layer may later explain an already completed structured decision, but it must not alter scientific predictions, constraints, feasibility, safety outcomes or approval state.

## 3. Decision-support boundary

Phase 7 is decision-support only.

It must never autonomously actuate plasma hardware.

It must never autonomously change an operating point.

Every proposed operating-point change requires explicit human approval.

Recording human approval does not grant Phase 7 hardware-actuation authority.

## 4. Scientific scope

The system remains synthetic, reduced-order and argon-only.

The frozen optimisation domain is absorbed power 15-90 W inclusive and pressure 10-60 mTorr inclusive.

Absorbed power is not generator RF power.

No claim may be made of experimental validation, industrial validation, production-fab validation, wafer-scale validation, reactive-chemistry validation, autonomous industrial plasma control, global physical optimality or global physical infeasibility.

## 5. Trusted Phase 5 boundary

The permitted Phase 5 production boundary is plasma_ai.monitoring.phase5g_persistence.

The loader is load_phase5g_monitoring and inference is performed only through LoadedPhase5GMonitoring.predict.

The frozen prediction fields are active_probability, fault_active and diagnostic_state.

The frozen diagnostic states are none, flow_delivery, power_coupling and pressure_path_anomaly.

The detector threshold is 0.4.

pressure_path_anomaly remains ambiguity-aware and must not be converted into a claim identifying a specific physical mechanism.

Development fitting, redevelopment, candidate selection, threshold selection, development-dataset loading and locked TEST re-execution are prohibited in Phase 7.

Phase 5 TEST must never be reopened.

## 6. Trusted Phase 6 boundary

The permitted optimisation boundary is plasma_ai.optimisation.runtime.

The production runtime is loaded through load_phase6_optimisation_runtime.

Requests must pass parse_runtime_request and execution occurs only through Phase6OptimisationRuntime.run.

The production method is Differential Evolution with seed 20260914.

The deterministic grid remains the mandatory reference and fallback.

SHGO is not production-qualified.

Continuous multiobjective optimisation and hidden scalarisation remain prohibited.

The exact frozen Phase 6 status vocabulary is:

- selected_method_accepted
- selected_method_accepted_grid_infeasible
- grid_fallback_selected_method_infeasible
- grid_fallback_selected_method_objective_regression
- no_feasible_point_found_under_search_protocol

Phase 7 must not add or reinterpret Phase 6 optimisation statuses.

## 7. Optimisation request governance

An optimisation request may originate only from an explicit operator request or a separately frozen deterministic policy template.

The agent may not invent free-form scientific objectives, constraints, target values or scalarisations.

All optimisation requests must pass the frozen Phase 6 validation contract.

## 8. Monitoring-driven policy

Missing, invalid, inconsistent or untrusted Phase 5 evidence causes fail-closed escalation and prevents a Phase 6 call.

If fault_active is false and diagnostic_state is none, the default monitoring-driven outcome is no_action.

If a valid active fault exists but no permitted optimisation request exists, the outcome is request_human_input.

If a valid active fault and permitted validated optimisation request both exist, Phase 6 evaluation may occur.

## 9. Phase 6 status policy

selected_method_accepted permits a recommendation pending human approval.

selected_method_accepted_grid_infeasible permits a recommendation pending human approval but requires an explicit warning that the mandatory grid reference was infeasible.

grid_fallback_selected_method_infeasible permits recommendation of the grid fallback, with explicit disclosure that the selected method was infeasible.

grid_fallback_selected_method_objective_regression permits recommendation of the grid fallback, with explicit disclosure that the selected method regressed against the mandatory reference.

no_feasible_point_found_under_search_protocol prohibits an operating-point recommendation and requires escalation.

That status is not evidence of global physical infeasibility.

Any unknown Phase 6 status causes fail-closed escalation.

## 10. Phase 7 decision outcomes

- no_action
- request_human_input
- recommend_change_pending_human_approval
- escalate_without_recommendation

These are Phase 7 governance outcomes and do not replace Phase 6 statuses.

## 11. Fail-closed policy

Phase 7 fails closed on missing required evidence, invalid schemas, non-finite inputs, upstream hash mismatches, manifest validation failure, unknown diagnostic states, invalid optimisation requests, out-of-domain requests, runtime failures, unknown Phase 6 statuses, missing mandatory Phase 6 reference evidence, missing provenance or policy-version mismatch.

Fail closed means that no operating-point recommendation is silently inferred.

## 12. Human approval

Every operating-point change recommendation requires human approval.

The agent cannot approve itself.

Approval states are not_applicable, pending, approved and rejected.

Even approved decisions remain decision-support records rather than hardware actuation commands.

## 13. Audit record

Every decision must preserve protocol identity, request identity, input evidence, Phase 5 runtime identity and prediction, monitoring-policy result, optimisation request identity, Phase 6 runtime identity and response, safety checks, warning codes, decision outcome, approval state, reason codes, provenance and replay fingerprint.

When Phase 6 runs, the complete structured Phase 6 response must be preserved rather than reducing evidence to only the chosen operating point.

## 14. Deterministic replay

Identical structured inputs, Phase 7 protocol, upstream artifacts, upstream protocols and production seed must produce an exactly matching canonical non-timing structured decision.

Wall-clock timestamps and optional free-form explanation text are excluded from authoritative replay identity.

## 15. Planned implementation namespace

The planned Phase 7 implementation namespace is src/plasma_ai/agentic.

That package must not be created during Phase 7A.

Implementation remains locked until Phase 7A is formally frozen.

## 16. Phase 7 structure

- Phase 7A - Agentic System Architecture, Decision Policy and Safety Protocol
- Phase 7B - Trusted Runtime Integration and Boundary Adapters
- Phase 7C - Deterministic Agentic Decision Orchestration
- Phase 7D - Safety, Human Approval and Audit Enforcement
- Phase 7E - Integrated Scenario Qualification
- Phase 7F - Deterministic Replay, Persistence and Final Integrated Demonstration
- Phase 7G - Final Project Audit, Consolidation and Formal Closure

The overall project must not be declared complete before formal Phase 7G closure.

## 17. Authoritative decision modes

Phase 7 has two explicit decision modes.

### monitoring_driven

A current valid trusted Phase 5G observation is mandatory.

The monitoring-driven path may consider Phase 6 only when a valid active fault is present and a separately frozen deterministic optimisation-request template supplies the request.

The diagnosis itself must never invent objectives or constraints.

### operator_initiated_optimisation

An explicit operator may supply a validated optimisation request independently of the Phase 5 fault trigger.

A current valid trusted Phase 5G observation remains mandatory as contextual safety evidence.

A nominal Phase 5 result does not prohibit the operator request, but the resulting decision must not be represented as fault-triggered.

## 18. Authoritative decision unit

Each authoritative Phase 7 decision record represents exactly one current monitoring observation.

Although the frozen Phase 5G runtime supports batch inference, the authoritative Phase 7 orchestration path accepts exactly one Phase 5 row per decision.

At most one Phase 6 optimisation scenario may be evaluated per decision record.

This narrowing does not alter the upstream Phase 5 batch interface.

## 19. Phase 5 boundary coherence

Phase 7 must verify that the Phase 5 prediction has exactly one row.

The frozen detector rule is fault_active equals active_probability greater than or equal to 0.4.

A mismatch between probability, binary decision and the frozen threshold is a fail-closed condition.

Prediction ordering must remain aligned with input-row ordering.

## 20. Phase 6 response coherence

selected_method_accepted and selected_method_accepted_grid_infeasible require chosen_source differential_evolution and a chosen operating point.

The two grid_fallback statuses require chosen_source deterministic_grid and a chosen operating point.

no_feasible_point_found_under_search_protocol requires no chosen source and no chosen operating point.

Phase 7 must never recompute the Phase 6 status or replace the runtime-selected source.

Any incoherent status/source response fails closed.

## 21. Canonical decision identity

Authoritative structured replay payloads use canonical UTF-8 JSON with lexicographically sorted object keys, compact separators, no non-finite numbers and no trailing newline inside the hashed payload.

decision_id is a SHA-256 content-addressed replay identity rather than a random event UUID.

Its identity payload contains the record schema version, decision mode, Phase 7 protocol SHA-256, Phase 5 manifest SHA-256, Phase 6 runtime-manifest SHA-256, raw monitoring observation and optimisation request or null.

replay_fingerprint is the SHA-256 of the canonical authoritative structured decision record excluding the replay_fingerprint itself, wall-clock timestamps, timing metadata, optional natural-language explanation and approval events.

## 22. Immutable human approval events

The authoritative scientific decision record is immutable after creation.

A recommendation record begins with human approval state pending.

No-action, request-human-input and escalation decisions use not_applicable.

A subsequent human approval or rejection is stored as a separate external approval event referencing decision_id.

The approval event does not mutate the original deterministic decision record and is excluded from authoritative replay equality.

An approval event still does not grant Phase 7 hardware-actuation authority.

## 23. Cross-system governance

Phase 5 diagnosis must not override Phase 6 feasibility.

Phase 6 optimisation must not reclassify the Phase 5 diagnosis.

No implicit mapping from a fault class to an optimisation objective or constraint is permitted.

Any such mapping must be separately explicit, deterministic and frozen before use.

An active fault combined with no feasible Phase 6 point produces escalation without recommendation.

Conflicting, malformed or incoherent cross-system evidence fails closed.

## 24. Final identity and approval-state consistency

The initial human-approval state mapping uses the exact Phase 7 decision-outcome identifiers.

- no_action -> not_applicable
- request_human_input -> not_applicable
- recommend_change_pending_human_approval -> pending
- escalate_without_recommendation -> not_applicable

No shorthand or alternate decision-outcome identifier is permitted in the authoritative record.

Canonical replay JSON uses the Python standard-library JSON serializer with sort_keys=True, separators=(comma, colon), ensure_ascii=True and allow_nan=False.

The hashed canonical payload contains no trailing newline.
