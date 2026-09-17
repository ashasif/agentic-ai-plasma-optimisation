# Phase 7C — Deterministic Decision Policy and Orchestration

**State:** FORMALLY CLOSED.

## 1. Scope

Phase 7C implements deterministic decision-support orchestration for the frozen synthetic reduced-order argon ICP project.

Phase 7C does not provide autonomous hardware control, autonomous setpoint changes, LLM decision authority, surrogate retraining, monitoring-model retraining, optimiser retuning, new scientific inference, reactive-chemistry claims, industrial validation claims, or extrapolation beyond the frozen domain.

## 2. Frozen protocol

- Phase 7C protocol commit: `75fcc3d2ad06cf075197bfa0ae31d3722fb09900`
- Phase 7C protocol SHA-256: `271b6620e7b86f34276a3735fbc248d6580a99f39115eb2f5732d9ed10744318`
- Phase 7C protocol document SHA-256: `a2b229a692522b3c01c2f8f3a522b130c2c4544005491b83e98ad51f56cda6ec`

The protocol was frozen before Phase 7C implementation.

## 3. Implemented components

### 3.1 Deterministic recording foundation

- Commit: `3bfc396133f22682022c74ba8a26fab494c3fc54`
- Source: `src/plasma_ai/agentic/recording.py`
- Source SHA-256: `d43c9aa0bf7d5705b154fac4318551432e97705ec9a88c80d397d531f5b32ab3`
- Test SHA-256: `4a0680d1ca60f01ebef679af597c85a5b4cecbad1d96e01009cfeba54f12f993`

This component provides deterministic JSON-native evidence conversion, canonical JSON serialisation, SHA-256 decision identity, SHA-256 replay fingerprints and immutable authoritative-record output.

### 3.2 Deterministic decision policy

- Commit: `6dd8aa5e318e704652c5be2602544eb58748658d`
- Source: `src/plasma_ai/agentic/decision_policy.py`
- Source SHA-256: `3fc72a2d315e1bb2684fb5c7b8b932015d8fd914207e947fbc1bce7ebe243bcb`
- Test SHA-256: `88b3f177483a420f894be4620996c83b7fca1f50dc695e9c1175f8cd74ddd16d`

The decision policy deterministically implements monitoring-driven and operator-initiated outcomes, Phase 6 status handling, human-approval state and fail-closed trusted-boundary handling.

### 3.3 Deterministic orchestrator

- Commit: `b72f684547fbd5aad16f6dc4e092035449bfb6e4`
- Source: `src/plasma_ai/agentic/orchestrator.py`
- Source SHA-256: `26addc8212792dd609187c27cfccbe747e219850736cc2525b8ed38cfdda8e62`
- Test SHA-256: `42ed084841419ab1ab88f073b6dcb99b5531dd8bcb574abcd2bd7c285cbb9ccf`

The orchestrator executes one authoritative monitoring observation per decision unit, at most one trusted Phase 5 evaluation and at most one trusted Phase 6 evaluation. It does not bypass the frozen trusted adapters.

## 4. Authoritative decision behavior

The implementation preserves the frozen 19-section authoritative-record contract.

Monitoring-driven behavior:

- inactive fault → `no_action`
- active fault without a separately frozen fault-to-optimisation template → `request_human_input`
- monitoring trusted-boundary failure → `escalate_without_recommendation`
- Phase 6 is not invoked from monitoring-driven mode under the current frozen protocol

Operator-initiated behavior:

- explicit operator optimisation request is required
- current trusted Phase 5 monitoring evidence remains required
- active or inactive monitoring state may proceed to the supplied operator request
- Phase 5 and Phase 7C do not rewrite the request
- Phase 6 statuses 1–4 produce `recommend_change_pending_human_approval`
- Phase 6 no-feasible status produces `escalate_without_recommendation`
- rejected or failed trusted Phase 6 requests fail closed without recommendation

## 5. Safety and governance

- Recommendations require human approval and begin with approval state `pending`.
- Non-recommendation outcomes use approval state `not_applicable`.
- Approval events are external to the immutable scientific decision record.
- Phase 7C has no hardware-actuation authority.
- Phase 7C does not recompute Phase 6 status.
- Phase 7C does not override the Phase 6 chosen source or fallback.
- `no_feasible_point_found_under_search_protocol` is not a claim of global physical infeasibility.
- Trusted-boundary failures do not retry, silently recover or invent substitute scientific evidence.

## 6. Qualification evidence

- Recording targeted tests: 20 passed.
- Decision-policy targeted tests: 35 passed.
- Orchestrator targeted tests: 31 passed.
- Complete agentic regression after orchestrator integration: 172 passed.
- Complete repository regression at final Phase 7C qualification: **793 passed**.
- Full-project pytest exit code: `0`.
- All audited frozen Phase 5, Phase 6, Phase 7A, Phase 7B and Phase 7C identities remained unchanged after the complete regression.
- Repository remained clean and unstaged after qualification.

## 7. Qualification conclusion

All implementation and scientific-governance requirements frozen for Phase 7C have been qualified.

Phase 7C implementation is complete and formally closed by the dedicated closure commit containing this document.

Phase 7C is formally closed. The frozen implementation, qualification evidence and scientific-governance boundaries remain unchanged by closure.
