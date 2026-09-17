# Phase 7B - Trusted Runtime Integration and Boundary Adapters Protocol

## Status

Protocol state: frozen_on_commit.

Implementation remains locked until this protocol is formally frozen and committed.

The normative machine-readable protocol is configs/phase7/phase7b_protocol.json.

## 1. Purpose

Phase 7B creates narrow Phase 7-owned adapters around the frozen Phase 5G monitoring runtime and frozen Phase 6 production optimisation runtime.

Phase 7B performs integration and evidence validation only.

It does not implement decision orchestration, fault-to-objective mapping, Phase 6 status-to-decision mapping, human approval logic or hardware actuation.

## 2. Trusted boundary error

TrustedBoundaryError is the Phase 7 integration-layer fail-closed exception.

It is not itself a Phase 7 decision outcome.

It records boundary, stage and cause_type and chains the original ordinary Exception.

Phase 7B must not catch KeyboardInterrupt, SystemExit or GeneratorExit.

Phase 7B does not silently recover from a trusted-boundary failure.

## 3. Monitoring adapter

TrustedMonitoringAdapter is the only Phase 7B monitoring integration class.

It loads Phase 5 through plasma_ai.monitoring.phase5g_persistence.load_phase5g_monitoring.

It must not directly load the detector or diagnoser and must not use Phase 5 development interfaces.

Each adapter call accepts exactly one current monitoring observation containing exactly the six frozen raw monitoring fields.

The adapter converts that observation into exactly one upstream inference row.

The upstream Phase 5 batch capability remains intact but is not exposed by the authoritative Phase 7 adapter.

## 4. Monitoring evidence

TrustedMonitoringEvidence is immutable and Phase 7-owned.

It contains active_probability, fault_active, diagnostic_state and monitoring_manifest_sha256.

The adapter validates probability range, exact threshold coherence, diagnostic vocabulary and active/inactive state coherence before evidence may cross the boundary.

pressure_path_anomaly remains ambiguity-aware and does not identify a specific physical mechanism.

## 5. Optimisation adapter

TrustedOptimisationAdapter is the only Phase 7B optimisation integration class.

It loads Phase 6 only through plasma_ai.optimisation.runtime.load_phase6_optimisation_runtime.

It parses requests only through plasma_ai.optimisation.runtime.parse_runtime_request.

It executes optimisation only through Phase6OptimisationRuntime.run.

It must not directly import or call the surrogate adapter, grid evaluator, robustness runner, continuous optimiser or build_runtime_from_components.

## 6. Internal Phase 6 loader behaviour

The frozen Phase 6 production loader internally uses the qualified surrogate adapter to reconstruct and verify the deterministic grid.

That internal use is part of the trusted frozen Phase 6 runtime boundary and is permitted.

It does not grant Phase 7 direct surrogate access.

Phase 7 treats the production loader as an opaque trusted boundary.

## 7. Optimisation request handling

One adapter call handles one raw optimisation request.

The adapter must not rewrite the request or invent objectives, constraints, targets, scalarisations or domain expansions.

The raw request is copied before it is passed to the frozen Phase 6 parser.

The parsed OptimisationScenario object is not exposed outside the trusted adapter.

## 8. Optimisation evidence

TrustedOptimisationEvidence is immutable and Phase 7-owned.

It preserves the complete RuntimeResponse.to_dict output plus the frozen runtime-manifest SHA-256.

The adapter validates the five frozen statuses and their required chosen_source coherence.

The no-feasible status requires no chosen source and no chosen operating point.

Phase 7B must not recompute the Phase 6 status, override chosen_source or silently substitute a fallback.

## 9. Integrity

The adapters are anchored to the Phase 5G and Phase 6 frozen artifacts recorded by Phase 7A.

Any required hash mismatch fails closed through TrustedBoundaryError.

Phase 6 manifest validation remains authoritative for its complete frozen upstream dependency chain.

Phase 7B must not bypass that validation.

## 10. Import boundary

The monitoring adapter may import project runtime functionality only from plasma_ai.monitoring.phase5g_persistence.

The optimisation adapter may import project runtime functionality only from plasma_ai.optimisation.runtime.

Direct Phase 4 imports, Phase 5 development imports and low-level Phase 6 optimisation imports are prohibited.

## 11. Testing

Phase 7B tests live under tests/agentic.

Default adapter unit tests use controlled test doubles at the trusted boundary symbols.

Unit tests do not execute the real production optimisation runtime or reopen Phase 5 TEST.

Tests must cover valid evidence, malformed inputs, upstream failures, frozen hash mismatch, output incoherence and all five Phase 6 status/source combinations.

## 12. Scope exclusions

Phase 7B does not decide whether Phase 6 should be invoked.

Phase 7B does not map a diagnosis to an optimisation request.

Phase 7B does not issue a recommendation.

Phase 7B does not request, approve or reject an operating-point change.

Phase 7B does not actuate hardware.

## 13. Implementation unlock

No Phase 7B implementation may begin until this protocol passes formal freeze verification and is committed.

The frozen Phase 7A, Phase 5 and Phase 6 evidence must remain unchanged.

## 14. Formal freeze lifecycle

This protocol becomes formally frozen only when committed by the dedicated verified Phase 7B protocol-freeze commit.

Implementation is prohibited before that formal freeze and permitted after the verified freeze commit.

The freeze commit is the implementation-unlock event; no post-freeze protocol mutation is required.

## 15. Canonical shared error module

TrustedBoundaryError is defined once in plasma_ai.agentic.errors at src/plasma_ai/agentic/errors.py.

The monitoring and optimisation adapters use the same exception class.

The public Phase 7B modules are errors, monitoring_adapter and optimisation_adapter.

## 16. Adapter construction lifecycle

TrustedMonitoringAdapter has a zero-argument public constructor.

It eagerly loads the frozen Phase 5G runtime exactly once per adapter instance using the frozen manifest and protocol paths.

TrustedOptimisationAdapter also has a zero-argument public constructor.

It eagerly loads the frozen Phase 6 production runtime exactly once per adapter instance.

Neither adapter exposes public dependency injection or alternate production artifact paths.

Controlled unit tests replace only the trusted boundary symbols through monkeypatching.

## 17. Deep immutable optimisation evidence

TrustedOptimisationEvidence is a frozen dataclass and its complete preserved RuntimeResponse mapping is recursively immutable.

Mappings are represented as read-only mapping proxies, sequences as tuples, and only immutable JSON scalar values are retained.

No mutable upstream runtime object, mutable dictionary or mutable list may cross the trusted Phase 7B boundary.

Non-finite floating-point values are rejected.

## 18. Phase 6 provenance coherence

The preserved runtime response must contain runtime_manifest_sha256, effective_runtime_protocol_sha256 and phase4g_surrogate_manifest_sha256 provenance entries.

The response runtime_manifest_sha256 must equal the frozen Phase 6 runtime-manifest SHA-256.

TrustedOptimisationEvidence.runtime_manifest_sha256 must equal that same response provenance value.

Missing or inconsistent provenance fails closed through TrustedBoundaryError.

## 19. Shared error qualification

Phase 7B tests include a dedicated test_phase7_errors.py contract.

The tests verify ordinary Exception wrapping and chaining, preservation of boundary, stage and cause_type, and that KeyboardInterrupt and SystemExit are not caught by the trusted-boundary wrapper.
