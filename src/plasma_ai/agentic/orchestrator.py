"""Deterministic single-decision-unit Phase 7C orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .decision_policy import (
    BoundaryFailureDecision,
    PolicyDecision,
    boundary_failure_decision,
    evaluate_phase6_status,
    evaluate_pre_optimisation,
)
from .errors import TrustedBoundaryError
from .monitoring_adapter import (
    TrustedMonitoringAdapter,
    TrustedMonitoringEvidence,
)
from .optimisation_adapter import (
    TrustedOptimisationAdapter,
    TrustedOptimisationEvidence,
)
from .recording import (
    PHASE5_MANIFEST_SHA256,
    PHASE6_RUNTIME_MANIFEST_SHA256,
    PHASE7C_PROTOCOL_SHA256,
    RECORD_SCHEMA_VERSION,
    build_decision_id,
    build_replay_fingerprint,
    decision_id_payload,
    deep_freeze,
    to_json_native,
)


AUTHORITATIVE_RECORD_SECTIONS: Final = (
    "record_schema_version",
    "decision_id",
    "phase7_protocol_identity",
    "request_identity",
    "decision_mode",
    "input_evidence",
    "phase5_runtime_identity",
    "phase5_prediction",
    "monitoring_policy_result",
    "optimisation_request_identity",
    "phase6_runtime_identity",
    "phase6_response",
    "safety_checks",
    "warning_codes",
    "decision_outcome",
    "human_approval_state",
    "reason_codes",
    "upstream_provenance",
    "replay_fingerprint",
)


def _policy_to_native(decision: PolicyDecision) -> dict[str, object]:
    return {
        "terminal_outcome": decision.terminal_outcome,
        "phase6_call_permitted": decision.phase6_call_permitted,
        "recommendation_permitted": decision.recommendation_permitted,
        "human_approval_state": decision.human_approval_state,
        "reason_codes": list(decision.reason_codes),
        "warning_codes": list(decision.warning_codes),
    }


def _monitoring_to_native(
    evidence: TrustedMonitoringEvidence,
) -> dict[str, object]:
    return {
        "active_probability": evidence.active_probability,
        "fault_active": evidence.fault_active,
        "diagnostic_state": evidence.diagnostic_state,
        "monitoring_manifest_sha256": evidence.monitoring_manifest_sha256,
    }


def _boundary_to_native(
    failure: BoundaryFailureDecision | None,
) -> dict[str, object] | None:
    if failure is None:
        return None

    return {
        "boundary": failure.boundary,
        "stage": failure.stage,
        "cause_type": failure.cause_type,
    }


def _build_authoritative_record(
    *,
    decision_mode: str,
    observation_snapshot: object,
    request_snapshot: object,
    phase5_evidence: TrustedMonitoringEvidence | None,
    monitoring_policy_result: PolicyDecision | None,
    phase6_evidence: TrustedOptimisationEvidence | None,
    final_decision: PolicyDecision,
    boundary_failure: BoundaryFailureDecision | None,
) -> Mapping[str, object]:
    if final_decision.terminal_outcome is None:
        raise ValueError("Authoritative record requires a terminal decision.")

    request_identity = decision_id_payload(
        decision_mode=decision_mode,
        raw_monitoring_observation=observation_snapshot,
        optimisation_request_or_null=request_snapshot,
    )

    decision_id = build_decision_id(
        decision_mode=decision_mode,
        raw_monitoring_observation=observation_snapshot,
        optimisation_request_or_null=request_snapshot,
    )

    if phase5_evidence is None:
        phase5_runtime_identity = None
        phase5_prediction = None
        phase5_manifest_sha256 = None
    else:
        phase5_manifest_sha256 = phase5_evidence.monitoring_manifest_sha256
        phase5_runtime_identity = {
            "phase5_manifest_sha256": phase5_manifest_sha256,
        }
        phase5_prediction = _monitoring_to_native(phase5_evidence)

    if phase6_evidence is None:
        phase6_runtime_identity = None
        phase6_response = None
        phase6_runtime_manifest_sha256 = None
        phase6_response_provenance = None
    else:
        phase6_runtime_manifest_sha256 = phase6_evidence.runtime_manifest_sha256
        phase6_runtime_identity = {
            "phase6_runtime_manifest_sha256": phase6_runtime_manifest_sha256,
        }
        phase6_response = to_json_native(phase6_evidence.runtime_response)

        if not isinstance(phase6_response, dict):
            raise TypeError("Trusted Phase 6 response must serialise to a mapping.")

        phase6_response_provenance = phase6_response.get("provenance")

    recommendation_has_trusted_phase6_response = (
        (not final_decision.recommendation_permitted)
        or (phase6_response is not None)
    )

    record: dict[str, object] = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "decision_id": decision_id,
        "phase7_protocol_identity": {
            "phase7_protocol_sha256": PHASE7C_PROTOCOL_SHA256,
        },
        "request_identity": request_identity,
        "decision_mode": decision_mode,
        "input_evidence": {
            "raw_monitoring_observation": to_json_native(observation_snapshot),
        },
        "phase5_runtime_identity": phase5_runtime_identity,
        "phase5_prediction": phase5_prediction,
        "monitoring_policy_result": (
            None
            if monitoring_policy_result is None
            else _policy_to_native(monitoring_policy_result)
        ),
        "optimisation_request_identity": {
            "optimisation_request_or_null": to_json_native(request_snapshot),
        },
        "phase6_runtime_identity": phase6_runtime_identity,
        "phase6_response": phase6_response,
        "safety_checks": {
            "no_hardware_actuation_authority": True,
            "human_approval_required_for_recommendations": True,
            "decision_unit_limits_respected": True,
            "recommendation_has_trusted_phase6_response": (
                recommendation_has_trusted_phase6_response
            ),
        },
        "warning_codes": list(final_decision.warning_codes),
        "decision_outcome": final_decision.terminal_outcome,
        "human_approval_state": final_decision.human_approval_state,
        "reason_codes": list(final_decision.reason_codes),
        "upstream_provenance": {
            "phase5_manifest_sha256": phase5_manifest_sha256,
            "phase6_runtime_manifest_sha256": phase6_runtime_manifest_sha256,
            "phase6_response_provenance": phase6_response_provenance,
            "trusted_boundary_failure": _boundary_to_native(boundary_failure),
        },
        "replay_fingerprint": None,
    }

    if tuple(record.keys()) != AUTHORITATIVE_RECORD_SECTIONS:
        raise RuntimeError("Authoritative record section order or membership changed.")

    if not recommendation_has_trusted_phase6_response:
        raise RuntimeError("Recommendation lacks trusted Phase 6 evidence.")

    record["replay_fingerprint"] = build_replay_fingerprint(record)

    frozen = deep_freeze(record)

    if not isinstance(frozen, Mapping):
        raise TypeError("Frozen authoritative record must remain a mapping.")

    return frozen


def _boundary_failure_record(
    *,
    exc: TrustedBoundaryError,
    decision_mode: str,
    observation_snapshot: object,
    request_snapshot: object,
    phase5_evidence: TrustedMonitoringEvidence | None,
    monitoring_policy_result: PolicyDecision | None,
) -> Mapping[str, object]:
    failure = boundary_failure_decision(
        boundary=exc.boundary,
        stage=exc.stage,
        cause_type=exc.cause_type,
    )

    return _build_authoritative_record(
        decision_mode=decision_mode,
        observation_snapshot=observation_snapshot,
        request_snapshot=request_snapshot,
        phase5_evidence=phase5_evidence,
        monitoring_policy_result=monitoring_policy_result,
        phase6_evidence=None,
        final_decision=failure.decision,
        boundary_failure=failure,
    )


def orchestrate_decision(
    *,
    decision_mode: str,
    monitoring_observation: Mapping[str, object],
    optimisation_request: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Execute exactly one frozen Phase 7C decision unit."""

    observation_snapshot = to_json_native(monitoring_observation)
    request_snapshot = to_json_native(optimisation_request)

    # Validate the frozen identity payload before crossing scientific boundaries.
    decision_id_payload(
        decision_mode=decision_mode,
        raw_monitoring_observation=observation_snapshot,
        optimisation_request_or_null=request_snapshot,
    )

    try:
        monitoring_adapter = TrustedMonitoringAdapter()
        phase5_evidence = monitoring_adapter.evaluate(monitoring_observation)
    except TrustedBoundaryError as exc:
        return _boundary_failure_record(
            exc=exc,
            decision_mode=decision_mode,
            observation_snapshot=observation_snapshot,
            request_snapshot=request_snapshot,
            phase5_evidence=None,
            monitoring_policy_result=None,
        )

    monitoring_policy_result = evaluate_pre_optimisation(
        decision_mode=decision_mode,
        fault_active=phase5_evidence.fault_active,
        diagnostic_state=phase5_evidence.diagnostic_state,
        operator_request_present=optimisation_request is not None,
    )

    if monitoring_policy_result.terminal_outcome is not None:
        return _build_authoritative_record(
            decision_mode=decision_mode,
            observation_snapshot=observation_snapshot,
            request_snapshot=request_snapshot,
            phase5_evidence=phase5_evidence,
            monitoring_policy_result=monitoring_policy_result,
            phase6_evidence=None,
            final_decision=monitoring_policy_result,
            boundary_failure=None,
        )

    if not monitoring_policy_result.phase6_call_permitted:
        raise RuntimeError("Non-terminal policy did not permit Phase 6.")

    if optimisation_request is None:
        raise RuntimeError("Phase 6 directive requires an operator request.")

    try:
        optimisation_adapter = TrustedOptimisationAdapter()
        phase6_evidence = optimisation_adapter.evaluate(optimisation_request)
    except TrustedBoundaryError as exc:
        return _boundary_failure_record(
            exc=exc,
            decision_mode=decision_mode,
            observation_snapshot=observation_snapshot,
            request_snapshot=request_snapshot,
            phase5_evidence=phase5_evidence,
            monitoring_policy_result=monitoring_policy_result,
        )

    status = phase6_evidence.runtime_response["status"]
    final_decision = evaluate_phase6_status(status)

    return _build_authoritative_record(
        decision_mode=decision_mode,
        observation_snapshot=observation_snapshot,
        request_snapshot=request_snapshot,
        phase5_evidence=phase5_evidence,
        monitoring_policy_result=monitoring_policy_result,
        phase6_evidence=phase6_evidence,
        final_decision=final_decision,
        boundary_failure=None,
    )
