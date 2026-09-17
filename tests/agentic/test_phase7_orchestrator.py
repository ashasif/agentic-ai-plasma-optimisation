from types import MappingProxyType

import pytest

import plasma_ai.agentic.orchestrator as orchestrator_module
from plasma_ai.agentic.errors import TrustedBoundaryError
from plasma_ai.agentic.monitoring_adapter import TrustedMonitoringEvidence
from plasma_ai.agentic.optimisation_adapter import TrustedOptimisationEvidence
from plasma_ai.agentic.orchestrator import (
    AUTHORITATIVE_RECORD_SECTIONS,
    orchestrate_decision,
)
from plasma_ai.agentic.recording import (
    PHASE5_MANIFEST_SHA256,
    PHASE6_RUNTIME_MANIFEST_SHA256,
    PHASE7C_PROTOCOL_SHA256,
    build_decision_id,
    build_replay_fingerprint,
    decision_id_payload,
    to_json_native,
)


OBSERVATION = {"observation_token": 1.0}
REQUEST = {"objective": "operator_supplied_test_request"}


def _monitoring_evidence(*, active=False, state="none"):
    return TrustedMonitoringEvidence(
        active_probability=0.9 if active else 0.1,
        fault_active=active,
        diagnostic_state=state,
        monitoring_manifest_sha256=PHASE5_MANIFEST_SHA256,
    )


def _optimisation_evidence(status="selected_method_accepted"):
    response = MappingProxyType(
        {
            "status": status,
            "chosen_operating_point": (
                None
                if status == "no_feasible_point_found_under_search_protocol"
                else MappingProxyType(
                    {
                        "coordinate_a": 42.0,
                        "coordinate_b": 24.0,
                    }
                )
            ),
            "provenance": MappingProxyType(
                {
                    "runtime_manifest_sha256": PHASE6_RUNTIME_MANIFEST_SHA256,
                    "effective_runtime_protocol_sha256": "effective-protocol",
                    "phase4g_surrogate_manifest_sha256": "surrogate-manifest",
                }
            ),
            "opaque_extension": MappingProxyType(
                {
                    "preserve_me": True,
                    "sequence": (1, 2, 3),
                }
            ),
        }
    )

    return TrustedOptimisationEvidence(
        runtime_response=response,
        runtime_manifest_sha256=PHASE6_RUNTIME_MANIFEST_SHA256,
    )


def _install_fakes(
    monkeypatch,
    *,
    monitoring_evidence=None,
    optimisation_evidence=None,
    monitoring_init_error=None,
    monitoring_evaluate_error=None,
    optimisation_init_error=None,
    optimisation_evaluate_error=None,
):
    if monitoring_evidence is None:
        monitoring_evidence = _monitoring_evidence()

    if optimisation_evidence is None:
        optimisation_evidence = _optimisation_evidence()

    calls = {
        "monitoring_instances": 0,
        "monitoring_evaluations": 0,
        "optimisation_instances": 0,
        "optimisation_evaluations": 0,
        "monitoring_argument": None,
        "optimisation_argument": None,
    }

    class FakeMonitoringAdapter:
        def __init__(self):
            calls["monitoring_instances"] += 1
            if monitoring_init_error is not None:
                raise monitoring_init_error

        def evaluate(self, observation):
            calls["monitoring_evaluations"] += 1
            calls["monitoring_argument"] = observation

            if monitoring_evaluate_error is not None:
                raise monitoring_evaluate_error

            return monitoring_evidence

    class FakeOptimisationAdapter:
        def __init__(self):
            calls["optimisation_instances"] += 1
            if optimisation_init_error is not None:
                raise optimisation_init_error

        def evaluate(self, request):
            calls["optimisation_evaluations"] += 1
            calls["optimisation_argument"] = request

            if optimisation_evaluate_error is not None:
                raise optimisation_evaluate_error

            return optimisation_evidence

    monkeypatch.setattr(
        orchestrator_module,
        "TrustedMonitoringAdapter",
        FakeMonitoringAdapter,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "TrustedOptimisationAdapter",
        FakeOptimisationAdapter,
    )

    return calls


def _assert_record_integrity(record):
    assert tuple(record.keys()) == AUTHORITATIVE_RECORD_SECTIONS
    assert record["replay_fingerprint"] == build_replay_fingerprint(record)
    assert record["phase7_protocol_identity"] == {
        "phase7_protocol_sha256": PHASE7C_PROTOCOL_SHA256
    }
    assert record["safety_checks"]["no_hardware_actuation_authority"] is True
    assert record["safety_checks"]["decision_unit_limits_respected"] is True

    with pytest.raises(TypeError):
        record["decision_outcome"] = "changed"


def test_authoritative_section_contract_is_exact():
    assert AUTHORITATIVE_RECORD_SECTIONS == (
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


def test_monitoring_inactive_is_single_phase5_call_and_no_phase6(monkeypatch):
    calls = _install_fakes(monkeypatch)

    record = orchestrate_decision(
        decision_mode="monitoring_driven",
        monitoring_observation=OBSERVATION,
    )

    assert calls["monitoring_instances"] == 1
    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_instances"] == 0
    assert calls["optimisation_evaluations"] == 0
    assert record["decision_outcome"] == "no_action"
    assert record["human_approval_state"] == "not_applicable"
    assert record["phase6_response"] is None
    _assert_record_integrity(record)


@pytest.mark.parametrize(
    "state",
    ["flow_delivery", "power_coupling", "pressure_path_anomaly"],
)
def test_monitoring_active_without_template_requests_human_input(monkeypatch, state):
    calls = _install_fakes(
        monkeypatch,
        monitoring_evidence=_monitoring_evidence(active=True, state=state),
    )

    record = orchestrate_decision(
        decision_mode="monitoring_driven",
        monitoring_observation=OBSERVATION,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_evaluations"] == 0
    assert record["decision_outcome"] == "request_human_input"
    assert record["human_approval_state"] == "not_applicable"
    _assert_record_integrity(record)


def test_monitoring_mode_with_operator_request_escalates_without_phase6(monkeypatch):
    calls = _install_fakes(monkeypatch)

    record = orchestrate_decision(
        decision_mode="monitoring_driven",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_evaluations"] == 0
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["reason_codes"] == ("DECISION_MODE_REQUEST_SOURCE_INCOHERENT",)


@pytest.mark.parametrize(
    ("active", "state"),
    [(False, "none"), (True, "flow_delivery")],
)
def test_operator_missing_request_requests_human_input(monkeypatch, active, state):
    calls = _install_fakes(
        monkeypatch,
        monitoring_evidence=_monitoring_evidence(active=active, state=state),
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_evaluations"] == 0
    assert record["decision_outcome"] == "request_human_input"
    assert record["reason_codes"] == ("MISSING_OPERATOR_OPTIMISATION_REQUEST",)


@pytest.mark.parametrize(
    ("active", "state"),
    [(False, "none"), (True, "power_coupling")],
)
def test_operator_request_permits_one_phase6_call_for_active_or_inactive_fault(
    monkeypatch,
    active,
    state,
):
    calls = _install_fakes(
        monkeypatch,
        monitoring_evidence=_monitoring_evidence(active=active, state=state),
    )

    request = {"operator": {"objective": "unchanged"}}

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=request,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_instances"] == 1
    assert calls["optimisation_evaluations"] == 1
    assert calls["optimisation_argument"] is request
    assert record["decision_outcome"] == "recommend_change_pending_human_approval"
    assert record["human_approval_state"] == "pending"
    _assert_record_integrity(record)


@pytest.mark.parametrize(
    "status",
    [
        "selected_method_accepted",
        "selected_method_accepted_grid_infeasible",
        "grid_fallback_selected_method_infeasible",
        "grid_fallback_selected_method_objective_regression",
    ],
)
def test_phase6_success_statuses_recommend_pending_approval(monkeypatch, status):
    calls = _install_fakes(
        monkeypatch,
        optimisation_evidence=_optimisation_evidence(status),
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["optimisation_evaluations"] == 1
    assert record["decision_outcome"] == "recommend_change_pending_human_approval"
    assert record["human_approval_state"] == "pending"
    assert record["safety_checks"]["recommendation_has_trusted_phase6_response"] is True


def test_no_feasible_phase6_status_escalates_without_recommendation(monkeypatch):
    calls = _install_fakes(
        monkeypatch,
        optimisation_evidence=_optimisation_evidence(
            "no_feasible_point_found_under_search_protocol"
        ),
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["optimisation_evaluations"] == 1
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["human_approval_state"] == "not_applicable"
    assert record["phase6_response"]["chosen_operating_point"] is None


def test_complete_phase6_response_is_preserved(monkeypatch):
    evidence = _optimisation_evidence()
    _install_fakes(monkeypatch, optimisation_evidence=evidence)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert to_json_native(record["phase6_response"]) == to_json_native(
        evidence.runtime_response
    )
    assert record["phase6_response"]["opaque_extension"]["preserve_me"] is True
    assert tuple(record["phase6_response"]["opaque_extension"]["sequence"]) == (1, 2, 3)


def test_monitoring_observation_is_passed_unchanged_to_trusted_adapter(monkeypatch):
    calls = _install_fakes(monkeypatch)
    observation = {"raw": {"value": 123}}

    orchestrate_decision(
        decision_mode="monitoring_driven",
        monitoring_observation=observation,
    )

    assert calls["monitoring_argument"] is observation


def test_decision_id_and_request_identity_use_frozen_payload(monkeypatch):
    _install_fakes(monkeypatch)

    observation = {"raw": [1, 2, 3]}
    request = {"request": {"x": 1}}

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=observation,
        optimisation_request=request,
    )

    expected_payload = decision_id_payload(
        decision_mode="operator_initiated_optimisation",
        raw_monitoring_observation=observation,
        optimisation_request_or_null=request,
    )
    expected_id = build_decision_id(
        decision_mode="operator_initiated_optimisation",
        raw_monitoring_observation=observation,
        optimisation_request_or_null=request,
    )

    assert to_json_native(record["request_identity"]) == expected_payload
    assert record["decision_id"] == expected_id


def test_authoritative_input_snapshot_is_not_changed_by_later_input_mutation(monkeypatch):
    _install_fakes(monkeypatch)

    observation = {"raw": {"value": 1}}
    request = {"request": {"value": 2}}

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=observation,
        optimisation_request=request,
    )

    observation["raw"]["value"] = 999
    request["request"]["value"] = 999

    assert record["input_evidence"]["raw_monitoring_observation"]["raw"]["value"] == 1
    assert (
        record["optimisation_request_identity"]["optimisation_request_or_null"]
        ["request"]["value"]
        == 2
    )


def test_phase5_boundary_failure_escalates_without_substitute_evidence(monkeypatch):
    error = TrustedBoundaryError(
        boundary="phase5_monitoring",
        stage="prediction",
        cause_type="RuntimeError",
    )

    calls = _install_fakes(
        monkeypatch,
        monitoring_evaluate_error=error,
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_instances"] == 0
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["phase5_runtime_identity"] is None
    assert record["phase5_prediction"] is None
    assert record["phase6_response"] is None
    assert record["monitoring_policy_result"] is None
    assert record["upstream_provenance"]["trusted_boundary_failure"] == {
        "boundary": "phase5_monitoring",
        "stage": "prediction",
        "cause_type": "RuntimeError",
    }
    _assert_record_integrity(record)


def test_phase5_constructor_boundary_failure_is_recorded(monkeypatch):
    error = TrustedBoundaryError(
        boundary="phase5_monitoring",
        stage="load",
        cause_type="ValueError",
    )

    calls = _install_fakes(
        monkeypatch,
        monitoring_init_error=error,
    )

    record = orchestrate_decision(
        decision_mode="monitoring_driven",
        monitoring_observation=OBSERVATION,
    )

    assert calls["monitoring_instances"] == 1
    assert calls["monitoring_evaluations"] == 0
    assert calls["optimisation_instances"] == 0
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["upstream_provenance"]["trusted_boundary_failure"]["stage"] == "load"


def test_phase6_boundary_failure_preserves_phase5_but_has_no_phase6_evidence(monkeypatch):
    error = TrustedBoundaryError(
        boundary="phase6_optimisation",
        stage="parse",
        cause_type="ValueError",
    )

    calls = _install_fakes(
        monkeypatch,
        optimisation_evaluate_error=error,
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["monitoring_evaluations"] == 1
    assert calls["optimisation_evaluations"] == 1
    assert record["phase5_prediction"] is not None
    assert record["phase6_runtime_identity"] is None
    assert record["phase6_response"] is None
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["upstream_provenance"]["trusted_boundary_failure"] == {
        "boundary": "phase6_optimisation",
        "stage": "parse",
        "cause_type": "ValueError",
    }
    assert record["monitoring_policy_result"]["terminal_outcome"] is None


def test_phase6_constructor_boundary_failure_is_recorded_without_retry(monkeypatch):
    error = TrustedBoundaryError(
        boundary="phase6_optimisation",
        stage="integrity",
        cause_type="RuntimeError",
    )

    calls = _install_fakes(
        monkeypatch,
        optimisation_init_error=error,
    )

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert calls["optimisation_instances"] == 1
    assert calls["optimisation_evaluations"] == 0
    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert (
        record["upstream_provenance"]["trusted_boundary_failure"]["stage"]
        == "integrity"
    )


def test_phase6_response_provenance_is_carried_into_upstream_provenance(monkeypatch):
    evidence = _optimisation_evidence()
    _install_fakes(monkeypatch, optimisation_evidence=evidence)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert (
        to_json_native(record["upstream_provenance"]["phase6_response_provenance"])
        == to_json_native(evidence.runtime_response["provenance"])
    )


def test_phase5_and_phase6_runtime_identities_are_exact(monkeypatch):
    _install_fakes(monkeypatch)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert record["phase5_runtime_identity"] == {
        "phase5_manifest_sha256": PHASE5_MANIFEST_SHA256
    }
    assert record["phase6_runtime_identity"] == {
        "phase6_runtime_manifest_sha256": PHASE6_RUNTIME_MANIFEST_SHA256
    }


def test_unknown_phase6_status_fails_closed(monkeypatch):
    evidence = _optimisation_evidence("future_unknown_status")
    _install_fakes(monkeypatch, optimisation_evidence=evidence)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert record["decision_outcome"] == "escalate_without_recommendation"
    assert record["reason_codes"] == ("UNRECOGNISED_PHASE6_STATUS",)


def test_unsupported_decision_mode_is_rejected_before_adapter_calls(monkeypatch):
    calls = _install_fakes(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported Phase 7 decision mode"):
        orchestrate_decision(
            decision_mode="unsupported",
            monitoring_observation=OBSERVATION,
        )

    assert calls["monitoring_instances"] == 0
    assert calls["optimisation_instances"] == 0


def test_nonfinite_identity_input_is_rejected_before_adapter_calls(monkeypatch):
    calls = _install_fakes(monkeypatch)

    with pytest.raises(ValueError, match="finite"):
        orchestrate_decision(
            decision_mode="monitoring_driven",
            monitoring_observation={"value": float("nan")},
        )

    assert calls["monitoring_instances"] == 0
    assert calls["optimisation_instances"] == 0


def test_monitoring_policy_result_is_preserved_for_phase6_path(monkeypatch):
    _install_fakes(monkeypatch)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    result = record["monitoring_policy_result"]
    assert result["terminal_outcome"] is None
    assert result["phase6_call_permitted"] is True
    assert result["recommendation_permitted"] is False
    assert result["human_approval_state"] is None


def test_no_approval_event_is_embedded_in_authoritative_record(monkeypatch):
    _install_fakes(monkeypatch)

    record = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert "approval_events" not in record
    assert record["human_approval_state"] == "pending"


def test_replay_fingerprint_is_deterministic_across_identical_decisions(monkeypatch):
    _install_fakes(monkeypatch)

    first = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    _install_fakes(monkeypatch)

    second = orchestrate_decision(
        decision_mode="operator_initiated_optimisation",
        monitoring_observation=OBSERVATION,
        optimisation_request=REQUEST,
    )

    assert first["decision_id"] == second["decision_id"]
    assert first["replay_fingerprint"] == second["replay_fingerprint"]
