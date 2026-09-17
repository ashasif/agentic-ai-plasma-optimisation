from dataclasses import FrozenInstanceError

import pytest

from plasma_ai.agentic.decision_policy import (
    BOUNDARIES,
    DECISION_MODES,
    DECISION_OUTCOMES,
    DIAGNOSTIC_STATES,
    PHASE6_STATUSES,
    BoundaryFailureDecision,
    PolicyDecision,
    boundary_failure_decision,
    evaluate_phase6_status,
    evaluate_pre_optimisation,
)


def test_frozen_vocabularies_are_exact():
    assert DECISION_MODES == frozenset(
        {"monitoring_driven", "operator_initiated_optimisation"}
    )
    assert DECISION_OUTCOMES == frozenset(
        {
            "no_action",
            "request_human_input",
            "recommend_change_pending_human_approval",
            "escalate_without_recommendation",
        }
    )
    assert DIAGNOSTIC_STATES == frozenset(
        {"none", "flow_delivery", "power_coupling", "pressure_path_anomaly"}
    )
    assert PHASE6_STATUSES == frozenset(
        {
            "selected_method_accepted",
            "selected_method_accepted_grid_infeasible",
            "grid_fallback_selected_method_infeasible",
            "grid_fallback_selected_method_objective_regression",
            "no_feasible_point_found_under_search_protocol",
        }
    )
    assert BOUNDARIES == frozenset({"phase5_monitoring", "phase6_optimisation"})


def test_policy_decision_is_immutable():
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=False,
        diagnostic_state="none",
        operator_request_present=False,
    )

    with pytest.raises(FrozenInstanceError):
        decision.terminal_outcome = "request_human_input"


def test_monitoring_inactive_returns_no_action_without_phase6():
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=False,
        diagnostic_state="none",
        operator_request_present=False,
    )

    assert decision.terminal_outcome == "no_action"
    assert decision.phase6_call_permitted is False
    assert decision.recommendation_permitted is False
    assert decision.human_approval_state == "not_applicable"
    assert decision.reason_codes == ("MONITORING_FAULT_INACTIVE",)


@pytest.mark.parametrize(
    "diagnostic_state",
    ["flow_delivery", "power_coupling", "pressure_path_anomaly"],
)
def test_monitoring_active_without_template_requests_human_input(diagnostic_state):
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=True,
        diagnostic_state=diagnostic_state,
        operator_request_present=False,
    )

    assert decision.terminal_outcome == "request_human_input"
    assert decision.phase6_call_permitted is False
    assert decision.recommendation_permitted is False
    assert decision.human_approval_state == "not_applicable"


@pytest.mark.parametrize(
    ("fault_active", "diagnostic_state"),
    [(True, "none"), (False, "flow_delivery")],
)
def test_incoherent_monitoring_evidence_fails_closed(fault_active, diagnostic_state):
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=fault_active,
        diagnostic_state=diagnostic_state,
        operator_request_present=False,
    )

    assert decision.terminal_outcome == "escalate_without_recommendation"
    assert decision.reason_codes == ("MONITORING_EVIDENCE_INCOHERENT",)


def test_unknown_diagnostic_state_fails_closed():
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=True,
        diagnostic_state="unknown",
        operator_request_present=False,
    )

    assert decision.terminal_outcome == "escalate_without_recommendation"
    assert decision.reason_codes == ("MONITORING_EVIDENCE_INCOHERENT",)


def test_monitoring_mode_rejects_operator_request_source_by_escalating():
    decision = evaluate_pre_optimisation(
        decision_mode="monitoring_driven",
        fault_active=False,
        diagnostic_state="none",
        operator_request_present=True,
    )

    assert decision.terminal_outcome == "escalate_without_recommendation"
    assert decision.phase6_call_permitted is False
    assert decision.reason_codes == ("DECISION_MODE_REQUEST_SOURCE_INCOHERENT",)


@pytest.mark.parametrize(
    ("fault_active", "diagnostic_state"),
    [(False, "none"), (True, "flow_delivery")],
)
def test_operator_missing_request_requests_human_input(fault_active, diagnostic_state):
    decision = evaluate_pre_optimisation(
        decision_mode="operator_initiated_optimisation",
        fault_active=fault_active,
        diagnostic_state=diagnostic_state,
        operator_request_present=False,
    )

    assert decision.terminal_outcome == "request_human_input"
    assert decision.phase6_call_permitted is False
    assert decision.reason_codes == ("MISSING_OPERATOR_OPTIMISATION_REQUEST",)


@pytest.mark.parametrize(
    ("fault_active", "diagnostic_state"),
    [
        (False, "none"),
        (True, "flow_delivery"),
        (True, "power_coupling"),
        (True, "pressure_path_anomaly"),
    ],
)
def test_operator_request_permits_one_phase6_evaluation_for_active_or_inactive_fault(
    fault_active,
    diagnostic_state,
):
    decision = evaluate_pre_optimisation(
        decision_mode="operator_initiated_optimisation",
        fault_active=fault_active,
        diagnostic_state=diagnostic_state,
        operator_request_present=True,
    )

    assert decision.terminal_outcome is None
    assert decision.phase6_call_permitted is True
    assert decision.recommendation_permitted is False
    assert decision.human_approval_state is None
    assert decision.reason_codes == (
        "EXPLICIT_OPERATOR_REQUEST_READY_FOR_TRUSTED_PHASE6_EVALUATION",
    )


def test_unknown_decision_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Phase 7 decision mode"):
        evaluate_pre_optimisation(
            decision_mode="unknown",
            fault_active=False,
            diagnostic_state="none",
            operator_request_present=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fault_active", 1),
        ("diagnostic_state", 1),
        ("operator_request_present", 1),
    ],
)
def test_pre_optimisation_requires_exact_input_types(field, value):
    kwargs = {
        "decision_mode": "monitoring_driven",
        "fault_active": False,
        "diagnostic_state": "none",
        "operator_request_present": False,
    }
    kwargs[field] = value

    with pytest.raises(TypeError):
        evaluate_pre_optimisation(**kwargs)


def test_selected_method_accepted_recommends_pending_human_approval():
    decision = evaluate_phase6_status("selected_method_accepted")

    assert decision.terminal_outcome == "recommend_change_pending_human_approval"
    assert decision.recommendation_permitted is True
    assert decision.human_approval_state == "pending"
    assert decision.warning_codes == ()


def test_selected_method_grid_infeasible_warning_is_exact():
    decision = evaluate_phase6_status("selected_method_accepted_grid_infeasible")

    assert decision.terminal_outcome == "recommend_change_pending_human_approval"
    assert decision.warning_codes == ("MANDATORY_GRID_REFERENCE_INFEASIBLE",)


def test_grid_fallback_infeasible_warning_set_is_exact():
    decision = evaluate_phase6_status("grid_fallback_selected_method_infeasible")

    assert decision.terminal_outcome == "recommend_change_pending_human_approval"
    assert decision.warning_codes == (
        "GRID_FALLBACK_USED",
        "SELECTED_METHOD_INFEASIBLE",
    )


def test_grid_fallback_regression_warning_set_is_exact():
    decision = evaluate_phase6_status(
        "grid_fallback_selected_method_objective_regression"
    )

    assert decision.terminal_outcome == "recommend_change_pending_human_approval"
    assert decision.warning_codes == (
        "GRID_FALLBACK_USED",
        "SELECTED_METHOD_OBJECTIVE_REGRESSION",
    )


def test_no_feasible_status_escalates_without_recommendation():
    decision = evaluate_phase6_status(
        "no_feasible_point_found_under_search_protocol"
    )

    assert decision.terminal_outcome == "escalate_without_recommendation"
    assert decision.recommendation_permitted is False
    assert decision.human_approval_state == "not_applicable"
    assert decision.warning_codes == (
        "NO_FEASIBLE_POINT_FOUND_UNDER_SEARCH_PROTOCOL",
    )


def test_unknown_phase6_status_fails_closed_without_recommendation():
    decision = evaluate_phase6_status("future_unknown_status")

    assert decision.terminal_outcome == "escalate_without_recommendation"
    assert decision.recommendation_permitted is False
    assert decision.warning_codes == ("UNRECOGNISED_PHASE6_STATUS",)


def test_phase6_status_requires_exact_string():
    with pytest.raises(TypeError, match="exact string"):
        evaluate_phase6_status(1)


@pytest.mark.parametrize("boundary", ["phase5_monitoring", "phase6_optimisation"])
def test_boundary_failure_escalates_and_preserves_metadata(boundary):
    result = boundary_failure_decision(
        boundary=boundary,
        stage="runtime",
        cause_type="ValueError",
    )

    assert isinstance(result, BoundaryFailureDecision)
    assert result.decision.terminal_outcome == "escalate_without_recommendation"
    assert result.decision.recommendation_permitted is False
    assert result.decision.reason_codes == ("TRUSTED_BOUNDARY_FAILURE",)
    assert result.boundary == boundary
    assert result.stage == "runtime"
    assert result.cause_type == "ValueError"


def test_boundary_failure_result_is_immutable():
    result = boundary_failure_decision(
        boundary="phase5_monitoring",
        stage="predict",
        cause_type="RuntimeError",
    )

    with pytest.raises(FrozenInstanceError):
        result.stage = "changed"


def test_unknown_boundary_is_rejected():
    with pytest.raises(ValueError, match="Unsupported trusted boundary"):
        boundary_failure_decision(
            boundary="unknown",
            stage="runtime",
            cause_type="ValueError",
        )


@pytest.mark.parametrize(
    ("stage", "cause_type"),
    [("", "ValueError"), ("runtime", "")],
)
def test_empty_boundary_metadata_is_rejected(stage, cause_type):
    with pytest.raises(ValueError):
        boundary_failure_decision(
            boundary="phase6_optimisation",
            stage=stage,
            cause_type=cause_type,
        )


def test_policy_decision_rejects_invalid_nonterminal_state():
    with pytest.raises(ValueError, match="must permit Phase 6"):
        PolicyDecision(
            terminal_outcome=None,
            phase6_call_permitted=False,
            recommendation_permitted=False,
            human_approval_state=None,
            reason_codes=("X",),
        )


def test_policy_decision_rejects_nonrecommendation_pending_approval():
    with pytest.raises(ValueError, match="not_applicable"):
        PolicyDecision(
            terminal_outcome="no_action",
            phase6_call_permitted=False,
            recommendation_permitted=False,
            human_approval_state="pending",
            reason_codes=("X",),
        )
