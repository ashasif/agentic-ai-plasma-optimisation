"""Pure deterministic decision policy for frozen Phase 7C orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DECISION_MODES: Final = frozenset(
    {
        "monitoring_driven",
        "operator_initiated_optimisation",
    }
)

DECISION_OUTCOMES: Final = frozenset(
    {
        "no_action",
        "request_human_input",
        "recommend_change_pending_human_approval",
        "escalate_without_recommendation",
    }
)

DIAGNOSTIC_STATES: Final = frozenset(
    {
        "none",
        "flow_delivery",
        "power_coupling",
        "pressure_path_anomaly",
    }
)

PHASE6_STATUSES: Final = frozenset(
    {
        "selected_method_accepted",
        "selected_method_accepted_grid_infeasible",
        "grid_fallback_selected_method_infeasible",
        "grid_fallback_selected_method_objective_regression",
        "no_feasible_point_found_under_search_protocol",
    }
)

BOUNDARIES: Final = frozenset(
    {
        "phase5_monitoring",
        "phase6_optimisation",
    }
)


@dataclass(frozen=True)
class PolicyDecision:
    """Immutable deterministic policy result.

    terminal_outcome is None only for the internal directive that permits
    exactly one trusted Phase 6 evaluation before a final outcome exists.
    """

    terminal_outcome: str | None
    phase6_call_permitted: bool
    recommendation_permitted: bool
    human_approval_state: str | None
    reason_codes: tuple[str, ...]
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.terminal_outcome is None:
            if not self.phase6_call_permitted:
                raise ValueError("Non-terminal policy directive must permit Phase 6.")
            if self.recommendation_permitted:
                raise ValueError("Pre-Phase6 directive cannot recommend a change.")
            if self.human_approval_state is not None:
                raise ValueError("Pre-Phase6 directive cannot have an approval state.")
            return

        if self.terminal_outcome not in DECISION_OUTCOMES:
            raise ValueError(f"Unsupported terminal outcome: {self.terminal_outcome}.")

        if self.phase6_call_permitted:
            raise ValueError("Terminal policy decision cannot permit another Phase 6 call.")

        if self.terminal_outcome == "recommend_change_pending_human_approval":
            if not self.recommendation_permitted:
                raise ValueError("Recommendation outcome must permit a recommendation.")
            if self.human_approval_state != "pending":
                raise ValueError("Recommendation outcome must start pending approval.")
        else:
            if self.recommendation_permitted:
                raise ValueError("Non-recommendation outcome cannot permit recommendation.")
            if self.human_approval_state != "not_applicable":
                raise ValueError("Non-recommendation outcome must use not_applicable approval.")


@dataclass(frozen=True)
class BoundaryFailureDecision:
    """Fail-closed policy decision preserving trusted-boundary metadata."""

    decision: PolicyDecision
    boundary: str
    stage: str
    cause_type: str


def _terminal(
    outcome: str,
    reason_code: str,
    *,
    warning_codes: tuple[str, ...] = (),
) -> PolicyDecision:
    recommendation = outcome == "recommend_change_pending_human_approval"

    return PolicyDecision(
        terminal_outcome=outcome,
        phase6_call_permitted=False,
        recommendation_permitted=recommendation,
        human_approval_state="pending" if recommendation else "not_applicable",
        reason_codes=(reason_code,),
        warning_codes=warning_codes,
    )


def _phase6_directive(reason_code: str) -> PolicyDecision:
    return PolicyDecision(
        terminal_outcome=None,
        phase6_call_permitted=True,
        recommendation_permitted=False,
        human_approval_state=None,
        reason_codes=(reason_code,),
        warning_codes=(),
    )


def _monitoring_state_is_coherent(*, fault_active: bool, diagnostic_state: str) -> bool:
    if diagnostic_state not in DIAGNOSTIC_STATES:
        return False

    if fault_active:
        return diagnostic_state != "none"

    return diagnostic_state == "none"


def evaluate_pre_optimisation(
    *,
    decision_mode: str,
    fault_active: bool,
    diagnostic_state: str,
    operator_request_present: bool,
) -> PolicyDecision:
    """Apply the frozen pre-optimisation Phase 7C decision policy."""

    if decision_mode not in DECISION_MODES:
        raise ValueError(f"Unsupported Phase 7 decision mode: {decision_mode}.")

    if type(fault_active) is not bool:
        raise TypeError("fault_active must be an exact bool.")

    if type(diagnostic_state) is not str:
        raise TypeError("diagnostic_state must be an exact string.")

    if type(operator_request_present) is not bool:
        raise TypeError("operator_request_present must be an exact bool.")

    if not _monitoring_state_is_coherent(
        fault_active=fault_active,
        diagnostic_state=diagnostic_state,
    ):
        return _terminal(
            "escalate_without_recommendation",
            "MONITORING_EVIDENCE_INCOHERENT",
        )

    if decision_mode == "monitoring_driven":
        if operator_request_present:
            return _terminal(
                "escalate_without_recommendation",
                "DECISION_MODE_REQUEST_SOURCE_INCOHERENT",
            )

        if not fault_active:
            return _terminal(
                "no_action",
                "MONITORING_FAULT_INACTIVE",
            )

        return _terminal(
            "request_human_input",
            "ACTIVE_FAULT_WITHOUT_FROZEN_OPTIMISATION_TEMPLATE",
        )

    if not operator_request_present:
        return _terminal(
            "request_human_input",
            "MISSING_OPERATOR_OPTIMISATION_REQUEST",
        )

    return _phase6_directive(
        "EXPLICIT_OPERATOR_REQUEST_READY_FOR_TRUSTED_PHASE6_EVALUATION"
    )


def evaluate_phase6_status(status: str) -> PolicyDecision:
    """Map the frozen Phase 6 runtime status to the Phase 7C final policy."""

    if type(status) is not str:
        raise TypeError("Phase 6 status must be an exact string.")

    if status == "selected_method_accepted":
        return _terminal(
            "recommend_change_pending_human_approval",
            "PHASE6_SELECTED_METHOD_ACCEPTED",
        )

    if status == "selected_method_accepted_grid_infeasible":
        return _terminal(
            "recommend_change_pending_human_approval",
            "PHASE6_SELECTED_METHOD_ACCEPTED_GRID_REFERENCE_INFEASIBLE",
            warning_codes=("MANDATORY_GRID_REFERENCE_INFEASIBLE",),
        )

    if status == "grid_fallback_selected_method_infeasible":
        return _terminal(
            "recommend_change_pending_human_approval",
            "PHASE6_GRID_FALLBACK_SELECTED_METHOD_INFEASIBLE",
            warning_codes=(
                "GRID_FALLBACK_USED",
                "SELECTED_METHOD_INFEASIBLE",
            ),
        )

    if status == "grid_fallback_selected_method_objective_regression":
        return _terminal(
            "recommend_change_pending_human_approval",
            "PHASE6_GRID_FALLBACK_OBJECTIVE_REGRESSION",
            warning_codes=(
                "GRID_FALLBACK_USED",
                "SELECTED_METHOD_OBJECTIVE_REGRESSION",
            ),
        )

    if status == "no_feasible_point_found_under_search_protocol":
        return _terminal(
            "escalate_without_recommendation",
            "PHASE6_NO_FEASIBLE_POINT_UNDER_SEARCH_PROTOCOL",
            warning_codes=("NO_FEASIBLE_POINT_FOUND_UNDER_SEARCH_PROTOCOL",),
        )

    return _terminal(
        "escalate_without_recommendation",
        "UNRECOGNISED_PHASE6_STATUS",
        warning_codes=("UNRECOGNISED_PHASE6_STATUS",),
    )


def boundary_failure_decision(
    *,
    boundary: str,
    stage: str,
    cause_type: str,
) -> BoundaryFailureDecision:
    """Translate a trusted-boundary failure into deterministic fail-closed policy."""

    if boundary not in BOUNDARIES:
        raise ValueError(f"Unsupported trusted boundary: {boundary}.")

    if type(stage) is not str or not stage:
        raise ValueError("Boundary failure stage must be a non-empty exact string.")

    if type(cause_type) is not str or not cause_type:
        raise ValueError("Boundary failure cause_type must be a non-empty exact string.")

    return BoundaryFailureDecision(
        decision=_terminal(
            "escalate_without_recommendation",
            "TRUSTED_BOUNDARY_FAILURE",
        ),
        boundary=boundary,
        stage=stage,
        cause_type=cause_type,
    )
