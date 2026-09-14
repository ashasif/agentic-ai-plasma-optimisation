"""Controlled optimisation primitives for Phase 6."""

from plasma_ai.optimisation.primitives import (
    CandidateEvaluation,
    DecisionPointError,
    boundary_margin_cost,
    density_relative_error,
    evaluate_candidate,
    lexicographic_objective_key,
    normalized_absorbed_power,
    normalized_boundary_margin,
    normalized_nominal_distance,
    validate_decision_point,
)
from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    ProtocolIntegrityError,
    effective_contract_sha256,
    file_sha256,
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
    OptimisationScenario,
    ScenarioValidationError,
    parse_optimisation_scenario,
)


__all__ = [
    "CandidateEvaluation",
    "DecisionPointError",
    "EffectivePhase6Protocol",
    "OperatingPoint",
    "OptimisationScenario",
    "ProtocolIntegrityError",
    "ScenarioValidationError",
    "boundary_margin_cost",
    "density_relative_error",
    "effective_contract_sha256",
    "evaluate_candidate",
    "file_sha256",
    "lexicographic_objective_key",
    "load_effective_phase6_protocol",
    "normalized_absorbed_power",
    "normalized_boundary_margin",
    "normalized_nominal_distance",
    "parse_optimisation_scenario",
    "validate_decision_point",
]
