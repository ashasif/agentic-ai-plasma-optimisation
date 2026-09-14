"""Controlled optimisation primitives for Phase 6."""

from plasma_ai.optimisation.grid import (
    DeterministicGrid,
    FrozenGridScenarioSet,
    GridBaselineError,
    GridScenarioEvaluation,
    build_deterministic_grid,
    evaluate_grid_scenario,
    evaluate_grid_scenarios,
    load_frozen_grid_scenario_set,
    pareto_candidate_indices,
)
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
from plasma_ai.optimisation.surrogate_adapter import (
    Phase6Prediction,
    Phase6PredictionBatch,
    Phase6SurrogateAdapter,
    PredictionContractError,
    SurrogateAdapterError,
    load_phase6_surrogate_adapter,
)


__all__ = [
    "CandidateEvaluation",
    "DecisionPointError",
    "DeterministicGrid",
    "EffectivePhase6Protocol",
    "FrozenGridScenarioSet",
    "GridBaselineError",
    "GridScenarioEvaluation",
    "OperatingPoint",
    "OptimisationScenario",
    "Phase6Prediction",
    "Phase6PredictionBatch",
    "Phase6SurrogateAdapter",
    "PredictionContractError",
    "ProtocolIntegrityError",
    "ScenarioValidationError",
    "SurrogateAdapterError",
    "boundary_margin_cost",
    "build_deterministic_grid",
    "density_relative_error",
    "effective_contract_sha256",
    "evaluate_candidate",
    "evaluate_grid_scenario",
    "evaluate_grid_scenarios",
    "file_sha256",
    "lexicographic_objective_key",
    "load_effective_phase6_protocol",
    "load_frozen_grid_scenario_set",
    "load_phase6_surrogate_adapter",
    "normalized_absorbed_power",
    "normalized_boundary_margin",
    "normalized_nominal_distance",
    "pareto_candidate_indices",
    "parse_optimisation_scenario",
    "validate_decision_point",
]
