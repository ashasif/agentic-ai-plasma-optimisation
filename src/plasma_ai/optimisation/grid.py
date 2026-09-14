"""Deterministic Phase 6 grid-baseline machinery.

This module implements the frozen grid construction, scenario evaluation,
lexicographic selection and Pareto extraction required by Phase 6C.

It deliberately does not load the surrogate, execute source physics, invoke
SciPy optimisers, or write scientific result artifacts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plasma_ai.optimisation.primitives import (
    CandidateEvaluation,
    evaluate_candidate,
    lexicographic_objective_key,
)
from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    file_sha256,
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
    OptimisationScenario,
    parse_optimisation_scenario,
)


DEFAULT_GRID_SCENARIO_PATH = Path(
    "configs/phase6/grid_qualification_scenarios.json"
)

EXPECTED_GRID_SCENARIO_SHA256 = (
    "6af65defab31691ffbaef1fc91c7a685267d6169af45f277c097488252d0fc1a"
)

EXPECTED_EFFECTIVE_CONTRACT_SHA256 = (
    "91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7"
)


class GridBaselineError(ValueError):
    """Raised when the frozen deterministic-grid contract is violated."""


@dataclass(frozen=True)
class FrozenGridScenarioSet:
    """Validated frozen Phase 6C qualification-scenario set."""

    payload: dict[str, Any]
    sha256: str
    scenarios: tuple[OptimisationScenario, ...]


@dataclass(frozen=True)
class DeterministicGrid:
    """Immutable deterministic candidate grid."""

    input_matrix: NDArray[np.float64]
    candidate_indices: NDArray[np.int64]
    power_values_W: NDArray[np.float64]
    pressure_values_mTorr: NDArray[np.float64]


@dataclass(frozen=True)
class GridScenarioEvaluation:
    """Deterministic evaluation summary for one frozen scenario."""

    scenario_id: str
    total_candidate_count: int
    feasible_candidate_count: int
    infeasible_candidate_count: int
    selected_candidate_index: int | None
    selected_evaluation: CandidateEvaluation | None
    selected_objective_key: tuple[float, ...] | None
    pareto_candidate_indices: tuple[int, ...]
    constraint_failure_counts: tuple[tuple[str, int], ...]


def _immutable_float64(
    values: ArrayLike,
) -> NDArray[np.float64]:
    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    result.setflags(write=False)

    return result


def _immutable_int64(
    values: ArrayLike,
) -> NDArray[np.int64]:
    result = np.array(
        values,
        dtype=np.int64,
        copy=True,
    )

    result.setflags(write=False)

    return result


def load_frozen_grid_scenario_set(
    path: str | Path = DEFAULT_GRID_SCENARIO_PATH,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> FrozenGridScenarioSet:
    """Load the exact frozen Phase 6C qualification-scenario set."""

    source = Path(path)

    try:
        observed_hash = file_sha256(source)
    except OSError as exc:
        raise GridBaselineError(
            "Frozen Phase 6C scenario file could not be read."
        ) from exc

    if observed_hash != EXPECTED_GRID_SCENARIO_SHA256:
        raise GridBaselineError(
            "Frozen Phase 6C scenario-set SHA-256 mismatch."
        )

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GridBaselineError(
            "Frozen Phase 6C scenario JSON could not be loaded."
        ) from exc

    if not isinstance(payload, dict):
        raise GridBaselineError(
            "Frozen Phase 6C scenario payload must be an object."
        )

    if payload.get("phase") != "6C1":
        raise GridBaselineError(
            "Unexpected frozen grid-scenario phase."
        )

    if (
        payload.get("artifact")
        != "grid_qualification_scenario_set"
    ):
        raise GridBaselineError(
            "Unexpected frozen grid-scenario artifact type."
        )

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    if (
        protocol.effective_sha256
        != EXPECTED_EFFECTIVE_CONTRACT_SHA256
    ):
        raise GridBaselineError(
            "Effective Phase 6 contract identity mismatch."
        )

    contract = payload.get("effective_contract")

    if not isinstance(contract, dict):
        raise GridBaselineError(
            "Frozen grid scenario effective-contract metadata is invalid."
        )

    if (
        contract.get("effective_sha256")
        != protocol.effective_sha256
    ):
        raise GridBaselineError(
            "Frozen grid scenarios reference the wrong effective contract."
        )

    raw_scenarios = payload.get("scenarios")

    if not isinstance(raw_scenarios, list):
        raise GridBaselineError(
            "Frozen grid scenarios must be a list."
        )

    if payload.get("scenario_count") != len(raw_scenarios):
        raise GridBaselineError(
            "Frozen grid scenario count is inconsistent."
        )

    scenarios = tuple(
        parse_optimisation_scenario(
            raw,
            protocol=protocol,
        )
        for raw in raw_scenarios
    )

    identifiers = tuple(
        scenario.scenario_id
        for scenario in scenarios
    )

    if len(set(identifiers)) != len(identifiers):
        raise GridBaselineError(
            "Frozen grid scenario identifiers are not unique."
        )

    return FrozenGridScenarioSet(
        payload=payload,
        sha256=observed_hash,
        scenarios=scenarios,
    )


def build_deterministic_grid(
    scenario_set: FrozenGridScenarioSet,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> DeterministicGrid:
    """Construct the exact frozen 151 x 101 candidate grid."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    grid = scenario_set.payload["grid_contract"]

    power_points = int(grid["power_points"])
    pressure_points = int(grid["pressure_points"])
    total_points = int(grid["total_points"])

    power_lower = float(grid["power_lower_W"])
    power_upper = float(grid["power_upper_W"])
    power_increment = float(grid["power_increment_W"])

    pressure_lower = float(grid["pressure_lower_mTorr"])
    pressure_upper = float(grid["pressure_upper_mTorr"])
    pressure_increment = float(
        grid["pressure_increment_mTorr"]
    )

    baseline = protocol.base[
        "deterministic_grid_baseline"
    ]

    if power_points != int(baseline["power_points"]):
        raise GridBaselineError(
            "Frozen grid power-point count disagrees with Phase 6A."
        )

    if pressure_points != int(
        baseline["pressure_points"]
    ):
        raise GridBaselineError(
            "Frozen grid pressure-point count disagrees with Phase 6A."
        )

    if total_points != int(
        baseline["total_points"]
    ):
        raise GridBaselineError(
            "Frozen grid total-point count disagrees with Phase 6A."
        )

    if power_increment != float(
        baseline["power_increment_W"]
    ):
        raise GridBaselineError(
            "Frozen grid power increment disagrees with Phase 6A."
        )

    if pressure_increment != float(
        baseline["pressure_increment_mTorr"]
    ):
        raise GridBaselineError(
            "Frozen grid pressure increment disagrees with Phase 6A."
        )

    domain = protocol.base[
        "frozen_phase4_contract"
    ]["qualified_domain"]

    expected_power_lower = float(
        domain["nominal_absorbed_power_W"]["lower"]
    )

    expected_power_upper = float(
        domain["nominal_absorbed_power_W"]["upper"]
    )

    expected_pressure_lower = float(
        domain["target_pressure_mTorr"]["lower"]
    )

    expected_pressure_upper = float(
        domain["target_pressure_mTorr"]["upper"]
    )

    if (
        power_lower != expected_power_lower
        or power_upper != expected_power_upper
        or pressure_lower != expected_pressure_lower
        or pressure_upper != expected_pressure_upper
    ):
        raise GridBaselineError(
            "Frozen grid bounds disagree with the qualified Phase 4 domain."
        )

    if total_points != power_points * pressure_points:
        raise GridBaselineError(
            "Frozen grid total-point count is internally inconsistent."
        )

    power_values = (
        power_lower
        + np.arange(
            power_points,
            dtype=np.float64,
        )
        * power_increment
    )

    pressure_values = (
        pressure_lower
        + np.arange(
            pressure_points,
            dtype=np.float64,
        )
        * pressure_increment
    )

    if power_values[-1] != power_upper:
        raise GridBaselineError(
            "Frozen grid does not terminate at the power upper bound."
        )

    if pressure_values[-1] != pressure_upper:
        raise GridBaselineError(
            "Frozen grid does not terminate at the pressure upper bound."
        )

    powers = np.repeat(
        power_values,
        pressure_points,
    )

    pressures = np.tile(
        pressure_values,
        power_points,
    )

    input_matrix = np.column_stack(
        (
            powers,
            pressures,
        )
    )

    candidate_indices = np.arange(
        total_points,
        dtype=np.int64,
    )

    if input_matrix.shape != (
        total_points,
        2,
    ):
        raise GridBaselineError(
            "Constructed deterministic grid has the wrong shape."
        )

    return DeterministicGrid(
        input_matrix=_immutable_float64(
            input_matrix
        ),
        candidate_indices=_immutable_int64(
            candidate_indices
        ),
        power_values_W=_immutable_float64(
            power_values
        ),
        pressure_values_mTorr=_immutable_float64(
            pressure_values
        ),
    )


def _prediction_vector(
    values: ArrayLike,
    *,
    expected_rows: int,
    field: str,
) -> NDArray[np.float64]:
    """Materialise one prediction vector for grid evaluation."""

    try:
        vector = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise GridBaselineError(
            f"{field} could not be converted to float64."
        ) from exc

    if vector.shape != (expected_rows,):
        raise GridBaselineError(
            f"{field} has shape {vector.shape}; "
            f"expected {(expected_rows,)}."
        )

    return vector


def pareto_candidate_indices(
    candidate_indices: ArrayLike,
    objective_matrix: ArrayLike,
) -> tuple[int, ...]:
    """Return exact non-dominated candidate indices in candidate order.

    All objective columns are assumed to already be in minimisation form.
    No numerical dominance tolerance is applied.
    """

    indices = np.asarray(
        candidate_indices,
        dtype=np.int64,
    )

    objectives = np.asarray(
        objective_matrix,
        dtype=np.float64,
    )

    if indices.ndim != 1:
        raise GridBaselineError(
            "Pareto candidate indices must be one-dimensional."
        )

    if objectives.ndim != 2:
        raise GridBaselineError(
            "Pareto objective matrix must be two-dimensional."
        )

    if objectives.shape[0] != indices.shape[0]:
        raise GridBaselineError(
            "Pareto candidate/objective row counts differ."
        )

    if objectives.shape[1] < 1:
        raise GridBaselineError(
            "Pareto objective matrix must contain at least one column."
        )

    if not np.all(np.isfinite(objectives)):
        raise GridBaselineError(
            "Pareto objective matrix contains non-finite values."
        )

    if indices.size == 0:
        return ()

    if objectives.shape[1] == 1:
        best = float(np.min(objectives[:, 0]))

        retained = indices[
            objectives[:, 0] == best
        ]

        return tuple(
            int(value)
            for value in retained
        )

    if objectives.shape[1] == 2:
        order = np.lexsort(
            (
                indices,
                objectives[:, 1],
                objectives[:, 0],
            )
        )

        retained_positions = np.zeros(
            indices.shape[0],
            dtype=bool,
        )

        best_previous_second = np.inf
        cursor = 0

        while cursor < order.size:
            first_position = int(order[cursor])
            first_value = objectives[
                first_position,
                0,
            ]

            end = cursor + 1

            while (
                end < order.size
                and objectives[
                    int(order[end]),
                    0,
                ]
                == first_value
            ):
                end += 1

            group_positions = order[
                cursor:end
            ]

            group_second = objectives[
                group_positions,
                1,
            ]

            group_best_second = float(
                np.min(group_second)
            )

            if group_best_second < best_previous_second:
                group_retained = group_positions[
                    group_second == group_best_second
                ]

                retained_positions[
                    group_retained
                ] = True

            best_previous_second = min(
                best_previous_second,
                group_best_second,
            )

            cursor = end

        retained_indices = indices[
            retained_positions
        ]

        return tuple(
            int(value)
            for value in retained_indices
        )

    # Generic exact fallback for >2 objectives.
    # Current frozen Phase 6C1 scenarios contain at most two objectives,
    # but keeping an exact fallback avoids silently changing semantics.
    frontier_positions: list[int] = []

    for position in range(indices.shape[0]):
        candidate = objectives[position]

        dominated = False

        for frontier_position in frontier_positions:
            incumbent = objectives[
                frontier_position
            ]

            if (
                np.all(incumbent <= candidate)
                and np.any(incumbent < candidate)
            ):
                dominated = True
                break

        if dominated:
            continue

        survivors: list[int] = []

        for frontier_position in frontier_positions:
            incumbent = objectives[
                frontier_position
            ]

            candidate_dominates = (
                np.all(candidate <= incumbent)
                and np.any(candidate < incumbent)
            )

            if not candidate_dominates:
                survivors.append(
                    frontier_position
                )

        survivors.append(position)
        frontier_positions = survivors

    frontier_set = set(
        frontier_positions
    )

    return tuple(
        int(indices[position])
        for position in range(indices.shape[0])
        if position in frontier_set
    )


def evaluate_grid_scenario(
    grid: DeterministicGrid,
    *,
    electron_density_m3: ArrayLike,
    electron_temperature_eV: ArrayLike,
    scenario: OptimisationScenario,
    protocol: EffectivePhase6Protocol | None = None,
) -> GridScenarioEvaluation:
    """Evaluate one scenario over an already-predicted candidate grid."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    rows = grid.input_matrix.shape[0]

    if grid.input_matrix.ndim != 2:
        raise GridBaselineError(
            "Grid input matrix must be two-dimensional."
        )

    if grid.input_matrix.shape[1] != 2:
        raise GridBaselineError(
            "Grid input matrix must contain exactly two columns."
        )

    if grid.candidate_indices.shape != (rows,):
        raise GridBaselineError(
            "Grid candidate-index vector has the wrong shape."
        )

    expected_indices = np.arange(
        rows,
        dtype=np.int64,
    )

    if not np.array_equal(
        grid.candidate_indices,
        expected_indices,
    ):
        raise GridBaselineError(
            "Grid candidate indices are not contiguous frozen row indices."
        )

    density = _prediction_vector(
        electron_density_m3,
        expected_rows=rows,
        field="electron_density_m3",
    )

    temperature = _prediction_vector(
        electron_temperature_eV,
        expected_rows=rows,
        field="electron_temperature_eV",
    )

    evaluations: list[CandidateEvaluation] = []
    feasible_positions: list[int] = []
    failure_counts: Counter[str] = Counter()

    for row_position in range(rows):
        point = OperatingPoint(
            nominal_absorbed_power_W=float(
                grid.input_matrix[
                    row_position,
                    0,
                ]
            ),
            target_pressure_mTorr=float(
                grid.input_matrix[
                    row_position,
                    1,
                ]
            ),
        )

        evaluation = evaluate_candidate(
            point,
            electron_density_m3=float(
                density[row_position]
            ),
            electron_temperature_eV=float(
                temperature[row_position]
            ),
            scenario=scenario,
            protocol=protocol,
        )

        evaluations.append(
            evaluation
        )

        for failure in evaluation.constraint_failures:
            failure_counts[failure] += 1

        if evaluation.scenario_feasible:
            feasible_positions.append(
                row_position
            )

    if not feasible_positions:
        return GridScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            total_candidate_count=rows,
            feasible_candidate_count=0,
            infeasible_candidate_count=rows,
            selected_candidate_index=None,
            selected_evaluation=None,
            selected_objective_key=None,
            pareto_candidate_indices=(),
            constraint_failure_counts=tuple(
                sorted(
                    failure_counts.items()
                )
            ),
        )

    selected_position = min(
        feasible_positions,
        key=lambda position: (
            lexicographic_objective_key(
                evaluations[position],
                scenario,
            ),
            int(
                grid.candidate_indices[
                    position
                ]
            ),
        ),
    )

    selected_evaluation = evaluations[
        selected_position
    ]

    selected_key = lexicographic_objective_key(
        selected_evaluation,
        scenario,
    )

    pareto_indices: tuple[int, ...] = ()

    if len(scenario.objective_priority) > 1:
        feasible_candidate_indices = np.asarray(
            [
                int(
                    grid.candidate_indices[
                        position
                    ]
                )
                for position in feasible_positions
            ],
            dtype=np.int64,
        )

        objective_matrix = np.asarray(
            [
                lexicographic_objective_key(
                    evaluations[position],
                    scenario,
                )
                for position in feasible_positions
            ],
            dtype=np.float64,
        )

        pareto_indices = pareto_candidate_indices(
            feasible_candidate_indices,
            objective_matrix,
        )

    feasible_count = len(
        feasible_positions
    )

    return GridScenarioEvaluation(
        scenario_id=scenario.scenario_id,
        total_candidate_count=rows,
        feasible_candidate_count=feasible_count,
        infeasible_candidate_count=(
            rows - feasible_count
        ),
        selected_candidate_index=int(
            grid.candidate_indices[
                selected_position
            ]
        ),
        selected_evaluation=selected_evaluation,
        selected_objective_key=tuple(
            float(value)
            for value in selected_key
        ),
        pareto_candidate_indices=pareto_indices,
        constraint_failure_counts=tuple(
            sorted(
                failure_counts.items()
            )
        ),
    )


def evaluate_grid_scenarios(
    grid: DeterministicGrid,
    *,
    electron_density_m3: ArrayLike,
    electron_temperature_eV: ArrayLike,
    scenarios: tuple[OptimisationScenario, ...],
    protocol: EffectivePhase6Protocol | None = None,
) -> tuple[GridScenarioEvaluation, ...]:
    """Evaluate multiple frozen scenarios using one prediction batch."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    return tuple(
        evaluate_grid_scenario(
            grid,
            electron_density_m3=electron_density_m3,
            electron_temperature_eV=electron_temperature_eV,
            scenario=scenario,
            protocol=protocol,
        )
        for scenario in scenarios
    )
