from __future__ import annotations

import json

import numpy as np
import pytest

from plasma_ai.optimisation.grid import (
    DeterministicGrid,
    EXPECTED_GRID_SCENARIO_SHA256,
    GridBaselineError,
    build_deterministic_grid,
    evaluate_grid_scenario,
    load_frozen_grid_scenario_set,
    pareto_candidate_indices,
)
from plasma_ai.optimisation.scenario import (
    parse_optimisation_scenario,
)


def _scenario(**updates):
    payload = {
        "scenario_id": "grid-test",
        "density_target_mode": "disabled",
        "target_electron_density_m3": None,
        "density_relative_tolerance": None,
        "minimum_electron_temperature_eV": None,
        "maximum_electron_temperature_eV": None,
        "nominal_operating_point": None,
        "minimum_normalized_boundary_margin": None,
        "objective_priority": [
            "absorbed_power",
        ],
    }

    payload.update(updates)

    return parse_optimisation_scenario(
        payload
    )


def _small_grid(rows):
    matrix = np.asarray(
        rows,
        dtype=np.float64,
    )

    indices = np.arange(
        matrix.shape[0],
        dtype=np.int64,
    )

    powers = np.unique(
        matrix[:, 0]
    )

    pressures = np.unique(
        matrix[:, 1]
    )

    matrix.setflags(write=False)
    indices.setflags(write=False)
    powers.setflags(write=False)
    pressures.setflags(write=False)

    return DeterministicGrid(
        input_matrix=matrix,
        candidate_indices=indices,
        power_values_W=powers,
        pressure_values_mTorr=pressures,
    )


def test_frozen_scenario_set_loads_exact_hash() -> None:
    scenario_set = load_frozen_grid_scenario_set()

    assert (
        scenario_set.sha256
        == EXPECTED_GRID_SCENARIO_SHA256
    )

    assert len(scenario_set.scenarios) == 10


def test_tampered_scenario_copy_is_rejected(
    tmp_path,
) -> None:
    scenario_set = load_frozen_grid_scenario_set()

    payload = dict(
        scenario_set.payload
    )

    payload["phase"] = "tampered"

    path = tmp_path / "scenarios.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        GridBaselineError,
        match="SHA-256 mismatch",
    ):
        load_frozen_grid_scenario_set(
            path
        )


def test_exact_frozen_grid_geometry() -> None:
    scenario_set = load_frozen_grid_scenario_set()

    grid = build_deterministic_grid(
        scenario_set
    )

    assert grid.input_matrix.shape == (
        15251,
        2,
    )

    assert grid.candidate_indices.shape == (
        15251,
    )

    np.testing.assert_array_equal(
        grid.input_matrix[0],
        np.array(
            [15.0, 10.0]
        ),
    )

    np.testing.assert_array_equal(
        grid.input_matrix[100],
        np.array(
            [15.0, 60.0]
        ),
    )

    np.testing.assert_array_equal(
        grid.input_matrix[101],
        np.array(
            [15.5, 10.0]
        ),
    )

    np.testing.assert_array_equal(
        grid.input_matrix[7625],
        np.array(
            [52.5, 35.0]
        ),
    )

    np.testing.assert_array_equal(
        grid.input_matrix[-1],
        np.array(
            [90.0, 60.0]
        ),
    )

    assert grid.candidate_indices[0] == 0
    assert grid.candidate_indices[-1] == 15250


def test_grid_arrays_are_read_only() -> None:
    scenario_set = load_frozen_grid_scenario_set()

    grid = build_deterministic_grid(
        scenario_set
    )

    assert grid.input_matrix.flags.writeable is False
    assert grid.candidate_indices.flags.writeable is False
    assert grid.power_values_W.flags.writeable is False
    assert grid.pressure_values_mTorr.flags.writeable is False

    with pytest.raises(ValueError):
        grid.input_matrix[0, 0] = 20.0


def test_absorbed_power_selection_uses_lowest_index_tie_break() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [15.0, 20.0],
            [20.0, 10.0],
            [20.0, 20.0],
        ]
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.full(
            4,
            1.0e17,
        ),
        electron_temperature_eV=np.full(
            4,
            1.8,
        ),
        scenario=_scenario(),
    )

    assert result.feasible_candidate_count == 4
    assert result.selected_candidate_index == 0
    assert result.selected_evaluation is not None
    assert (
        result.selected_evaluation
        .operating_point
        .nominal_absorbed_power_W
        == 15.0
    )


def test_nominal_distance_selects_exact_nominal_point() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [52.5, 35.0],
            [90.0, 60.0],
        ]
    )

    scenario = _scenario(
        nominal_operating_point={
            "nominal_absorbed_power_W": 52.5,
            "target_pressure_mTorr": 35.0,
        },
        objective_priority=[
            "nominal_distance",
        ],
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.full(
            3,
            1.0e17,
        ),
        electron_temperature_eV=np.full(
            3,
            1.8,
        ),
        scenario=scenario,
    )

    assert result.selected_candidate_index == 1
    assert result.selected_objective_key == (
        pytest.approx(0.0),
    )


def test_hard_density_infeasibility_returns_no_selection() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [52.5, 35.0],
            [90.0, 60.0],
        ]
    )

    scenario = _scenario(
        density_target_mode="hard_constraint",
        target_electron_density_m3=1.0e20,
        density_relative_tolerance=0.01,
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.array(
            [
                1.0e17,
                1.5e17,
                2.0e17,
            ]
        ),
        electron_temperature_eV=np.full(
            3,
            1.8,
        ),
        scenario=scenario,
    )

    assert result.feasible_candidate_count == 0
    assert result.infeasible_candidate_count == 3
    assert result.selected_candidate_index is None
    assert result.selected_evaluation is None
    assert result.selected_objective_key is None
    assert result.pareto_candidate_indices == ()

    failures = dict(
        result.constraint_failure_counts
    )

    assert failures["density_target"] == 3


def test_temperature_constraint_failure_summary() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [20.0, 20.0],
            [25.0, 30.0],
        ]
    )

    scenario = _scenario(
        maximum_electron_temperature_eV=1.8,
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.full(
            3,
            1.0e17,
        ),
        electron_temperature_eV=np.array(
            [
                1.7,
                1.8,
                1.9,
            ]
        ),
        scenario=scenario,
    )

    assert result.feasible_candidate_count == 2

    failures = dict(
        result.constraint_failure_counts
    )

    assert (
        failures[
            "maximum_electron_temperature"
        ]
        == 1
    )


def test_exact_two_objective_pareto_indices() -> None:
    indices = np.array(
        [0, 1, 2, 3, 4],
        dtype=np.int64,
    )

    objectives = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    assert pareto_candidate_indices(
        indices,
        objectives,
    ) == (
        0,
        1,
        2,
        4,
    )


def test_pareto_equal_first_objective_dominance() -> None:
    indices = np.array(
        [10, 11, 12],
        dtype=np.int64,
    )

    objectives = np.array(
        [
            [1.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert pareto_candidate_indices(
        indices,
        objectives,
    ) == (
        11,
        12,
    )


def test_pareto_previous_group_equal_second_dominates() -> None:
    indices = np.array(
        [0, 1],
        dtype=np.int64,
    )

    objectives = np.array(
        [
            [1.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    assert pareto_candidate_indices(
        indices,
        objectives,
    ) == (0,)


def test_multiobjective_grid_returns_pareto_candidates() -> None:
    grid = _small_grid(
        [
            [20.0, 10.0],
            [30.0, 10.0],
            [40.0, 10.0],
            [50.0, 10.0],
        ]
    )

    scenario = _scenario(
        density_target_mode="soft_objective",
        target_electron_density_m3=5.0e16,
        objective_priority=[
            "density_target_error",
            "absorbed_power",
        ],
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.array(
            [
                2.0e16,
                3.0e16,
                4.0e16,
                5.0e16,
            ]
        ),
        electron_temperature_eV=np.full(
            4,
            1.8,
        ),
        scenario=scenario,
    )

    assert result.feasible_candidate_count == 4

    assert result.selected_candidate_index == 3

    assert result.pareto_candidate_indices == (
        0,
        1,
        2,
        3,
    )


def test_invalid_prediction_vector_shape_rejected() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [20.0, 20.0],
        ]
    )

    with pytest.raises(
        GridBaselineError,
        match="shape",
    ):
        evaluate_grid_scenario(
            grid,
            electron_density_m3=np.array(
                [1.0e17]
            ),
            electron_temperature_eV=np.array(
                [1.8, 1.8]
            ),
            scenario=_scenario(),
        )


def test_invalid_prediction_becomes_infeasible_candidate() -> None:
    grid = _small_grid(
        [
            [15.0, 10.0],
            [20.0, 20.0],
        ]
    )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=np.array(
            [
                np.nan,
                1.0e17,
            ]
        ),
        electron_temperature_eV=np.array(
            [
                1.8,
                1.8,
            ]
        ),
        scenario=_scenario(),
    )

    assert result.feasible_candidate_count == 1
    assert result.selected_candidate_index == 1

    failures = dict(
        result.constraint_failure_counts
    )

    assert failures["invalid_prediction"] == 1
