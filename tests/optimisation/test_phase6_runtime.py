from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plasma_ai.optimisation import (
    OperatingPoint,
    ProductionRuntimeError,
    SeededDERunResult,
    build_deterministic_grid,
    build_runtime_from_components,
    load_effective_phase6_protocol,
    load_frozen_grid_scenario_set,
    parse_runtime_request,
)


def _grid():
    scenario_set = (
        load_frozen_grid_scenario_set()
    )

    grid = build_deterministic_grid(
        scenario_set
    )

    return scenario_set, grid


def _synthetic_predictions(grid):
    power = grid.input_matrix[
        :,
        0,
    ]

    pressure = grid.input_matrix[
        :,
        1,
    ]

    density = (
        2.0e15
        * power
    )

    temperature = (
        1.55
        + 0.005
        * pressure
    )

    return density, temperature


def _scenario(
    scenario_set,
    scenario_id: str,
):
    return next(
        item
        for item
        in scenario_set.scenarios
        if item.scenario_id
        == scenario_id
    )


def _selected_result(
    scenario,
    *,
    run_pass: bool,
    feasible: bool,
    objective: float | None,
    power: float | None = 52.5,
    pressure: float | None = 35.0,
):
    return SeededDERunResult(
        scenario_id=scenario.scenario_id,
        seed=20260914,
        decision_coordinates=(
            None
            if power is None
            else (
                float(power),
                float(pressure),
            )
        ),
        objective_name=(
            scenario.objective_priority[0]
        ),
        objective_value=objective,
        electron_density_m3=(
            None
            if power is None
            else float(
                2.0e15
                * power
            )
        ),
        electron_temperature_eV=(
            None
            if pressure is None
            else float(
                1.55
                + 0.005
                * pressure
            )
        ),
        final_phase6_feasible=feasible,
        run_pass=run_pass,
        scipy_success=run_pass,
        scipy_message="synthetic",
        scipy_nfev=1,
        scipy_nit=1,
        constraint_values=(),
        prediction_cache_hits=0,
        prediction_cache_misses=1,
    )


def _runtime_for_result(
    scenario_set,
    grid,
    density,
    temperature,
    result,
):
    def runner(_scenario):
        assert (
            _scenario.scenario_id
            == result.scenario_id
        )
        return result

    return build_runtime_from_components(
        grid=grid,
        grid_density_m3=density,
        grid_temperature_eV=temperature,
        selected_runner=runner,
        provenance=(
            ("synthetic", "true"),
        ),
    )


def test_runtime_rejects_multiobjective_request() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "soft_density_1e17_low_power",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=0.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    with pytest.raises(
        ProductionRuntimeError,
        match="multiobjective",
    ):
        runtime.run(
            scenario
        )


def test_status_selected_method_accepted() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "soft_density_5e16",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=0.0,
        power=25.0,
        pressure=35.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    response = runtime.run(
        scenario
    )

    assert response.status == (
        "selected_method_accepted"
    )

    assert response.chosen_source == (
        "differential_evolution"
    )


def test_status_grid_fallback_selected_method_infeasible() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "temperature_window_min_power",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=False,
        feasible=False,
        objective=None,
        power=None,
        pressure=None,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    response = runtime.run(
        scenario
    )

    assert response.status == (
        "grid_fallback_selected_method_infeasible"
    )

    assert response.chosen_source == (
        "deterministic_grid"
    )


def test_status_grid_fallback_objective_regression() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "temperature_window_min_power",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=0.9,
        power=82.5,
        pressure=35.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    response = runtime.run(
        scenario
    )

    assert response.status == (
        "grid_fallback_selected_method_objective_regression"
    )

    assert response.chosen_source == (
        "deterministic_grid"
    )

    assert (
        response.selected_minus_grid_primary_objective
        is not None
    )

    assert (
        response.selected_minus_grid_primary_objective
        > 1.0e-6
    )


def test_status_no_feasible_point_found() -> None:
    scenario_set, grid = _grid()

    protocol = (
        load_effective_phase6_protocol()
    )

    raw = {
        "scenario_id": "synthetic_no_feasible",
        "density_target_mode": "hard_constraint",
        "target_electron_density_m3": 1.0e20,
        "density_relative_tolerance": 0.001,
        "minimum_electron_temperature_eV": None,
        "maximum_electron_temperature_eV": None,
        "nominal_operating_point": None,
        "minimum_normalized_boundary_margin": None,
        "objective_priority": [
            "absorbed_power",
        ],
    }

    scenario = parse_runtime_request(
        raw,
        protocol=protocol,
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=False,
        feasible=False,
        objective=None,
        power=None,
        pressure=None,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    response = runtime.run(
        scenario
    )

    assert response.status == (
        "no_feasible_point_found_under_search_protocol"
    )

    assert response.chosen_source is None
    assert response.chosen_operating_point is None
    assert response.chosen_primary_objective is None


def test_added_status_selected_method_accepted_grid_infeasible() -> None:
    scenario_set, grid = _grid()

    protocol = (
        load_effective_phase6_protocol()
    )

    raw = {
        "scenario_id": "synthetic_between_grid_feasible",
        "density_target_mode": "hard_constraint",
        "target_electron_density_m3": 1.001e17,
        "density_relative_tolerance": 1.0e-6,
        "minimum_electron_temperature_eV": None,
        "maximum_electron_temperature_eV": None,
        "nominal_operating_point": None,
        "minimum_normalized_boundary_margin": None,
        "objective_priority": [
            "absorbed_power",
        ],
    }

    scenario = parse_runtime_request(
        raw,
        protocol=protocol,
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    # Exact continuous synthetic density:
    # 2e15 * 50.05 = 1.001e17.
    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=(
            (50.05 - 15.0)
            / 75.0
        ),
        power=50.05,
        pressure=35.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    response = runtime.run(
        scenario
    )

    assert (
        response.grid_reference_result
        .feasible_candidate_count
        == 0
    )

    assert response.status == (
        "selected_method_accepted_grid_infeasible"
    )

    assert response.chosen_source == (
        "differential_evolution"
    )

    assert (
        response.selected_minus_grid_primary_objective
        is None
    )


def test_response_to_dict_is_exactly_repeatable() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "soft_density_5e16",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=0.0,
        power=25.0,
        pressure=35.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    first = runtime.run(
        scenario
    ).to_dict()

    second = runtime.run(
        scenario
    ).to_dict()

    assert first == second


def test_runtime_prediction_vectors_are_read_only() -> None:
    scenario_set, grid = _grid()

    scenario = _scenario(
        scenario_set,
        "soft_density_5e16",
    )

    density, temperature = (
        _synthetic_predictions(
            grid
        )
    )

    result = _selected_result(
        scenario,
        run_pass=True,
        feasible=True,
        objective=0.0,
    )

    runtime = _runtime_for_result(
        scenario_set,
        grid,
        density,
        temperature,
        result,
    )

    assert (
        runtime.grid_density_m3.flags.writeable
        is False
    )

    assert (
        runtime.grid_temperature_eV.flags.writeable
        is False
    )


def test_parse_runtime_request_rejects_multiobjective() -> None:
    scenario_set, _ = _grid()

    scenario = _scenario(
        scenario_set,
        "soft_density_1e17_low_power",
    )

    raw = {
        "scenario_id": scenario.scenario_id,
        "density_target_mode": scenario.density_target_mode,
        "target_electron_density_m3": (
            scenario.target_electron_density_m3
        ),
        "density_relative_tolerance": (
            scenario.density_relative_tolerance
        ),
        "minimum_electron_temperature_eV": (
            scenario.minimum_electron_temperature_eV
        ),
        "maximum_electron_temperature_eV": (
            scenario.maximum_electron_temperature_eV
        ),
        "nominal_operating_point": (
            None
            if scenario.nominal_operating_point is None
            else {
                "nominal_absorbed_power_W": (
                    scenario.nominal_operating_point
                    .nominal_absorbed_power_W
                ),
                "target_pressure_mTorr": (
                    scenario.nominal_operating_point
                    .target_pressure_mTorr
                ),
            }
        ),
        "minimum_normalized_boundary_margin": (
            scenario.minimum_normalized_boundary_margin
        ),
        "objective_priority": list(
            scenario.objective_priority
        ),
    }

    with pytest.raises(
        ProductionRuntimeError,
        match="multiobjective",
    ):
        parse_runtime_request(
            raw
        )
