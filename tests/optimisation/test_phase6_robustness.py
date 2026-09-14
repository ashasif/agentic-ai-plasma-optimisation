from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_ai.optimisation import (
    OperatingPoint,
    RobustnessQualificationError,
    allowed_robustness_seeds,
    build_deterministic_grid,
    evaluate_grid_reference,
    evaluate_perturbations,
    load_effective_phase6_protocol,
    load_frozen_grid_scenario_set,
    load_robustness_challenge_scenarios,
    load_robustness_qualification_protocol,
    run_seeded_differential_evolution,
    summarize_tradeoff_scenario,
)


def synthetic_predictor(
    point: OperatingPoint,
):
    """Smooth development-only predictor."""

    density = (
        2.5e15
        * point.nominal_absorbed_power_W
    )

    temperature = (
        1.55
        + 0.005
        * point.target_pressure_mTorr
    )

    return SimpleNamespace(
        electron_density_m3=density,
        electron_temperature_eV=temperature,
    )


def test_robustness_protocol_identity_and_seed_schedule() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    assert robustness.sha256 == (
        "5897dd29172a64e1192d0f40c848c36d717c6c044b8792e8d7fbc3de47a4cfda"
    )

    assert allowed_robustness_seeds(
        robustness
    ) == (
        20260914,
        20260915,
        20260916,
    )


def test_exact_three_challenge_order() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenarios = (
        load_robustness_challenge_scenarios(
            robustness
        )
    )

    assert [
        scenario.scenario_id
        for scenario in scenarios
    ] == [
        "hard_density_1e17_min_power",
        "phase6e_boundary_margin_constraint_edge",
        "phase6e_infeasible_high_density",
    ]


def test_unapproved_seed_is_rejected() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenario = (
        load_robustness_challenge_scenarios(
            robustness
        )[1]
    )

    with pytest.raises(
        RobustnessQualificationError,
        match="Seed is not",
    ):
        run_seeded_differential_evolution(
            scenario,
            seed=123,
            predictor=synthetic_predictor,
        )


def test_boundary_challenge_all_three_seeds_pass_and_are_stable() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenarios = (
        load_robustness_challenge_scenarios(
            robustness
        )
    )

    boundary = scenarios[1]

    results = [
        run_seeded_differential_evolution(
            boundary,
            seed=seed,
            predictor=synthetic_predictor,
            robustness=robustness,
        )
        for seed
        in allowed_robustness_seeds(
            robustness
        )
    ]

    assert all(
        result.run_pass
        for result in results
    )

    assert all(
        result.final_phase6_feasible
        for result in results
    )

    objectives = [
        result.objective_value
        for result in results
    ]

    assert all(
        value is not None
        for value in objectives
    )

    numeric = [
        float(value)
        for value in objectives
    ]

    spread = (
        max(numeric)
        - min(numeric)
    )

    assert spread <= 1.0e-6

    for result in results:
        assert (
            result.decision_coordinates
            is not None
        )

        power, pressure = (
            result.decision_coordinates
        )

        assert power >= (
            22.5 - 1.0e-6
        )

        assert power <= (
            22.5 + 1.0e-4
        )

        assert (
            15.0 - 1.0e-8
            <= pressure
            <= 55.0 + 1.0e-8
        )


def test_deliberate_infeasible_challenge_never_returns_false_feasible() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    infeasible = (
        load_robustness_challenge_scenarios(
            robustness
        )[2]
    )

    results = [
        run_seeded_differential_evolution(
            infeasible,
            seed=seed,
            predictor=synthetic_predictor,
            robustness=robustness,
        )
        for seed
        in allowed_robustness_seeds(
            robustness
        )
    ]

    assert all(
        result.final_phase6_feasible
        is False
        for result in results
    )

    assert all(
        result.run_pass
        is False
        for result in results
    )


def test_grid_reference_boundary_challenge() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    boundary = (
        load_robustness_challenge_scenarios(
            robustness
        )[1]
    )

    scenario_set = (
        load_frozen_grid_scenario_set()
    )

    grid = build_deterministic_grid(
        scenario_set
    )

    power = grid.input_matrix[
        :,
        0,
    ]

    pressure = grid.input_matrix[
        :,
        1,
    ]

    density = (
        2.5e15 * power
    )

    temperature = (
        1.55
        + 0.005 * pressure
    )

    result = evaluate_grid_reference(
        grid,
        electron_density_m3=density,
        electron_temperature_eV=temperature,
        scenario=boundary,
    )

    assert (
        result.feasible_candidate_count
        > 0
    )

    assert (
        result.selected_candidate_index
        is not None
    )

    selected = grid.input_matrix[
        result.selected_candidate_index
    ]

    assert selected[0] == pytest.approx(
        22.5
    )

    assert (
        15.0
        <= selected[1]
        <= 55.0
    )


def test_perturbation_diagnostic_accounts_for_all_offsets_without_clipping() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenario = (
        load_robustness_challenge_scenarios(
            robustness
        )[0]
    )

    diagnostic = evaluate_perturbations(
        source_point=OperatingPoint(
            nominal_absorbed_power_W=15.0,
            target_pressure_mTorr=10.0,
        ),
        scenario=scenario,
        predictor=synthetic_predictor,
        robustness=robustness,
    )

    assert (
        diagnostic.expected_offset_count
        == 8
    )

    assert len(
        diagnostic.records
    ) == 8

    out_of_domain = [
        record
        for record
        in diagnostic.records
        if record.out_of_domain
    ]

    evaluated = [
        record
        for record
        in diagnostic.records
        if record.evaluated
    ]

    assert len(
        out_of_domain
    ) > 0

    assert len(
        evaluated
    ) > 0

    for record in out_of_domain:
        assert record.evaluated is False
        assert record.primary_objective is None
        assert record.electron_density_m3 is None


def test_perturbation_diagnostic_all_interior_neighbours_evaluated() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenario = (
        load_robustness_challenge_scenarios(
            robustness
        )[0]
    )

    diagnostic = evaluate_perturbations(
        source_point=OperatingPoint(
            nominal_absorbed_power_W=52.5,
            target_pressure_mTorr=35.0,
        ),
        scenario=scenario,
        predictor=synthetic_predictor,
        robustness=robustness,
    )

    assert len(
        diagnostic.records
    ) == 8

    assert all(
        record.evaluated
        for record
        in diagnostic.records
    )

    assert all(
        record.out_of_domain
        is False
        for record
        in diagnostic.records
    )

    assert all(
        record.primary_objective
        is not None
        for record
        in diagnostic.records
    )


def test_tradeoff_summary_uses_exact_grid_pareto_set() -> None:
    scenario_set = (
        load_frozen_grid_scenario_set()
    )

    scenario = next(
        item
        for item
        in scenario_set.scenarios
        if item.scenario_id
        == "soft_density_1e17_low_power"
    )

    assert len(
        scenario.objective_priority
    ) > 1

    grid = build_deterministic_grid(
        scenario_set
    )

    power = grid.input_matrix[
        :,
        0,
    ]

    pressure = grid.input_matrix[
        :,
        1,
    ]

    density = (
        2.5e15 * power
    )

    temperature = (
        1.55
        + 0.005 * pressure
    )

    summary = summarize_tradeoff_scenario(
        grid,
        electron_density_m3=density,
        electron_temperature_eV=temperature,
        scenario=scenario,
    )

    assert (
        summary.feasible_candidate_count
        > 0
    )

    assert (
        summary.pareto_candidate_count
        > 0
    )

    assert len(
        summary.objective_ranges
    ) == len(
        scenario.objective_priority
    )

    assert (
        summary.lexicographic_selected_candidate_index
        is not None
    )

    assert (
        summary.lexicographic_selected_is_pareto
        is True
    )

    assert (
        summary.power_range_W
        is not None
    )

    assert (
        summary.pressure_range_mTorr
        is not None
    )


def test_tradeoff_summary_rejects_single_objective_scenario() -> None:
    robustness = (
        load_robustness_qualification_protocol()
    )

    scenario = (
        load_robustness_challenge_scenarios(
            robustness
        )[0]
    )

    scenario_set = (
        load_frozen_grid_scenario_set()
    )

    grid = build_deterministic_grid(
        scenario_set
    )

    n = grid.input_matrix.shape[0]

    density = np.full(
        n,
        1.0e17,
        dtype=np.float64,
    )

    temperature = np.full(
        n,
        1.8,
        dtype=np.float64,
    )

    with pytest.raises(
        RobustnessQualificationError,
        match="multiobjective",
    ):
        summarize_tradeoff_scenario(
            grid,
            electron_density_m3=density,
            electron_temperature_eV=temperature,
            scenario=scenario,
        )
