from __future__ import annotations

import math

import pytest

from plasma_ai.optimisation.primitives import (
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
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
    parse_optimisation_scenario,
)


def _scenario(**updates):
    payload = {
        "scenario_id": "primitive-test",
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

    return parse_optimisation_scenario(payload)


def test_domain_lower_boundary_is_valid() -> None:
    point = validate_decision_point(
        15.0,
        10.0,
    )

    assert point == OperatingPoint(
        15.0,
        10.0,
    )


def test_domain_upper_boundary_is_valid() -> None:
    point = validate_decision_point(
        90.0,
        60.0,
    )

    assert point == OperatingPoint(
        90.0,
        60.0,
    )


@pytest.mark.parametrize(
    ("power", "pressure"),
    [
        (14.999, 10.0),
        (90.001, 10.0),
        (15.0, 9.999),
        (15.0, 60.001),
    ],
)
def test_out_of_domain_point_rejected(
    power,
    pressure,
) -> None:
    with pytest.raises(
        DecisionPointError,
        match="outside",
    ):
        validate_decision_point(
            power,
            pressure,
        )


def test_nonfinite_decision_rejected() -> None:
    with pytest.raises(
        DecisionPointError,
        match="finite",
    ):
        validate_decision_point(
            math.nan,
            35.0,
        )


def test_density_relative_error() -> None:
    assert density_relative_error(
        1.1e17,
        1.0e17,
    ) == pytest.approx(0.1)


def test_absorbed_power_normalization() -> None:
    low = OperatingPoint(
        15.0,
        35.0,
    )

    high = OperatingPoint(
        90.0,
        35.0,
    )

    assert normalized_absorbed_power(
        low
    ) == pytest.approx(0.0)

    assert normalized_absorbed_power(
        high
    ) == pytest.approx(1.0)


def test_nominal_distance_full_diagonal() -> None:
    point = OperatingPoint(
        90.0,
        60.0,
    )

    nominal = OperatingPoint(
        15.0,
        10.0,
    )

    assert normalized_nominal_distance(
        point,
        nominal,
    ) == pytest.approx(
        math.sqrt(2.0)
    )


def test_boundary_margin_at_domain_boundary() -> None:
    point = OperatingPoint(
        15.0,
        35.0,
    )

    assert normalized_boundary_margin(
        point
    ) == pytest.approx(0.0)

    assert boundary_margin_cost(
        point
    ) == pytest.approx(0.5)


def test_boundary_margin_at_domain_centre() -> None:
    point = OperatingPoint(
        52.5,
        35.0,
    )

    assert normalized_boundary_margin(
        point
    ) == pytest.approx(0.5)

    assert boundary_margin_cost(
        point
    ) == pytest.approx(0.0)


def test_valid_prediction_candidate_is_feasible() -> None:
    point = OperatingPoint(
        50.0,
        30.0,
    )

    evaluation = evaluate_candidate(
        point,
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=_scenario(),
    )

    assert evaluation.prediction_valid is True
    assert evaluation.scenario_feasible is True
    assert evaluation.optimisation_success is True
    assert evaluation.constraint_failures == ()


@pytest.mark.parametrize(
    ("density", "temperature"),
    [
        (math.nan, 1.8),
        (1.0e17, math.inf),
        (0.0, 1.8),
        (1.0e17, -1.0),
    ],
)
def test_invalid_prediction_is_infeasible(
    density,
    temperature,
) -> None:
    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=density,
        electron_temperature_eV=temperature,
        scenario=_scenario(),
    )

    assert evaluation.prediction_valid is False
    assert evaluation.scenario_feasible is False
    assert "invalid_prediction" in (
        evaluation.constraint_failures
    )


def test_hard_density_target_passes_inside_tolerance() -> None:
    scenario = _scenario(
        density_target_mode="hard_constraint",
        target_electron_density_m3=1.0e17,
        density_relative_tolerance=0.10,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.09e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.density_target_achieved is True
    assert evaluation.scenario_feasible is True


def test_hard_density_target_fails_outside_tolerance() -> None:
    scenario = _scenario(
        density_target_mode="hard_constraint",
        target_electron_density_m3=1.0e17,
        density_relative_tolerance=0.05,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.10e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.density_target_achieved is False
    assert evaluation.scenario_feasible is False
    assert "density_target" in (
        evaluation.constraint_failures
    )


def test_soft_density_target_is_not_hard_constraint() -> None:
    scenario = _scenario(
        density_target_mode="soft_objective",
        target_electron_density_m3=1.0e17,
        objective_priority=[
            "density_target_error",
        ],
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=2.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is True
    assert (
        evaluation.metrics[
            "density_relative_error"
        ]
        == pytest.approx(1.0)
    )


def test_minimum_temperature_constraint() -> None:
    scenario = _scenario(
        minimum_electron_temperature_eV=1.7,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.6,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is False
    assert (
        "minimum_electron_temperature"
        in evaluation.constraint_failures
    )


def test_maximum_temperature_constraint() -> None:
    scenario = _scenario(
        maximum_electron_temperature_eV=1.7,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is False
    assert (
        "maximum_electron_temperature"
        in evaluation.constraint_failures
    )


def test_temperature_bounds_are_inclusive() -> None:
    scenario = _scenario(
        minimum_electron_temperature_eV=1.8,
        maximum_electron_temperature_eV=1.8,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is True


def test_minimum_boundary_margin_is_hard_constraint() -> None:
    scenario = _scenario(
        minimum_normalized_boundary_margin=0.1,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            15.0,
            35.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is False
    assert (
        "minimum_boundary_margin"
        in evaluation.constraint_failures
    )


def test_lexicographic_key_follows_declared_order() -> None:
    scenario = _scenario(
        density_target_mode="soft_objective",
        target_electron_density_m3=1.0e17,
        nominal_operating_point={
            "nominal_absorbed_power_W": 60.0,
            "target_pressure_mTorr": 40.0,
        },
        objective_priority=[
            "density_target_error",
            "nominal_distance",
            "absorbed_power",
            "boundary_margin",
        ],
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.1e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    key = lexicographic_objective_key(
        evaluation,
        scenario,
    )

    assert len(key) == 4
    assert key[0] == pytest.approx(0.1)
    assert key[1] == pytest.approx(
        evaluation.metrics[
            "normalized_nominal_distance"
        ]
    )
    assert key[2] == pytest.approx(
        evaluation.metrics[
            "normalized_absorbed_power"
        ]
    )
    assert key[3] == pytest.approx(
        evaluation.metrics[
            "boundary_margin_cost"
        ]
    )


def test_boundary_margin_ranking_prefers_larger_margin() -> None:
    scenario = _scenario(
        objective_priority=[
            "boundary_margin",
        ],
    )

    centre = evaluate_candidate(
        OperatingPoint(
            52.5,
            35.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    edge = evaluate_candidate(
        OperatingPoint(
            15.0,
            35.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert (
        lexicographic_objective_key(
            centre,
            scenario,
        )
        <
        lexicographic_objective_key(
            edge,
            scenario,
        )
    )


def test_infeasible_candidate_cannot_be_ranked() -> None:
    scenario = _scenario(
        minimum_electron_temperature_eV=2.0,
    )

    evaluation = evaluate_candidate(
        OperatingPoint(
            50.0,
            30.0,
        ),
        electron_density_m3=1.0e17,
        electron_temperature_eV=1.8,
        scenario=scenario,
    )

    assert evaluation.scenario_feasible is False

    with pytest.raises(
        ValueError,
        match="Infeasible candidates",
    ):
        lexicographic_objective_key(
            evaluation,
            scenario,
        )
