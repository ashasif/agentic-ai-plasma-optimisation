from __future__ import annotations

import math

import pytest

from plasma_ai.optimisation.scenario import (
    ScenarioValidationError,
    parse_optimisation_scenario,
)


def _scenario(**updates):
    payload = {
        "scenario_id": "baseline",
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

    return payload


def test_disabled_density_scenario_is_valid() -> None:
    scenario = parse_optimisation_scenario(
        _scenario()
    )

    assert scenario.scenario_id == "baseline"
    assert scenario.density_target_mode == "disabled"
    assert scenario.objective_priority == (
        "absorbed_power",
    )


def test_unknown_field_is_rejected() -> None:
    raw = _scenario()
    raw["unexpected"] = 1

    with pytest.raises(
        ScenarioValidationError,
        match="Unknown scenario fields",
    ):
        parse_optimisation_scenario(raw)


def test_missing_field_is_rejected() -> None:
    raw = _scenario()
    del raw["scenario_id"]

    with pytest.raises(
        ScenarioValidationError,
        match="Missing required scenario fields",
    ):
        parse_optimisation_scenario(raw)


def test_empty_scenario_id_is_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="scenario_id",
    ):
        parse_optimisation_scenario(
            _scenario(scenario_id="   ")
        )


def test_unknown_density_mode_is_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="density_target_mode",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="other",
            )
        )


def test_disabled_mode_rejects_density_target() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="Disabled density targeting",
    ):
        parse_optimisation_scenario(
            _scenario(
                target_electron_density_m3=1e17,
            )
        )


def test_disabled_mode_rejects_density_objective() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="cannot be active",
    ):
        parse_optimisation_scenario(
            _scenario(
                objective_priority=[
                    "density_target_error",
                ],
            )
        )


def test_soft_density_mode_requires_target() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="requires an explicit positive target",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="soft_objective",
                objective_priority=[
                    "density_target_error",
                ],
            )
        )


def test_soft_density_mode_requires_objective() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="requires density_target_error",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="soft_objective",
                target_electron_density_m3=1e17,
            )
        )


def test_soft_density_mode_rejects_tolerance() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="requires density_relative_tolerance=null",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="soft_objective",
                target_electron_density_m3=1e17,
                density_relative_tolerance=0.05,
                objective_priority=[
                    "density_target_error",
                ],
            )
        )


def test_hard_density_mode_requires_tolerance() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="requires an explicit density_relative_tolerance",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="hard_constraint",
                target_electron_density_m3=1e17,
            )
        )


def test_hard_density_mode_does_not_require_density_objective() -> None:
    scenario = parse_optimisation_scenario(
        _scenario(
            density_target_mode="hard_constraint",
            target_electron_density_m3=1e17,
            density_relative_tolerance=0.05,
            objective_priority=[
                "absorbed_power",
            ],
        )
    )

    assert scenario.density_target_mode == "hard_constraint"


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -0.1,
        1.0001,
    ],
)
def test_invalid_density_tolerance_rejected(
    tolerance,
) -> None:
    with pytest.raises(
        ScenarioValidationError,
        match=r"\(0, 1\]",
    ):
        parse_optimisation_scenario(
            _scenario(
                density_target_mode="hard_constraint",
                target_electron_density_m3=1e17,
                density_relative_tolerance=tolerance,
            )
        )


def test_invalid_temperature_range_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="must not exceed",
    ):
        parse_optimisation_scenario(
            _scenario(
                minimum_electron_temperature_eV=2.0,
                maximum_electron_temperature_eV=1.5,
            )
        )


def test_minimum_only_temperature_allowed() -> None:
    scenario = parse_optimisation_scenario(
        _scenario(
            minimum_electron_temperature_eV=1.5,
        )
    )

    assert (
        scenario.minimum_electron_temperature_eV
        == 1.5
    )


def test_maximum_only_temperature_allowed() -> None:
    scenario = parse_optimisation_scenario(
        _scenario(
            maximum_electron_temperature_eV=2.0,
        )
    )

    assert (
        scenario.maximum_electron_temperature_eV
        == 2.0
    )


def test_nominal_point_inside_domain_is_valid() -> None:
    scenario = parse_optimisation_scenario(
        _scenario(
            nominal_operating_point={
                "nominal_absorbed_power_W": 52.5,
                "target_pressure_mTorr": 35.0,
            },
            objective_priority=[
                "nominal_distance",
            ],
        )
    )

    assert scenario.nominal_operating_point is not None


def test_nominal_point_outside_domain_is_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="outside",
    ):
        parse_optimisation_scenario(
            _scenario(
                nominal_operating_point={
                    "nominal_absorbed_power_W": 100.0,
                    "target_pressure_mTorr": 35.0,
                },
                objective_priority=[
                    "nominal_distance",
                ],
            )
        )


def test_nominal_distance_requires_nominal_point() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="requires nominal_operating_point",
    ):
        parse_optimisation_scenario(
            _scenario(
                objective_priority=[
                    "nominal_distance",
                ],
            )
        )


def test_duplicate_objectives_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="duplicates",
    ):
        parse_optimisation_scenario(
            _scenario(
                objective_priority=[
                    "absorbed_power",
                    "absorbed_power",
                ],
            )
        )


def test_empty_objective_priority_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="must not be empty",
    ):
        parse_optimisation_scenario(
            _scenario(
                objective_priority=[],
            )
        )


def test_unknown_objective_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="Unsupported objective",
    ):
        parse_optimisation_scenario(
            _scenario(
                objective_priority=[
                    "magic_objective",
                ],
            )
        )


@pytest.mark.parametrize(
    "margin",
    [
        -0.01,
        0.5001,
    ],
)
def test_invalid_boundary_margin_rejected(
    margin,
) -> None:
    with pytest.raises(
        ScenarioValidationError,
        match=r"\[0, 0.5\]",
    ):
        parse_optimisation_scenario(
            _scenario(
                minimum_normalized_boundary_margin=margin,
            )
        )


def test_nonfinite_numeric_value_rejected() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="finite",
    ):
        parse_optimisation_scenario(
            _scenario(
                minimum_electron_temperature_eV=math.inf,
            )
        )


def test_boolean_is_not_accepted_as_number() -> None:
    with pytest.raises(
        ScenarioValidationError,
        match="real number",
    ):
        parse_optimisation_scenario(
            _scenario(
                minimum_electron_temperature_eV=True,
            )
        )
