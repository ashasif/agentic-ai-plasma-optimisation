"""Phase 6 optimisation-scenario validation.

The scenario representation is governed by the frozen Phase 6A protocol
and Phase 6A.1 controlled amendment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    load_effective_phase6_protocol,
)


class ScenarioValidationError(ValueError):
    """Raised when a Phase 6 optimisation scenario is invalid."""


@dataclass(frozen=True)
class OperatingPoint:
    """Two-variable operating point inside the qualified domain."""

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float


@dataclass(frozen=True)
class OptimisationScenario:
    """Validated Phase 6 optimisation scenario."""

    scenario_id: str
    density_target_mode: str
    target_electron_density_m3: float | None
    density_relative_tolerance: float | None
    minimum_electron_temperature_eV: float | None
    maximum_electron_temperature_eV: float | None
    nominal_operating_point: OperatingPoint | None
    minimum_normalized_boundary_margin: float | None
    objective_priority: tuple[str, ...]


def _finite_number(
    value: Any,
    *,
    field: str,
) -> float:
    """Validate one finite real scalar, rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ScenarioValidationError(
            f"{field} must be a real number."
        )

    result = float(value)

    if not math.isfinite(result):
        raise ScenarioValidationError(
            f"{field} must be finite."
        )

    return result


def _optional_positive_number(
    value: Any,
    *,
    field: str,
) -> float | None:
    """Validate a nullable strictly positive finite scalar."""

    if value is None:
        return None

    result = _finite_number(
        value,
        field=field,
    )

    if result <= 0.0:
        raise ScenarioValidationError(
            f"{field} must be strictly positive."
        )

    return result


def _domain_bounds(
    protocol: EffectivePhase6Protocol,
) -> tuple[float, float, float, float]:
    """Return frozen inclusive decision-space bounds."""

    domain = protocol.base[
        "frozen_phase4_contract"
    ]["qualified_domain"]

    power = domain[
        "nominal_absorbed_power_W"
    ]

    pressure = domain[
        "target_pressure_mTorr"
    ]

    return (
        float(power["lower"]),
        float(power["upper"]),
        float(pressure["lower"]),
        float(pressure["upper"]),
    )


def _parse_nominal_operating_point(
    raw: Any,
    *,
    protocol: EffectivePhase6Protocol,
) -> OperatingPoint | None:
    """Validate the optional nominal operating point."""

    if raw is None:
        return None

    if not isinstance(raw, dict):
        raise ScenarioValidationError(
            "nominal_operating_point must be an object or null."
        )

    expected = {
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
    }

    observed = set(raw)

    if observed != expected:
        raise ScenarioValidationError(
            "nominal_operating_point must contain exactly "
            "nominal_absorbed_power_W and target_pressure_mTorr."
        )

    power = _finite_number(
        raw["nominal_absorbed_power_W"],
        field=(
            "nominal_operating_point."
            "nominal_absorbed_power_W"
        ),
    )

    pressure = _finite_number(
        raw["target_pressure_mTorr"],
        field=(
            "nominal_operating_point."
            "target_pressure_mTorr"
        ),
    )

    (
        power_lower,
        power_upper,
        pressure_lower,
        pressure_upper,
    ) = _domain_bounds(protocol)

    if not power_lower <= power <= power_upper:
        raise ScenarioValidationError(
            "Nominal absorbed power lies outside the "
            "qualified Phase 4 domain."
        )

    if not pressure_lower <= pressure <= pressure_upper:
        raise ScenarioValidationError(
            "Nominal pressure lies outside the "
            "qualified Phase 4 domain."
        )

    return OperatingPoint(
        nominal_absorbed_power_W=power,
        target_pressure_mTorr=pressure,
    )


def parse_optimisation_scenario(
    raw: dict[str, Any],
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> OptimisationScenario:
    """Validate and materialise one frozen-contract scenario."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    if not isinstance(raw, dict):
        raise ScenarioValidationError(
            "Scenario must be a JSON-like object."
        )

    schema = protocol.amendment["scenario_schema"]

    required_fields = set(
        schema["required_fields"]
    )

    observed_fields = set(raw)

    missing = sorted(
        required_fields - observed_fields
    )

    unknown = sorted(
        observed_fields - required_fields
    )

    if missing:
        raise ScenarioValidationError(
            "Missing required scenario fields: "
            + ", ".join(missing)
        )

    if unknown:
        raise ScenarioValidationError(
            "Unknown scenario fields: "
            + ", ".join(unknown)
        )

    scenario_id = raw["scenario_id"]

    if (
        not isinstance(scenario_id, str)
        or not scenario_id.strip()
    ):
        raise ScenarioValidationError(
            "scenario_id must be a non-empty string."
        )

    density_mode = raw[
        "density_target_mode"
    ]

    allowed_modes = tuple(
        schema[
            "density_target_mode"
        ]["allowed_values"]
    )

    if density_mode not in allowed_modes:
        raise ScenarioValidationError(
            "density_target_mode is not allowed."
        )

    target_density = _optional_positive_number(
        raw["target_electron_density_m3"],
        field="target_electron_density_m3",
    )

    tolerance_raw = raw[
        "density_relative_tolerance"
    ]

    if tolerance_raw is None:
        density_tolerance = None
    else:
        density_tolerance = _finite_number(
            tolerance_raw,
            field="density_relative_tolerance",
        )

        if not 0.0 < density_tolerance <= 1.0:
            raise ScenarioValidationError(
                "density_relative_tolerance must lie "
                "in (0, 1]."
            )

    minimum_temperature = _optional_positive_number(
        raw["minimum_electron_temperature_eV"],
        field="minimum_electron_temperature_eV",
    )

    maximum_temperature = _optional_positive_number(
        raw["maximum_electron_temperature_eV"],
        field="maximum_electron_temperature_eV",
    )

    if (
        minimum_temperature is not None
        and maximum_temperature is not None
        and minimum_temperature > maximum_temperature
    ):
        raise ScenarioValidationError(
            "Minimum electron temperature must not "
            "exceed maximum electron temperature."
        )

    nominal_point = _parse_nominal_operating_point(
        raw["nominal_operating_point"],
        protocol=protocol,
    )

    margin_raw = raw[
        "minimum_normalized_boundary_margin"
    ]

    if margin_raw is None:
        minimum_margin = None
    else:
        minimum_margin = _finite_number(
            margin_raw,
            field=(
                "minimum_normalized_boundary_margin"
            ),
        )

        if not 0.0 <= minimum_margin <= 0.5:
            raise ScenarioValidationError(
                "minimum_normalized_boundary_margin must "
                "lie in [0, 0.5]."
            )

    objective_raw = raw[
        "objective_priority"
    ]

    if not isinstance(objective_raw, list):
        raise ScenarioValidationError(
            "objective_priority must be a list."
        )

    if not objective_raw:
        raise ScenarioValidationError(
            "objective_priority must not be empty."
        )

    if not all(
        isinstance(item, str)
        for item in objective_raw
    ):
        raise ScenarioValidationError(
            "Every objective name must be a string."
        )

    objective_priority = tuple(
        objective_raw
    )

    if len(set(objective_priority)) != len(
        objective_priority
    ):
        raise ScenarioValidationError(
            "objective_priority contains duplicates."
        )

    allowed_objectives = set(
        schema[
            "objective_priority"
        ]["allowed_values"]
    )

    unsupported = sorted(
        set(objective_priority)
        - allowed_objectives
    )

    if unsupported:
        raise ScenarioValidationError(
            "Unsupported objective names: "
            + ", ".join(unsupported)
        )

    has_density_objective = (
        "density_target_error"
        in objective_priority
    )

    has_nominal_objective = (
        "nominal_distance"
        in objective_priority
    )

    if density_mode == "disabled":
        if target_density is not None:
            raise ScenarioValidationError(
                "Disabled density targeting requires "
                "target_electron_density_m3=null."
            )

        if density_tolerance is not None:
            raise ScenarioValidationError(
                "Disabled density targeting requires "
                "density_relative_tolerance=null."
            )

        if has_density_objective:
            raise ScenarioValidationError(
                "density_target_error cannot be active "
                "when density targeting is disabled."
            )

    elif density_mode == "soft_objective":
        if target_density is None:
            raise ScenarioValidationError(
                "Soft density targeting requires an "
                "explicit positive target density."
            )

        if density_tolerance is not None:
            raise ScenarioValidationError(
                "Soft density targeting requires "
                "density_relative_tolerance=null."
            )

        if not has_density_objective:
            raise ScenarioValidationError(
                "Soft density targeting requires "
                "density_target_error in objective_priority."
            )

    elif density_mode == "hard_constraint":
        if target_density is None:
            raise ScenarioValidationError(
                "Hard density targeting requires an "
                "explicit positive target density."
            )

        if density_tolerance is None:
            raise ScenarioValidationError(
                "Hard density targeting requires an "
                "explicit density_relative_tolerance."
            )

    if (
        has_nominal_objective
        and nominal_point is None
    ):
        raise ScenarioValidationError(
            "nominal_distance requires "
            "nominal_operating_point."
        )

    return OptimisationScenario(
        scenario_id=scenario_id,
        density_target_mode=density_mode,
        target_electron_density_m3=target_density,
        density_relative_tolerance=density_tolerance,
        minimum_electron_temperature_eV=minimum_temperature,
        maximum_electron_temperature_eV=maximum_temperature,
        nominal_operating_point=nominal_point,
        minimum_normalized_boundary_margin=minimum_margin,
        objective_priority=objective_priority,
    )
