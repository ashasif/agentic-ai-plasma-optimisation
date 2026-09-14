"""Pure Phase 6 optimisation mathematics and feasibility primitives.

No model loading, surrogate inference, search algorithm, SciPy optimiser,
source-model execution, or TEST-data access occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
    OptimisationScenario,
)


class DecisionPointError(ValueError):
    """Raised when an operating point violates the frozen domain."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """Pure evaluation of one already-predicted operating point."""

    operating_point: OperatingPoint
    electron_density_m3: float
    electron_temperature_eV: float
    domain_feasible: bool
    prediction_valid: bool
    scenario_feasible: bool
    optimisation_success: bool
    density_target_achieved: bool | None
    constraint_failures: tuple[str, ...]
    metrics: dict[str, float]


def _finite_real(
    value: float,
    *,
    field: str,
) -> float:
    """Validate a finite real scalar and reject booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise DecisionPointError(
            f"{field} must be a real number."
        )

    result = float(value)

    if not math.isfinite(result):
        raise DecisionPointError(
            f"{field} must be finite."
        )

    return result


def _domain_bounds(
    protocol: EffectivePhase6Protocol,
) -> tuple[float, float, float, float]:
    """Return frozen inclusive operating-domain bounds."""

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


def validate_decision_point(
    nominal_absorbed_power_W: float,
    target_pressure_mTorr: float,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> OperatingPoint:
    """Validate an operating point without clipping or extrapolation."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    power = _finite_real(
        nominal_absorbed_power_W,
        field="nominal_absorbed_power_W",
    )

    pressure = _finite_real(
        target_pressure_mTorr,
        field="target_pressure_mTorr",
    )

    (
        power_lower,
        power_upper,
        pressure_lower,
        pressure_upper,
    ) = _domain_bounds(protocol)

    if not power_lower <= power <= power_upper:
        raise DecisionPointError(
            "nominal_absorbed_power_W lies outside "
            "the qualified Phase 4 domain."
        )

    if not pressure_lower <= pressure <= pressure_upper:
        raise DecisionPointError(
            "target_pressure_mTorr lies outside "
            "the qualified Phase 4 domain."
        )

    return OperatingPoint(
        nominal_absorbed_power_W=power,
        target_pressure_mTorr=pressure,
    )


def density_relative_error(
    predicted_density_m3: float,
    target_density_m3: float,
) -> float:
    """Return absolute relative electron-density error."""

    predicted = float(predicted_density_m3)
    target = float(target_density_m3)

    if (
        not math.isfinite(predicted)
        or predicted <= 0.0
    ):
        raise ValueError(
            "predicted_density_m3 must be finite and positive."
        )

    if (
        not math.isfinite(target)
        or target <= 0.0
    ):
        raise ValueError(
            "target_density_m3 must be finite and positive."
        )

    return abs(predicted - target) / target


def normalized_absorbed_power(
    point: OperatingPoint,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> float:
    """Return absorbed-power coordinate normalized to [0, 1]."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    (
        power_lower,
        power_upper,
        _,
        _,
    ) = _domain_bounds(protocol)

    validated = validate_decision_point(
        point.nominal_absorbed_power_W,
        point.target_pressure_mTorr,
        protocol=protocol,
    )

    return (
        validated.nominal_absorbed_power_W
        - power_lower
    ) / (
        power_upper
        - power_lower
    )


def normalized_nominal_distance(
    point: OperatingPoint,
    nominal_point: OperatingPoint,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> float:
    """Return Euclidean distance normalized by each domain width."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    (
        power_lower,
        power_upper,
        pressure_lower,
        pressure_upper,
    ) = _domain_bounds(protocol)

    validated_point = validate_decision_point(
        point.nominal_absorbed_power_W,
        point.target_pressure_mTorr,
        protocol=protocol,
    )

    validated_nominal = validate_decision_point(
        nominal_point.nominal_absorbed_power_W,
        nominal_point.target_pressure_mTorr,
        protocol=protocol,
    )

    power_width = power_upper - power_lower
    pressure_width = pressure_upper - pressure_lower

    power_delta = (
        validated_point.nominal_absorbed_power_W
        - validated_nominal.nominal_absorbed_power_W
    ) / power_width

    pressure_delta = (
        validated_point.target_pressure_mTorr
        - validated_nominal.target_pressure_mTorr
    ) / pressure_width

    return math.sqrt(
        power_delta * power_delta
        + pressure_delta * pressure_delta
    )


def normalized_boundary_margin(
    point: OperatingPoint,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> float:
    """Return minimum normalized distance to the domain boundary."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    (
        power_lower,
        power_upper,
        pressure_lower,
        pressure_upper,
    ) = _domain_bounds(protocol)

    validated = validate_decision_point(
        point.nominal_absorbed_power_W,
        point.target_pressure_mTorr,
        protocol=protocol,
    )

    power_width = power_upper - power_lower
    pressure_width = pressure_upper - pressure_lower

    return min(
        (
            validated.nominal_absorbed_power_W
            - power_lower
        ) / power_width,
        (
            power_upper
            - validated.nominal_absorbed_power_W
        ) / power_width,
        (
            validated.target_pressure_mTorr
            - pressure_lower
        ) / pressure_width,
        (
            pressure_upper
            - validated.target_pressure_mTorr
        ) / pressure_width,
    )


def boundary_margin_cost(
    point: OperatingPoint,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> float:
    """Return the frozen minimization-form boundary-margin cost."""

    return 0.5 - normalized_boundary_margin(
        point,
        protocol=protocol,
    )


def _positive_finite_prediction(
    value: float,
) -> bool:
    """Return whether one predicted physical scalar is valid."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(numeric)
        and numeric > 0.0
    )


def evaluate_candidate(
    point: OperatingPoint,
    *,
    electron_density_m3: float,
    electron_temperature_eV: float,
    scenario: OptimisationScenario,
    protocol: EffectivePhase6Protocol | None = None,
) -> CandidateEvaluation:
    """Evaluate hard constraints and objective components purely."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    validated_point = validate_decision_point(
        point.nominal_absorbed_power_W,
        point.target_pressure_mTorr,
        protocol=protocol,
    )

    density_valid = _positive_finite_prediction(
        electron_density_m3
    )

    temperature_valid = _positive_finite_prediction(
        electron_temperature_eV
    )

    prediction_valid = (
        density_valid
        and temperature_valid
    )

    margin = normalized_boundary_margin(
        validated_point,
        protocol=protocol,
    )

    metrics: dict[str, float] = {
        "normalized_absorbed_power":
            normalized_absorbed_power(
                validated_point,
                protocol=protocol,
            ),
        "normalized_boundary_margin": margin,
        "boundary_margin_cost": 0.5 - margin,
    }

    failures: list[str] = []

    density_achieved: bool | None = None

    if not prediction_valid:
        failures.append("invalid_prediction")

    else:
        density = float(
            electron_density_m3
        )

        temperature = float(
            electron_temperature_eV
        )

        target = scenario.target_electron_density_m3

        if target is not None:
            relative_error = density_relative_error(
                density,
                target,
            )

            metrics[
                "density_relative_error"
            ] = relative_error

        if scenario.nominal_operating_point is not None:
            metrics[
                "normalized_nominal_distance"
            ] = normalized_nominal_distance(
                validated_point,
                scenario.nominal_operating_point,
                protocol=protocol,
            )

        if (
            scenario.density_target_mode
            == "hard_constraint"
        ):
            assert target is not None
            assert (
                scenario.density_relative_tolerance
                is not None
            )

            density_achieved = (
                metrics["density_relative_error"]
                <= scenario.density_relative_tolerance
            )

            if not density_achieved:
                failures.append(
                    "density_target"
                )

        if (
            scenario.minimum_electron_temperature_eV
            is not None
            and temperature
            < scenario.minimum_electron_temperature_eV
        ):
            failures.append(
                "minimum_electron_temperature"
            )

        if (
            scenario.maximum_electron_temperature_eV
            is not None
            and temperature
            > scenario.maximum_electron_temperature_eV
        ):
            failures.append(
                "maximum_electron_temperature"
            )

    if (
        scenario.minimum_normalized_boundary_margin
        is not None
        and margin
        < scenario.minimum_normalized_boundary_margin
    ):
        failures.append(
            "minimum_boundary_margin"
        )

    scenario_feasible = (
        prediction_valid
        and not failures
    )

    return CandidateEvaluation(
        operating_point=validated_point,
        electron_density_m3=float(
            electron_density_m3
        ),
        electron_temperature_eV=float(
            electron_temperature_eV
        ),
        domain_feasible=True,
        prediction_valid=prediction_valid,
        scenario_feasible=scenario_feasible,
        optimisation_success=scenario_feasible,
        density_target_achieved=density_achieved,
        constraint_failures=tuple(failures),
        metrics=metrics,
    )


def lexicographic_objective_key(
    evaluation: CandidateEvaluation,
    scenario: OptimisationScenario,
) -> tuple[float, ...]:
    """Return minimization-form key for feasible-candidate ranking."""

    if not evaluation.scenario_feasible:
        raise ValueError(
            "Infeasible candidates must not be ranked "
            "against feasible candidates."
        )

    values: list[float] = []

    for objective in scenario.objective_priority:
        if objective == "density_target_error":
            metric = "density_relative_error"

        elif objective == "nominal_distance":
            metric = "normalized_nominal_distance"

        elif objective == "absorbed_power":
            metric = "normalized_absorbed_power"

        elif objective == "boundary_margin":
            metric = "boundary_margin_cost"

        else:
            raise ValueError(
                f"Unsupported objective {objective!r}."
            )

        if metric not in evaluation.metrics:
            raise ValueError(
                f"Required objective metric {metric!r} "
                "is unavailable."
            )

        values.append(
            float(evaluation.metrics[metric])
        )

    return tuple(values)
