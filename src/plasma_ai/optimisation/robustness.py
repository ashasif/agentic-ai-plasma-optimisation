"""Phase 6E robustness-qualification machinery.

This module implements the already-frozen Phase 6E1 qualification contract.

It is predictor-agnostic and does not import or load the production Phase 4G
surrogate. This permits synthetic qualification before real Phase 6E evidence
is generated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution

from plasma_ai.optimisation.continuous import (
    ContinuousBenchmarkError,
    EffectiveContinuousBenchmark,
    PredictionCache,
    build_constraint_functions,
    build_de_constraints,
    continuous_objective_value,
    evaluate_constraint_functions,
    load_effective_continuous_benchmark,
)
from plasma_ai.optimisation.grid import (
    DeterministicGrid,
    GridScenarioEvaluation,
    evaluate_grid_scenario,
)
from plasma_ai.optimisation.primitives import (
    DecisionPointError,
    evaluate_candidate,
    validate_decision_point,
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


DEFAULT_ROBUSTNESS_PROTOCOL_PATH = Path(
    "configs/phase6/robustness_qualification_protocol.json"
)

EXPECTED_ROBUSTNESS_PROTOCOL_SHA256 = (
    "5897dd29172a64e1192d0f40c848c36d717c6c044b8792e8d7fbc3de47a4cfda"
)

EXPECTED_OPTIMIZER_SELECTION_SHA256 = (
    "c4c4963f1f4c0bee2803c4087e514e768a0e95bacd3d58ef1289f887696c3038"
)


class RobustnessQualificationError(RuntimeError):
    """Raised when the frozen Phase 6E contract is violated."""


@dataclass(frozen=True)
class FrozenRobustnessProtocol:
    """Validated Phase 6E1 protocol."""

    payload: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class SeededDERunResult:
    """One selected-method run under one predeclared Phase 6E seed."""

    scenario_id: str
    seed: int

    decision_coordinates: tuple[float, float] | None
    objective_name: str
    objective_value: float | None

    electron_density_m3: float | None
    electron_temperature_eV: float | None

    final_phase6_feasible: bool
    run_pass: bool

    scipy_success: bool
    scipy_message: str
    scipy_nfev: int | None
    scipy_nit: int | None

    constraint_values: tuple[tuple[str, float], ...]

    prediction_cache_hits: int
    prediction_cache_misses: int


@dataclass(frozen=True)
class PerturbationRecord:
    """One predeclared local operating-point perturbation."""

    delta_power_W: float
    delta_pressure_mTorr: float

    candidate_power_W: float
    candidate_pressure_mTorr: float

    evaluated: bool
    out_of_domain: bool

    electron_density_m3: float | None
    electron_temperature_eV: float | None

    scenario_feasible: bool | None
    constraint_values: tuple[tuple[str, float], ...]

    primary_objective: float | None
    primary_objective_delta: float | None
    normalized_boundary_margin: float | None


@dataclass(frozen=True)
class PerturbationDiagnostic:
    """Complete perturbation result for one Phase 6D source candidate."""

    scenario_id: str
    source_power_W: float
    source_pressure_mTorr: float
    source_primary_objective: float

    expected_offset_count: int
    records: tuple[PerturbationRecord, ...]


@dataclass(frozen=True)
class TradeoffSummary:
    """Exact deterministic-grid Pareto summary for one scenario."""

    scenario_id: str
    feasible_candidate_count: int
    pareto_candidate_count: int

    objective_ranges: tuple[
        tuple[str, float, float],
        ...
    ]

    power_range_W: tuple[float, float] | None
    pressure_range_mTorr: tuple[float, float] | None

    electron_density_range_m3: tuple[float, float] | None
    electron_temperature_range_eV: tuple[float, float] | None

    lexicographic_selected_candidate_index: int | None
    lexicographic_selected_is_pareto: bool


PredictionFunction = Callable[[OperatingPoint], Any]


def load_robustness_qualification_protocol(
    path: str | Path = DEFAULT_ROBUSTNESS_PROTOCOL_PATH,
) -> FrozenRobustnessProtocol:
    """Load the exact frozen Phase 6E1 protocol."""

    source = Path(path)

    observed_hash = file_sha256(
        source
    )

    if observed_hash != EXPECTED_ROBUSTNESS_PROTOCOL_SHA256:
        raise RobustnessQualificationError(
            "Phase 6E robustness protocol SHA-256 mismatch."
        )

    try:
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessQualificationError(
            "Phase 6E robustness protocol could not be loaded."
        ) from exc

    if payload.get("phase") != "6E1":
        raise RobustnessQualificationError(
            "Unexpected robustness-protocol phase."
        )

    if (
        payload.get("artifact")
        != "robustness_qualification_protocol"
    ):
        raise RobustnessQualificationError(
            "Unexpected robustness-protocol artifact."
        )

    selected = payload[
        "selected_method"
    ]

    if selected["name"] != "differential_evolution":
        raise RobustnessQualificationError(
            "Unexpected selected Phase 6E optimizer."
        )

    if selected["parameter_retuning_allowed"] is not False:
        raise RobustnessQualificationError(
            "Phase 6E protocol unexpectedly allows retuning."
        )

    if selected["shgo_executed_in_phase6e"] is not False:
        raise RobustnessQualificationError(
            "Phase 6E protocol unexpectedly allows SHGO."
        )

    provenance = payload[
        "provenance"
    ]

    if (
        provenance["optimizer_selection_sha256"]
        != EXPECTED_OPTIMIZER_SELECTION_SHA256
    ):
        raise RobustnessQualificationError(
            "Phase 6E protocol references wrong optimizer selection."
        )

    seeds = payload[
        "seed_robustness"
    ][
        "seeds"
    ]

    if seeds != [
        20260914,
        20260915,
        20260916,
    ]:
        raise RobustnessQualificationError(
            "Unexpected Phase 6E seed schedule."
        )

    return FrozenRobustnessProtocol(
        payload=payload,
        sha256=observed_hash,
    )


def allowed_robustness_seeds(
    robustness: FrozenRobustnessProtocol,
) -> tuple[int, ...]:
    """Return the exact frozen Phase 6E seed sequence."""

    return tuple(
        int(seed)
        for seed
        in robustness.payload[
            "seed_robustness"
        ]["seeds"]
    )


def load_robustness_challenge_scenarios(
    robustness: FrozenRobustnessProtocol,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> tuple[OptimisationScenario, ...]:
    """Parse the three frozen Phase 6E challenge scenarios."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    challenges = robustness.payload[
        "challenge_suite"
    ]

    if len(challenges) != 3:
        raise RobustnessQualificationError(
            "Unexpected Phase 6E challenge count."
        )

    scenarios = tuple(
        parse_optimisation_scenario(
            item["scenario"],
            protocol=protocol,
        )
        for item
        in challenges
    )

    return scenarios


def _failed_seeded_run(
    *,
    scenario: OptimisationScenario,
    seed: int,
    scipy_result: Any,
    cache: PredictionCache,
) -> SeededDERunResult:
    """Return a structured non-feasible DE run."""

    return SeededDERunResult(
        scenario_id=scenario.scenario_id,
        seed=seed,
        decision_coordinates=None,
        objective_name=scenario.objective_priority[0],
        objective_value=None,
        electron_density_m3=None,
        electron_temperature_eV=None,
        final_phase6_feasible=False,
        run_pass=False,
        scipy_success=bool(
            getattr(
                scipy_result,
                "success",
                False,
            )
        ),
        scipy_message=str(
            getattr(
                scipy_result,
                "message",
                "",
            )
        ),
        scipy_nfev=(
            None
            if getattr(
                scipy_result,
                "nfev",
                None,
            ) is None
            else int(
                scipy_result.nfev
            )
        ),
        scipy_nit=(
            None
            if getattr(
                scipy_result,
                "nit",
                None,
            ) is None
            else int(
                scipy_result.nit
            )
        ),
        constraint_values=(),
        prediction_cache_hits=cache.hits,
        prediction_cache_misses=cache.misses,
    )


def run_seeded_differential_evolution(
    scenario: OptimisationScenario,
    *,
    seed: int,
    predictor: PredictionFunction,
    robustness: FrozenRobustnessProtocol | None = None,
    benchmark: EffectiveContinuousBenchmark | None = None,
    protocol: EffectivePhase6Protocol | None = None,
) -> SeededDERunResult:
    """Run selected DE with only the predeclared RNG seed varied."""

    if robustness is None:
        robustness = (
            load_robustness_qualification_protocol()
        )

    if benchmark is None:
        benchmark = (
            load_effective_continuous_benchmark()
        )

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    allowed = allowed_robustness_seeds(
        robustness
    )

    if int(seed) not in allowed:
        raise RobustnessQualificationError(
            "Seed is not in the frozen Phase 6E schedule."
        )

    if len(scenario.objective_priority) != 1:
        raise RobustnessQualificationError(
            "Selected-method Phase 6E challenges must be single-objective."
        )

    config = benchmark.base[
        "differential_evolution"
    ]

    cache = PredictionCache(
        predictor,
        protocol=protocol,
    )

    constraint_functions = build_constraint_functions(
        scenario,
        cache=cache,
        protocol=protocol,
    )

    def objective(
        x: ArrayLike,
    ) -> float:
        return continuous_objective_value(
            x,
            scenario=scenario,
            cache=cache,
            protocol=protocol,
        )

    bounds = [
        tuple(
            float(value)
            for value
            in pair
        )
        for pair
        in benchmark.base[
            "decision_space"
        ]["bounds"]
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy=config["strategy"],
        maxiter=int(
            config["maxiter"]
        ),
        popsize=int(
            config["popsize"]
        ),
        tol=float(
            config["tol"]
        ),
        atol=float(
            config["atol"]
        ),
        mutation=tuple(
            float(value)
            for value
            in config["mutation"]
        ),
        recombination=float(
            config["recombination"]
        ),
        rng=np.random.default_rng(
            int(seed)
        ),
        polish=bool(
            config["polish"]
        ),
        workers=int(
            config["workers"]
        ),
        updating=config[
            "updating"
        ],
        constraints=build_de_constraints(
            constraint_functions
        ),
        vectorized=bool(
            config["vectorized"]
        ),
    )

    raw_x = getattr(
        result,
        "x",
        None,
    )

    if raw_x is None:
        return _failed_seeded_run(
            scenario=scenario,
            seed=int(seed),
            scipy_result=result,
            cache=cache,
        )

    try:
        x = np.asarray(
            raw_x,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return _failed_seeded_run(
            scenario=scenario,
            seed=int(seed),
            scipy_result=result,
            cache=cache,
        )

    if (
        x.shape != (2,)
        or not np.all(
            np.isfinite(x)
        )
    ):
        return _failed_seeded_run(
            scenario=scenario,
            seed=int(seed),
            scipy_result=result,
            cache=cache,
        )

    try:
        prediction = cache.get(
            x
        )

        evaluation = evaluate_candidate(
            prediction.operating_point,
            electron_density_m3=(
                prediction.electron_density_m3
            ),
            electron_temperature_eV=(
                prediction.electron_temperature_eV
            ),
            scenario=scenario,
            protocol=protocol,
        )

        objective_value = (
            continuous_objective_value(
                x,
                scenario=scenario,
                cache=cache,
                protocol=protocol,
            )
        )

        constraint_values = (
            evaluate_constraint_functions(
                constraint_functions,
                x,
            )
        )

    except (
        ContinuousBenchmarkError,
        DecisionPointError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return _failed_seeded_run(
            scenario=scenario,
            seed=int(seed),
            scipy_result=result,
            cache=cache,
        )

    constraints_pass = all(
        value >= 0.0
        for _, value
        in constraint_values
    )

    run_pass = bool(
        evaluation.domain_feasible
        and evaluation.prediction_valid
        and evaluation.scenario_feasible
        and constraints_pass
        and math.isfinite(
            objective_value
        )
    )

    return SeededDERunResult(
        scenario_id=scenario.scenario_id,
        seed=int(seed),
        decision_coordinates=(
            float(
                prediction.operating_point
                .nominal_absorbed_power_W
            ),
            float(
                prediction.operating_point
                .target_pressure_mTorr
            ),
        ),
        objective_name=scenario.objective_priority[0],
        objective_value=float(
            objective_value
        ),
        electron_density_m3=float(
            prediction.electron_density_m3
        ),
        electron_temperature_eV=float(
            prediction.electron_temperature_eV
        ),
        final_phase6_feasible=bool(
            evaluation.scenario_feasible
        ),
        run_pass=run_pass,
        scipy_success=bool(
            getattr(
                result,
                "success",
                False,
            )
        ),
        scipy_message=str(
            getattr(
                result,
                "message",
                "",
            )
        ),
        scipy_nfev=(
            None
            if getattr(
                result,
                "nfev",
                None,
            ) is None
            else int(
                result.nfev
            )
        ),
        scipy_nit=(
            None
            if getattr(
                result,
                "nit",
                None,
            ) is None
            else int(
                result.nit
            )
        ),
        constraint_values=constraint_values,
        prediction_cache_hits=cache.hits,
        prediction_cache_misses=cache.misses,
    )


def evaluate_grid_reference(
    grid: DeterministicGrid,
    *,
    electron_density_m3: ArrayLike,
    electron_temperature_eV: ArrayLike,
    scenario: OptimisationScenario,
    protocol: EffectivePhase6Protocol | None = None,
) -> GridScenarioEvaluation:
    """Evaluate one Phase 6E scenario on the frozen grid contract."""

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    return evaluate_grid_scenario(
        grid,
        electron_density_m3=electron_density_m3,
        electron_temperature_eV=electron_temperature_eV,
        scenario=scenario,
        protocol=protocol,
    )


def evaluate_perturbations(
    *,
    source_point: OperatingPoint,
    scenario: OptimisationScenario,
    predictor: PredictionFunction,
    robustness: FrozenRobustnessProtocol | None = None,
    protocol: EffectivePhase6Protocol | None = None,
) -> PerturbationDiagnostic:
    """Evaluate the exact eight Phase 6E perturbations without clipping."""

    if robustness is None:
        robustness = (
            load_robustness_qualification_protocol()
        )

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    raw_offsets = robustness.payload[
        "perturbation_diagnostics"
    ]["offsets"]

    if len(raw_offsets) != 8:
        raise RobustnessQualificationError(
            "Unexpected perturbation-offset count."
        )

    cache = PredictionCache(
        predictor,
        protocol=protocol,
    )

    constraint_functions = (
        build_constraint_functions(
            scenario,
            cache=cache,
            protocol=protocol,
        )
    )

    source_vector = np.asarray(
        [
            source_point.nominal_absorbed_power_W,
            source_point.target_pressure_mTorr,
        ],
        dtype=np.float64,
    )

    source_objective = (
        continuous_objective_value(
            source_vector,
            scenario=scenario,
            cache=cache,
            protocol=protocol,
        )
    )

    records: list[
        PerturbationRecord
    ] = []

    for raw_delta in raw_offsets:
        delta_power = float(
            raw_delta[0]
        )

        delta_pressure = float(
            raw_delta[1]
        )

        candidate_power = float(
            source_point.nominal_absorbed_power_W
            + delta_power
        )

        candidate_pressure = float(
            source_point.target_pressure_mTorr
            + delta_pressure
        )

        try:
            validated = validate_decision_point(
                candidate_power,
                candidate_pressure,
                protocol=protocol,
            )
        except DecisionPointError:
            records.append(
                PerturbationRecord(
                    delta_power_W=delta_power,
                    delta_pressure_mTorr=delta_pressure,
                    candidate_power_W=candidate_power,
                    candidate_pressure_mTorr=candidate_pressure,
                    evaluated=False,
                    out_of_domain=True,
                    electron_density_m3=None,
                    electron_temperature_eV=None,
                    scenario_feasible=None,
                    constraint_values=(),
                    primary_objective=None,
                    primary_objective_delta=None,
                    normalized_boundary_margin=None,
                )
            )

            continue

        vector = np.asarray(
            [
                validated.nominal_absorbed_power_W,
                validated.target_pressure_mTorr,
            ],
            dtype=np.float64,
        )

        prediction = cache.get(
            vector
        )

        evaluation = evaluate_candidate(
            prediction.operating_point,
            electron_density_m3=(
                prediction.electron_density_m3
            ),
            electron_temperature_eV=(
                prediction.electron_temperature_eV
            ),
            scenario=scenario,
            protocol=protocol,
        )

        objective = (
            continuous_objective_value(
                vector,
                scenario=scenario,
                cache=cache,
                protocol=protocol,
            )
        )

        constraints = (
            evaluate_constraint_functions(
                constraint_functions,
                vector,
            )
        )

        margin = float(
            evaluation.metrics[
                "normalized_boundary_margin"
            ]
        )

        records.append(
            PerturbationRecord(
                delta_power_W=delta_power,
                delta_pressure_mTorr=delta_pressure,
                candidate_power_W=candidate_power,
                candidate_pressure_mTorr=candidate_pressure,
                evaluated=True,
                out_of_domain=False,
                electron_density_m3=float(
                    prediction.electron_density_m3
                ),
                electron_temperature_eV=float(
                    prediction.electron_temperature_eV
                ),
                scenario_feasible=bool(
                    evaluation.scenario_feasible
                ),
                constraint_values=constraints,
                primary_objective=float(
                    objective
                ),
                primary_objective_delta=float(
                    objective
                    - source_objective
                ),
                normalized_boundary_margin=margin,
            )
        )

    if len(records) != 8:
        raise RobustnessQualificationError(
            "Perturbation accounting failed."
        )

    return PerturbationDiagnostic(
        scenario_id=scenario.scenario_id,
        source_power_W=float(
            source_point.nominal_absorbed_power_W
        ),
        source_pressure_mTorr=float(
            source_point.target_pressure_mTorr
        ),
        source_primary_objective=float(
            source_objective
        ),
        expected_offset_count=8,
        records=tuple(
            records
        ),
    )


def _objective_component(
    evaluation,
    objective_name: str,
) -> float:
    """Return frozen minimization-form objective component."""

    mapping = {
        "density_target_error":
            "density_relative_error",
        "nominal_distance":
            "normalized_nominal_distance",
        "absorbed_power":
            "normalized_absorbed_power",
        "boundary_margin":
            "boundary_margin_cost",
    }

    try:
        metric_name = mapping[
            objective_name
        ]
    except KeyError as exc:
        raise RobustnessQualificationError(
            f"Unsupported tradeoff objective {objective_name!r}."
        ) from exc

    try:
        value = float(
            evaluation.metrics[
                metric_name
            ]
        )
    except KeyError as exc:
        raise RobustnessQualificationError(
            f"Missing tradeoff metric {metric_name!r}."
        ) from exc

    if not math.isfinite(value):
        raise RobustnessQualificationError(
            "Tradeoff objective is non-finite."
        )

    return value


def summarize_tradeoff_scenario(
    grid: DeterministicGrid,
    *,
    electron_density_m3: ArrayLike,
    electron_temperature_eV: ArrayLike,
    scenario: OptimisationScenario,
    protocol: EffectivePhase6Protocol | None = None,
) -> TradeoffSummary:
    """Summarize the exact deterministic-grid Pareto set."""

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    if len(
        scenario.objective_priority
    ) <= 1:
        raise RobustnessQualificationError(
            "Tradeoff diagnostics require a multiobjective scenario."
        )

    density = np.asarray(
        electron_density_m3,
        dtype=np.float64,
    )

    temperature = np.asarray(
        electron_temperature_eV,
        dtype=np.float64,
    )

    if density.shape != (
        grid.input_matrix.shape[0],
    ):
        raise RobustnessQualificationError(
            "Density prediction vector shape mismatch."
        )

    if temperature.shape != (
        grid.input_matrix.shape[0],
    ):
        raise RobustnessQualificationError(
            "Temperature prediction vector shape mismatch."
        )

    result = evaluate_grid_scenario(
        grid,
        electron_density_m3=density,
        electron_temperature_eV=temperature,
        scenario=scenario,
        protocol=protocol,
    )

    pareto_indices = tuple(
        int(index)
        for index
        in result.pareto_candidate_indices
    )

    if not pareto_indices:
        return TradeoffSummary(
            scenario_id=scenario.scenario_id,
            feasible_candidate_count=int(
                result.feasible_candidate_count
            ),
            pareto_candidate_count=0,
            objective_ranges=(),
            power_range_W=None,
            pressure_range_mTorr=None,
            electron_density_range_m3=None,
            electron_temperature_range_eV=None,
            lexicographic_selected_candidate_index=(
                None
                if result.selected_candidate_index is None
                else int(
                    result.selected_candidate_index
                )
            ),
            lexicographic_selected_is_pareto=False,
        )

    objective_values: dict[
        str,
        list[float],
    ] = {
        objective: []
        for objective
        in scenario.objective_priority
    }

    powers: list[float] = []
    pressures: list[float] = []
    densities: list[float] = []
    temperatures: list[float] = []

    for index in pareto_indices:
        row = grid.input_matrix[
            index
        ]

        point = OperatingPoint(
            nominal_absorbed_power_W=float(
                row[0]
            ),
            target_pressure_mTorr=float(
                row[1]
            ),
        )

        evaluation = evaluate_candidate(
            point,
            electron_density_m3=float(
                density[index]
            ),
            electron_temperature_eV=float(
                temperature[index]
            ),
            scenario=scenario,
            protocol=protocol,
        )

        if not evaluation.scenario_feasible:
            raise RobustnessQualificationError(
                "Pareto set unexpectedly contains infeasible candidate."
            )

        for objective in (
            scenario.objective_priority
        ):
            objective_values[
                objective
            ].append(
                _objective_component(
                    evaluation,
                    objective,
                )
            )

        powers.append(
            float(
                row[0]
            )
        )

        pressures.append(
            float(
                row[1]
            )
        )

        densities.append(
            float(
                density[index]
            )
        )

        temperatures.append(
            float(
                temperature[index]
            )
        )

    objective_ranges = tuple(
        (
            objective,
            float(
                min(
                    objective_values[
                        objective
                    ]
                )
            ),
            float(
                max(
                    objective_values[
                        objective
                    ]
                )
            ),
        )
        for objective
        in scenario.objective_priority
    )

    selected_index = (
        None
        if result.selected_candidate_index is None
        else int(
            result.selected_candidate_index
        )
    )

    return TradeoffSummary(
        scenario_id=scenario.scenario_id,
        feasible_candidate_count=int(
            result.feasible_candidate_count
        ),
        pareto_candidate_count=len(
            pareto_indices
        ),
        objective_ranges=objective_ranges,
        power_range_W=(
            float(min(powers)),
            float(max(powers)),
        ),
        pressure_range_mTorr=(
            float(min(pressures)),
            float(max(pressures)),
        ),
        electron_density_range_m3=(
            float(min(densities)),
            float(max(densities)),
        ),
        electron_temperature_range_eV=(
            float(min(temperatures)),
            float(max(temperatures)),
        ),
        lexicographic_selected_candidate_index=selected_index,
        lexicographic_selected_is_pareto=bool(
            selected_index
            in pareto_indices
        ),
    )
