"""Continuous optimisation benchmark machinery for Phase 6D.

The module implements the frozen Phase 6D1 benchmark and the controlled
Phase 6D1.1 SHGO compatibility amendment.

The mathematical hard-constraint functions are defined once. Differential
Evolution and SHGO receive method-specific SciPy wrappers around those same
functions.

This module does not load the production Phase 4G surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import (
    NonlinearConstraint,
    differential_evolution,
    shgo,
)

from plasma_ai.optimisation.grid import (
    load_frozen_grid_scenario_set,
)
from plasma_ai.optimisation.primitives import (
    CandidateEvaluation,
    density_relative_error,
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
)


DEFAULT_BENCHMARK_BASE_PATH = Path(
    "configs/phase6/continuous_optimizer_benchmark.json"
)

DEFAULT_BENCHMARK_AMENDMENT_PATH = Path(
    "configs/phase6/"
    "continuous_optimizer_benchmark_amendment_001.json"
)

EXPECTED_BENCHMARK_BASE_SHA256 = (
    "0e618bf046a0d3749393003f15d2015ba2b607b79bc7490c24ac861ca03e9ad9"
)

EXPECTED_BENCHMARK_AMENDMENT_SHA256 = (
    "e54e2e3a6d947f274171afc4e01bd203ea5e617b2f27730add1109321431d62f"
)

EXPECTED_EFFECTIVE_BENCHMARK_SHA256 = (
    "13ce4e5a488d4df6ba5b1e5370a93099f560629e49600ded77be658de8e3ce2a"
)

EXPECTED_GRID_SCENARIO_SHA256 = (
    "6af65defab31691ffbaef1fc91c7a685267d6169af45f277c097488252d0fc1a"
)

EXPECTED_GRID_RESULT_SHA256 = (
    "cb3cbad2345d5a83286e7940fdd7f1960e96b1a3fdd7eea6f0492caa058a7f59"
)


class ContinuousBenchmarkError(RuntimeError):
    """Raised when the frozen continuous benchmark contract is violated."""


@dataclass(frozen=True)
class EffectiveContinuousBenchmark:
    """Frozen base benchmark plus compatibility amendment."""

    base: dict[str, Any]
    amendment: dict[str, Any]
    base_sha256: str
    amendment_sha256: str
    effective_sha256: str


@dataclass(frozen=True)
class CachedPrediction:
    """One physical prediction cached at an exact float64 coordinate."""

    operating_point: OperatingPoint
    electron_density_m3: float
    electron_temperature_eV: float


@dataclass(frozen=True)
class ContinuousRunResult:
    """One method/scenario/repeat benchmark result."""

    method: str
    scenario_id: str
    repeat_index: int
    objective_name: str

    decision_coordinates: tuple[float, float] | None
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


PredictionFunction = Callable[[OperatingPoint], Any]
ConstraintFunction = Callable[[ArrayLike], float]
NamedConstraintFunction = tuple[str, ConstraintFunction]


def effective_benchmark_sha256(
    base_hash: str,
    amendment_hash: str,
) -> str:
    """Return the controlled effective Phase 6D benchmark identity."""

    payload = (
        base_hash
        + "\n"
        + amendment_hash
        + "\n"
    ).encode("ascii")

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_effective_continuous_benchmark(
    base_path: str | Path = DEFAULT_BENCHMARK_BASE_PATH,
    amendment_path: str | Path = DEFAULT_BENCHMARK_AMENDMENT_PATH,
) -> EffectiveContinuousBenchmark:
    """Load and verify the frozen effective continuous benchmark."""

    base_source = Path(base_path)
    amendment_source = Path(amendment_path)

    try:
        base_hash = file_sha256(
            base_source
        )

        amendment_hash = file_sha256(
            amendment_source
        )
    except OSError as exc:
        raise ContinuousBenchmarkError(
            "Continuous benchmark contract file could not be read."
        ) from exc

    if base_hash != EXPECTED_BENCHMARK_BASE_SHA256:
        raise ContinuousBenchmarkError(
            "Continuous benchmark base SHA-256 mismatch."
        )

    if amendment_hash != EXPECTED_BENCHMARK_AMENDMENT_SHA256:
        raise ContinuousBenchmarkError(
            "Continuous benchmark amendment SHA-256 mismatch."
        )

    effective_hash = effective_benchmark_sha256(
        base_hash,
        amendment_hash,
    )

    if effective_hash != EXPECTED_EFFECTIVE_BENCHMARK_SHA256:
        raise ContinuousBenchmarkError(
            "Effective continuous benchmark identity mismatch."
        )

    try:
        base = json.loads(
            base_source.read_text(
                encoding="utf-8"
            )
        )

        amendment = json.loads(
            amendment_source.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuousBenchmarkError(
            "Continuous benchmark JSON could not be loaded."
        ) from exc

    if base.get("phase") != "6D1":
        raise ContinuousBenchmarkError(
            "Unexpected continuous benchmark base phase."
        )

    if amendment.get("phase") != "6D1.1":
        raise ContinuousBenchmarkError(
            "Unexpected continuous benchmark amendment phase."
        )

    reference = amendment.get(
        "base_benchmark_protocol"
    )

    if not isinstance(reference, dict):
        raise ContinuousBenchmarkError(
            "Benchmark amendment base reference is invalid."
        )

    if reference.get("sha256") != base_hash:
        raise ContinuousBenchmarkError(
            "Benchmark amendment references the wrong base hash."
        )

    representation = amendment[
        "effective_hard_constraint_representation"
    ][
        "method_specific_representation"
    ]

    if (
        representation[
            "differential_evolution"
        ]["representation"]
        != "scipy.optimize.NonlinearConstraint"
    ):
        raise ContinuousBenchmarkError(
            "Unexpected Differential Evolution constraint representation."
        )

    if (
        representation["shgo"]["representation"]
        != "scipy_old_style_inequality_dict"
    ):
        raise ContinuousBenchmarkError(
            "Unexpected SHGO constraint representation."
        )

    return EffectiveContinuousBenchmark(
        base=base,
        amendment=amendment,
        base_sha256=base_hash,
        amendment_sha256=amendment_hash,
        effective_sha256=effective_hash,
    )


def load_continuous_qualification_scenarios(
    benchmark: EffectiveContinuousBenchmark,
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> tuple[OptimisationScenario, ...]:
    """Load the exact five frozen single-objective scenarios."""

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    scenario_set = load_frozen_grid_scenario_set(
        protocol=protocol
    )

    if scenario_set.sha256 != EXPECTED_GRID_SCENARIO_SHA256:
        raise ContinuousBenchmarkError(
            "Frozen grid scenario-set drift."
        )

    eligible_ids = tuple(
        benchmark.base[
            "scenario_eligibility"
        ]["qualification_scenarios"]
    )

    mapping = {
        scenario.scenario_id: scenario
        for scenario in scenario_set.scenarios
    }

    try:
        scenarios = tuple(
            mapping[scenario_id]
            for scenario_id in eligible_ids
        )
    except KeyError as exc:
        raise ContinuousBenchmarkError(
            "Frozen continuous qualification scenario is missing."
        ) from exc

    if len(scenarios) != 5:
        raise ContinuousBenchmarkError(
            "Unexpected continuous qualification scenario count."
        )

    for scenario in scenarios:
        if len(scenario.objective_priority) != 1:
            raise ContinuousBenchmarkError(
                "Continuous qualification scenario is not single-objective."
            )

    return scenarios


def _float64_bits(
    value: float,
) -> int:
    scalar = np.asarray(
        [float(value)],
        dtype=np.float64,
    )

    return int(
        scalar.view(
            np.uint64
        )[0]
    )


class PredictionCache:
    """Exact-coordinate, per-run prediction cache."""

    def __init__(
        self,
        predictor: PredictionFunction,
        *,
        protocol: EffectivePhase6Protocol,
    ) -> None:
        self._predictor = predictor
        self._protocol = protocol

        self._cache: dict[
            tuple[int, int],
            CachedPrediction,
        ] = {}

        self.hits = 0
        self.misses = 0

    def get(
        self,
        values: ArrayLike,
    ) -> CachedPrediction:
        """Return a validated physical prediction without clipping."""

        try:
            vector = np.asarray(
                values,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ContinuousBenchmarkError(
                "Continuous decision vector could not be converted to float64."
            ) from exc

        if vector.shape != (2,):
            raise ContinuousBenchmarkError(
                "Continuous decision vector must contain exactly two values."
            )

        point = validate_decision_point(
            float(vector[0]),
            float(vector[1]),
            protocol=self._protocol,
        )

        key = (
            _float64_bits(
                point.nominal_absorbed_power_W
            ),
            _float64_bits(
                point.target_pressure_mTorr
            ),
        )

        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        raw = self._predictor(
            point
        )

        try:
            density = float(
                raw.electron_density_m3
            )

            temperature = float(
                raw.electron_temperature_eV
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContinuousBenchmarkError(
                "Prediction callable violated the physical prediction contract."
            ) from exc

        if not math.isfinite(density) or density <= 0.0:
            raise ContinuousBenchmarkError(
                "Prediction callable returned invalid electron density."
            )

        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ContinuousBenchmarkError(
                "Prediction callable returned invalid electron temperature."
            )

        prediction = CachedPrediction(
            operating_point=point,
            electron_density_m3=density,
            electron_temperature_eV=temperature,
        )

        self._cache[key] = prediction
        self.misses += 1

        return prediction


def _evaluate(
    values: ArrayLike,
    *,
    scenario: OptimisationScenario,
    cache: PredictionCache,
    protocol: EffectivePhase6Protocol,
) -> CandidateEvaluation:
    prediction = cache.get(
        values
    )

    return evaluate_candidate(
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


def _objective_metric_name(
    objective_name: str,
) -> str:
    mapping = {
        "density_target_error":
            "density_relative_error",
        "absorbed_power":
            "normalized_absorbed_power",
        "boundary_margin":
            "boundary_margin_cost",
        "nominal_distance":
            "normalized_nominal_distance",
    }

    try:
        return mapping[
            objective_name
        ]
    except KeyError as exc:
        raise ContinuousBenchmarkError(
            f"Unsupported continuous objective {objective_name!r}."
        ) from exc


def continuous_objective_value(
    values: ArrayLike,
    *,
    scenario: OptimisationScenario,
    cache: PredictionCache,
    protocol: EffectivePhase6Protocol,
) -> float:
    """Evaluate the frozen single minimization objective."""

    if len(scenario.objective_priority) != 1:
        raise ContinuousBenchmarkError(
            "Continuous benchmark supports only single-objective scenarios."
        )

    evaluation = _evaluate(
        values,
        scenario=scenario,
        cache=cache,
        protocol=protocol,
    )

    metric_name = _objective_metric_name(
        scenario.objective_priority[0]
    )

    try:
        value = float(
            evaluation.metrics[
                metric_name
            ]
        )
    except KeyError as exc:
        raise ContinuousBenchmarkError(
            f"Objective metric {metric_name!r} is unavailable."
        ) from exc

    if not math.isfinite(value):
        raise ContinuousBenchmarkError(
            "Continuous objective value is non-finite."
        )

    return value


def build_constraint_functions(
    scenario: OptimisationScenario,
    *,
    cache: PredictionCache,
    protocol: EffectivePhase6Protocol,
) -> tuple[NamedConstraintFunction, ...]:
    """Create each mathematical g(x) function exactly once."""

    functions: list[
        NamedConstraintFunction
    ] = []

    if scenario.density_target_mode == "hard_constraint":
        if scenario.target_electron_density_m3 is None:
            raise ContinuousBenchmarkError(
                "Hard density scenario is missing its target."
            )

        if scenario.density_relative_tolerance is None:
            raise ContinuousBenchmarkError(
                "Hard density scenario is missing its tolerance."
            )

        def density_constraint(
            x: ArrayLike,
        ) -> float:
            prediction = cache.get(
                x
            )

            relative_error = density_relative_error(
                prediction.electron_density_m3,
                scenario.target_electron_density_m3,
            )

            return float(
                scenario.density_relative_tolerance
                - relative_error
            )

        functions.append(
            (
                "density_hard_constraint",
                density_constraint,
            )
        )

    if scenario.minimum_electron_temperature_eV is not None:

        def minimum_temperature_constraint(
            x: ArrayLike,
        ) -> float:
            prediction = cache.get(
                x
            )

            return float(
                prediction.electron_temperature_eV
                - scenario.minimum_electron_temperature_eV
            )

        functions.append(
            (
                "minimum_temperature_constraint",
                minimum_temperature_constraint,
            )
        )

    if scenario.maximum_electron_temperature_eV is not None:

        def maximum_temperature_constraint(
            x: ArrayLike,
        ) -> float:
            prediction = cache.get(
                x
            )

            return float(
                scenario.maximum_electron_temperature_eV
                - prediction.electron_temperature_eV
            )

        functions.append(
            (
                "maximum_temperature_constraint",
                maximum_temperature_constraint,
            )
        )

    if scenario.minimum_normalized_boundary_margin is not None:

        def boundary_margin_constraint(
            x: ArrayLike,
        ) -> float:
            evaluation = _evaluate(
                x,
                scenario=scenario,
                cache=cache,
                protocol=protocol,
            )

            margin = float(
                evaluation.metrics[
                    "normalized_boundary_margin"
                ]
            )

            return float(
                margin
                - scenario.minimum_normalized_boundary_margin
            )

        functions.append(
            (
                "minimum_boundary_margin_constraint",
                boundary_margin_constraint,
            )
        )

    return tuple(
        functions
    )


def evaluate_constraint_functions(
    functions: tuple[NamedConstraintFunction, ...],
    values: ArrayLike,
) -> tuple[tuple[str, float], ...]:
    """Evaluate shared g(x) functions in frozen order."""

    output: list[
        tuple[str, float]
    ] = []

    for name, function in functions:
        value = float(
            function(values)
        )

        if not math.isfinite(value):
            raise ContinuousBenchmarkError(
                f"Constraint {name!r} returned a non-finite value."
            )

        output.append(
            (
                name,
                value,
            )
        )

    return tuple(
        output
    )


def build_de_constraints(
    functions: tuple[NamedConstraintFunction, ...],
) -> tuple[NonlinearConstraint, ...]:
    """Wrap shared g(x) functions for Differential Evolution."""

    return tuple(
        NonlinearConstraint(
            function,
            0.0,
            np.inf,
        )
        for _, function
        in functions
    )


def build_shgo_constraints(
    functions: tuple[NamedConstraintFunction, ...],
) -> tuple[dict[str, Any], ...]:
    """Wrap the same shared g(x) functions for SHGO."""

    return tuple(
        {
            "type": "ineq",
            "fun": function,
        }
        for _, function
        in functions
    )


def _failed_run(
    *,
    method: str,
    scenario: OptimisationScenario,
    repeat_index: int,
    scipy_result: Any,
    cache: PredictionCache,
) -> ContinuousRunResult:
    return ContinuousRunResult(
        method=method,
        scenario_id=scenario.scenario_id,
        repeat_index=repeat_index,
        objective_name=scenario.objective_priority[0],
        decision_coordinates=None,
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
            int(scipy_result.nfev)
            if getattr(
                scipy_result,
                "nfev",
                None,
            ) is not None
            else None
        ),
        scipy_nit=(
            int(scipy_result.nit)
            if getattr(
                scipy_result,
                "nit",
                None,
            ) is not None
            else None
        ),
        constraint_values=(),
        prediction_cache_hits=cache.hits,
        prediction_cache_misses=cache.misses,
    )


def _finalize_run(
    *,
    method: str,
    scenario: OptimisationScenario,
    repeat_index: int,
    scipy_result: Any,
    cache: PredictionCache,
    protocol: EffectivePhase6Protocol,
    constraint_functions: tuple[NamedConstraintFunction, ...],
) -> ContinuousRunResult:
    """Apply independent frozen Phase 6 final validation."""

    raw_x = getattr(
        scipy_result,
        "x",
        None,
    )

    if raw_x is None:
        return _failed_run(
            method=method,
            scenario=scenario,
            repeat_index=repeat_index,
            scipy_result=scipy_result,
            cache=cache,
        )

    try:
        x = np.asarray(
            raw_x,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return _failed_run(
            method=method,
            scenario=scenario,
            repeat_index=repeat_index,
            scipy_result=scipy_result,
            cache=cache,
        )

    if x.shape != (2,) or not np.all(np.isfinite(x)):
        return _failed_run(
            method=method,
            scenario=scenario,
            repeat_index=repeat_index,
            scipy_result=scipy_result,
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

        objective = continuous_objective_value(
            x,
            scenario=scenario,
            cache=cache,
            protocol=protocol,
        )

        constraint_values = evaluate_constraint_functions(
            constraint_functions,
            x,
        )

    except Exception:
        return _failed_run(
            method=method,
            scenario=scenario,
            repeat_index=repeat_index,
            scipy_result=scipy_result,
            cache=cache,
        )

    explicit_constraints_pass = all(
        value >= 0.0
        for _, value
        in constraint_values
    )

    run_pass = bool(
        evaluation.domain_feasible
        and evaluation.prediction_valid
        and evaluation.scenario_feasible
        and explicit_constraints_pass
        and math.isfinite(objective)
    )

    return ContinuousRunResult(
        method=method,
        scenario_id=scenario.scenario_id,
        repeat_index=repeat_index,
        objective_name=scenario.objective_priority[0],
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
        objective_value=float(
            objective
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
            int(scipy_result.nfev)
            if getattr(
                scipy_result,
                "nfev",
                None,
            ) is not None
            else None
        ),
        scipy_nit=(
            int(scipy_result.nit)
            if getattr(
                scipy_result,
                "nit",
                None,
            ) is not None
            else None
        ),
        constraint_values=constraint_values,
        prediction_cache_hits=cache.hits,
        prediction_cache_misses=cache.misses,
    )


def run_continuous_method(
    method: str,
    scenario: OptimisationScenario,
    *,
    predictor: PredictionFunction,
    benchmark: EffectiveContinuousBenchmark | None = None,
    protocol: EffectivePhase6Protocol | None = None,
    repeat_index: int = 0,
) -> ContinuousRunResult:
    """Execute one frozen continuous-method run."""

    if benchmark is None:
        benchmark = load_effective_continuous_benchmark()

    if protocol is None:
        protocol = load_effective_phase6_protocol()

    if repeat_index < 0:
        raise ContinuousBenchmarkError(
            "repeat_index must be non-negative."
        )

    eligible = tuple(
        benchmark.base[
            "scenario_eligibility"
        ]["qualification_scenarios"]
    )

    if scenario.scenario_id not in eligible:
        raise ContinuousBenchmarkError(
            "Scenario is not eligible for the frozen continuous benchmark."
        )

    if len(scenario.objective_priority) != 1:
        raise ContinuousBenchmarkError(
            "Continuous benchmark requires a single-objective scenario."
        )

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

    if method == "differential_evolution":
        config = benchmark.base[
            "differential_evolution"
        ]

        rng = np.random.default_rng(
            int(
                config[
                    "rng_seed"
                ]
            )
        )

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
            rng=rng,
            polish=bool(
                config["polish"]
            ),
            workers=int(
                config["workers"]
            ),
            updating=config["updating"],
            constraints=build_de_constraints(
                constraint_functions
            ),
            vectorized=bool(
                config["vectorized"]
            ),
        )

    elif method == "shgo":
        config = benchmark.base[
            "shgo"
        ]

        shgo_constraints = build_shgo_constraints(
            constraint_functions
        )

        result = shgo(
            objective,
            bounds=bounds,
            constraints=(
                shgo_constraints
                if shgo_constraints
                else None
            ),
            n=int(
                config["n"]
            ),
            iters=int(
                config["iters"]
            ),
            sampling_method=config[
                "sampling_method"
            ],
            workers=int(
                config["workers"]
            ),
        )

    else:
        raise ContinuousBenchmarkError(
            f"Unsupported continuous method {method!r}."
        )

    return _finalize_run(
        method=method,
        scenario=scenario,
        repeat_index=repeat_index,
        scipy_result=result,
        cache=cache,
        protocol=protocol,
        constraint_functions=constraint_functions,
    )


def exact_repeatability(
    first: ContinuousRunResult,
    second: ContinuousRunResult,
) -> bool:
    """Apply the frozen exact repeatability contract."""

    if first.method != second.method:
        return False

    if first.scenario_id != second.scenario_id:
        return False

    return bool(
        first.decision_coordinates
        == second.decision_coordinates
        and first.objective_value
        == second.objective_value
        and first.electron_density_m3
        == second.electron_density_m3
        and first.electron_temperature_eV
        == second.electron_temperature_eV
        and first.final_phase6_feasible
        == second.final_phase6_feasible
    )
