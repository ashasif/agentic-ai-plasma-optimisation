from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_ai.optimisation.continuous import (
    EXPECTED_BENCHMARK_AMENDMENT_SHA256,
    EXPECTED_BENCHMARK_BASE_SHA256,
    EXPECTED_EFFECTIVE_BENCHMARK_SHA256,
    ContinuousBenchmarkError,
    PredictionCache,
    build_constraint_functions,
    build_de_constraints,
    build_shgo_constraints,
    evaluate_constraint_functions,
    exact_repeatability,
    load_continuous_qualification_scenarios,
    load_effective_continuous_benchmark,
    run_continuous_method,
)
from plasma_ai.optimisation.protocol import (
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
)


def synthetic_predictor(
    point: OperatingPoint,
):
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


def _benchmark():
    return load_effective_continuous_benchmark()


def _scenario(
    scenario_id: str,
):
    benchmark = _benchmark()

    scenarios = (
        load_continuous_qualification_scenarios(
            benchmark
        )
    )

    return next(
        scenario
        for scenario in scenarios
        if scenario.scenario_id == scenario_id
    )


def test_effective_benchmark_identity() -> None:
    benchmark = _benchmark()

    assert (
        benchmark.base_sha256
        == EXPECTED_BENCHMARK_BASE_SHA256
    )

    assert (
        benchmark.amendment_sha256
        == EXPECTED_BENCHMARK_AMENDMENT_SHA256
    )

    assert (
        benchmark.effective_sha256
        == EXPECTED_EFFECTIVE_BENCHMARK_SHA256
    )


def test_exact_qualification_scenario_order() -> None:
    benchmark = _benchmark()

    scenarios = (
        load_continuous_qualification_scenarios(
            benchmark
        )
    )

    assert [
        scenario.scenario_id
        for scenario in scenarios
    ] == [
        "soft_density_5e16",
        "hard_density_1e17_min_power",
        "temperature_window_min_power",
        "boundary_margin_reference",
        "nominal_centre_reference",
    ]


def test_prediction_cache_reuses_exact_coordinate() -> None:
    protocol = load_effective_phase6_protocol()

    calls = []

    def predictor(point):
        calls.append(point)
        return synthetic_predictor(point)

    cache = PredictionCache(
        predictor,
        protocol=protocol,
    )

    first = cache.get(
        [52.5, 35.0]
    )

    second = cache.get(
        [52.5, 35.0]
    )

    assert first == second
    assert len(calls) == 1
    assert cache.misses == 1
    assert cache.hits == 1


def test_prediction_cache_does_not_round_nearby_coordinates() -> None:
    protocol = load_effective_phase6_protocol()

    calls = []

    def predictor(point):
        calls.append(point)
        return synthetic_predictor(point)

    cache = PredictionCache(
        predictor,
        protocol=protocol,
    )

    pressure_a = 35.0

    pressure_b = float(
        np.nextafter(
            np.float64(35.0),
            np.float64(36.0),
        )
    )

    cache.get(
        [52.5, pressure_a]
    )

    cache.get(
        [52.5, pressure_b]
    )

    assert len(calls) == 2
    assert cache.misses == 2
    assert cache.hits == 0


def test_out_of_domain_prediction_remains_fail_closed() -> None:
    protocol = load_effective_phase6_protocol()

    cache = PredictionCache(
        synthetic_predictor,
        protocol=protocol,
    )

    with pytest.raises(
        Exception,
        match="outside",
    ):
        cache.get(
            [0.013333333333333334, 0.02]
        )

    assert cache.misses == 0


def test_same_g_function_drives_de_and_shgo_wrappers() -> None:
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "hard_density_1e17_min_power"
    )

    cache = PredictionCache(
        synthetic_predictor,
        protocol=protocol,
    )

    functions = build_constraint_functions(
        scenario,
        cache=cache,
        protocol=protocol,
    )

    assert len(functions) == 1

    raw = evaluate_constraint_functions(
        functions,
        [40.0, 35.0],
    )

    de_constraints = build_de_constraints(
        functions
    )

    shgo_constraints = build_shgo_constraints(
        functions
    )

    assert len(de_constraints) == 1
    assert len(shgo_constraints) == 1

    raw_value = raw[0][1]

    de_value = float(
        np.asarray(
            de_constraints[0].fun(
                np.array(
                    [40.0, 35.0]
                )
            )
        ).reshape(-1)[0]
    )

    shgo_value = float(
        shgo_constraints[0]["fun"](
            np.array(
                [40.0, 35.0]
            )
        )
    )

    assert raw_value == pytest.approx(
        0.05
    )

    assert de_value == raw_value
    assert shgo_value == raw_value


def test_temperature_constraint_signs_and_wrapper_equivalence() -> None:
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "temperature_window_min_power"
    )

    cache = PredictionCache(
        synthetic_predictor,
        protocol=protocol,
    )

    functions = build_constraint_functions(
        scenario,
        cache=cache,
        protocol=protocol,
    )

    assert [
        name
        for name, _
        in functions
    ] == [
        "minimum_temperature_constraint",
        "maximum_temperature_constraint",
    ]

    values = dict(
        evaluate_constraint_functions(
            functions,
            [50.0, 20.0],
        )
    )

    assert (
        values[
            "minimum_temperature_constraint"
        ]
        == pytest.approx(0.0)
    )

    assert (
        values[
            "maximum_temperature_constraint"
        ]
        == pytest.approx(0.20)
    )

    de_constraints = build_de_constraints(
        functions
    )

    shgo_constraints = build_shgo_constraints(
        functions
    )

    for index in range(2):
        de_value = float(
            np.asarray(
                de_constraints[index].fun(
                    np.array(
                        [50.0, 20.0]
                    )
                )
            ).reshape(-1)[0]
        )

        shgo_value = float(
            shgo_constraints[index]["fun"](
                np.array(
                    [50.0, 20.0]
                )
            )
        )

        assert de_value == shgo_value


def test_de_exact_repeatability_with_synthetic_constraint() -> None:
    benchmark = _benchmark()
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "hard_density_1e17_min_power"
    )

    first = run_continuous_method(
        "differential_evolution",
        scenario,
        predictor=synthetic_predictor,
        benchmark=benchmark,
        protocol=protocol,
        repeat_index=0,
    )

    second = run_continuous_method(
        "differential_evolution",
        scenario,
        predictor=synthetic_predictor,
        benchmark=benchmark,
        protocol=protocol,
        repeat_index=1,
    )

    assert first.run_pass is True
    assert second.run_pass is True
    assert first.final_phase6_feasible is True
    assert second.final_phase6_feasible is True

    assert exact_repeatability(
        first,
        second,
    ) is True

    assert first.decision_coordinates is not None

    power = first.decision_coordinates[0]

    assert 38.0 <= power <= 42.0

    assert first.objective_value is not None
    assert np.isfinite(first.objective_value)


def test_shgo_old_style_constraint_executes_without_placeholder_failure() -> None:
    benchmark = _benchmark()
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "temperature_window_min_power"
    )

    result = run_continuous_method(
        "shgo",
        scenario,
        predictor=synthetic_predictor,
        benchmark=benchmark,
        protocol=protocol,
        repeat_index=0,
    )

    assert result.run_pass is True
    assert result.final_phase6_feasible is True
    assert result.decision_coordinates is not None

    power, pressure = result.decision_coordinates

    assert power == pytest.approx(
        15.0,
        abs=1.0e-8,
    )

    assert pressure >= (
        20.0 - 1.0e-8
    )

    assert pressure <= (
        60.0 + 1.0e-8
    )


def test_shgo_exact_repeatability_with_old_style_constraints() -> None:
    benchmark = _benchmark()
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "temperature_window_min_power"
    )

    first = run_continuous_method(
        "shgo",
        scenario,
        predictor=synthetic_predictor,
        benchmark=benchmark,
        protocol=protocol,
        repeat_index=0,
    )

    second = run_continuous_method(
        "shgo",
        scenario,
        predictor=synthetic_predictor,
        benchmark=benchmark,
        protocol=protocol,
        repeat_index=1,
    )

    assert first.run_pass is True
    assert second.run_pass is True

    assert exact_repeatability(
        first,
        second,
    ) is True


def test_shgo_old_style_wrapper_is_dict_inequality() -> None:
    protocol = load_effective_phase6_protocol()

    scenario = _scenario(
        "temperature_window_min_power"
    )

    cache = PredictionCache(
        synthetic_predictor,
        protocol=protocol,
    )

    functions = build_constraint_functions(
        scenario,
        cache=cache,
        protocol=protocol,
    )

    constraints = build_shgo_constraints(
        functions
    )

    assert len(constraints) == 2

    for constraint in constraints:
        assert constraint["type"] == "ineq"
        assert callable(
            constraint["fun"]
        )


def test_ineligible_multiobjective_scenario_is_rejected() -> None:
    scenario_set = load_frozen_grid_scenario_set()

    scenario = next(
        item
        for item in scenario_set.scenarios
        if item.scenario_id
        == "soft_density_1e17_low_power"
    )

    with pytest.raises(
        ContinuousBenchmarkError,
        match="not eligible",
    ):
        run_continuous_method(
            "differential_evolution",
            scenario,
            predictor=synthetic_predictor,
        )


def test_unknown_method_is_rejected() -> None:
    scenario = _scenario(
        "boundary_margin_reference"
    )

    with pytest.raises(
        ContinuousBenchmarkError,
        match="Unsupported continuous method",
    ):
        run_continuous_method(
            "unknown",
            scenario,
            predictor=synthetic_predictor,
        )


# Imported here so the multiobjective rejection test remains explicit.
from plasma_ai.optimisation.grid import (
    load_frozen_grid_scenario_set,
)
