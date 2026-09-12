"""Tests for the frozen Phase 4G inference-speed benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import plasma_ai.surrogate.phase4g_benchmark as benchmark


def test_phase4g_benchmark_grid_is_exact_frozen_8_by_8():
    grid = benchmark.build_phase4g_benchmark_grid()

    assert grid.features.shape == (
        64,
        2,
    )

    assert grid.total_points == 64

    assert np.array_equal(
        grid.features[0],
        np.asarray(
            [
                15.0,
                10.0,
            ]
        ),
    )

    assert np.array_equal(
        grid.features[-1],
        np.asarray(
            [
                90.0,
                60.0,
            ]
        ),
    )

    assert np.unique(
        grid.features,
        axis=0,
    ).shape[0] == 64


def test_phase4g_protocol_timing_constants_remain_frozen():
    protocol = benchmark.load_phase4g_protocol()

    spec = protocol[
        "speed_benchmark"
    ]

    assert spec[
        "timing_function"
    ] == "time.perf_counter_ns"

    assert spec[
        "primary_summary"
    ] == "median"

    assert spec[
        "source"
    ][
        "warmup_workloads"
    ] == 1

    assert spec[
        "source"
    ][
        "timed_repetitions"
    ] == 3

    assert spec[
        "surrogate_scalar"
    ][
        "warmup_workloads"
    ] == 5

    assert spec[
        "surrogate_scalar"
    ][
        "timed_repetitions"
    ] == 200

    assert spec[
        "surrogate_batch"
    ][
        "warmup_workloads"
    ] == 10

    assert spec[
        "surrogate_batch"
    ][
        "timed_repetitions"
    ] == 1000


def test_time_workload_uses_warmups_and_preserves_all_raw_timings():
    calls = []

    def workload():
        calls.append(
            "call"
        )

    clock_values = iter(
        [
            10,
            20,
            30,
            55,
            100,
            140,
        ]
    )

    result = benchmark.time_workload(
        workload,
        warmup_workloads=2,
        timed_repetitions=3,
        points_per_workload=5,
        clock=lambda: next(
            clock_values
        ),
    )

    assert len(
        calls
    ) == 5

    assert result.raw_runtime_ns == (
        10,
        25,
        40,
    )

    assert result.median_workload_runtime_ns == 25.0
    assert result.minimum_workload_runtime_ns == 10
    assert result.maximum_workload_runtime_ns == 40
    assert result.mean_workload_runtime_ns == 25.0
    assert result.median_runtime_per_point_ns == 5.0


def test_time_workload_rejects_invalid_counts():
    with pytest.raises(
        ValueError,
        match="warmup_workloads",
    ):
        benchmark.time_workload(
            lambda: None,
            warmup_workloads=-1,
            timed_repetitions=1,
            points_per_workload=1,
        )

    with pytest.raises(
        ValueError,
        match="timed_repetitions",
    ):
        benchmark.time_workload(
            lambda: None,
            warmup_workloads=0,
            timed_repetitions=0,
            points_per_workload=1,
        )

    with pytest.raises(
        ValueError,
        match="points_per_workload",
    ):
        benchmark.time_workload(
            lambda: None,
            warmup_workloads=0,
            timed_repetitions=1,
            points_per_workload=0,
        )


def test_scalar_and_batch_workloads_preserve_feature_order():
    observed_scalar = []
    observed_batch = []

    class DummySurrogate:
        def predict_physical(
            self,
            X,
        ):
            array = np.asarray(
                X,
                dtype=np.float64,
            )

            if array.shape[0] == 1:
                observed_scalar.append(
                    array.copy()
                )
            else:
                observed_batch.append(
                    array.copy()
                )

            return SimpleNamespace(
                electron_density_m3=(
                    array[
                        :,
                        0
                    ]
                    * 10.0
                ),
                electron_temperature_eV=(
                    array[
                        :,
                        1
                    ]
                    / 10.0
                ),
            )

    features = np.asarray(
        [
            [15.0, 10.0],
            [90.0, 60.0],
        ],
        dtype=np.float64,
    )

    surrogate = DummySurrogate()

    scalar_density, scalar_temperature = (
        benchmark.surrogate_scalar_workload(
            features,
            surrogate,
        )
    )

    batch_density, batch_temperature = (
        benchmark.surrogate_batch_workload(
            features,
            surrogate,
        )
    )

    assert len(
        observed_scalar
    ) == 2

    assert np.array_equal(
        observed_scalar[0],
        np.asarray(
            [
                [15.0, 10.0],
            ]
        ),
    )

    assert np.array_equal(
        observed_scalar[1],
        np.asarray(
            [
                [90.0, 60.0],
            ]
        ),
    )

    assert len(
        observed_batch
    ) == 1

    assert np.array_equal(
        observed_batch[0],
        features,
    )

    assert np.array_equal(
        scalar_density,
        batch_density,
    )

    assert np.array_equal(
        scalar_temperature,
        batch_temperature,
    )


def test_source_workload_constructs_canonical_points_without_real_solver(
    monkeypatch,
):
    config = SimpleNamespace(
        fixed_flow_sccm=20.0,
        gas_temperature_K=300.0,
        ion_neutral_cross_section_m2=1.0e-18,
        radius_m=0.17,
        length_m=0.25,
    )

    observed = []

    def fake_simulate(
        point,
        received_config,
    ):
        observed.append(
            point
        )

        assert received_config is config

        return SimpleNamespace(
            simulation_id=point.simulation_id,
            qualification_valid=True,
            integration_success=True,
            true_electron_density_m3=(
                point.nominal_absorbed_power_W
                * 1.0e15
            ),
            true_electron_temperature_eV=(
                point.target_pressure_mTorr
                / 10.0
            ),
        )

    monkeypatch.setattr(
        benchmark,
        "simulate_base_design_point",
        fake_simulate,
    )

    features = np.asarray(
        [
            [15.0, 10.0],
            [90.0, 60.0],
        ],
        dtype=np.float64,
    )

    density, temperature = benchmark.source_workload(
        features,
        config,
    )

    assert len(
        observed
    ) == 2

    assert observed[
        0
    ].nominal_absorbed_power_W == 15.0

    assert observed[
        0
    ].target_pressure_mTorr == 10.0

    assert observed[
        1
    ].nominal_absorbed_power_W == 90.0

    assert observed[
        1
    ].target_pressure_mTorr == 60.0

    assert np.array_equal(
        density,
        np.asarray(
            [
                1.5e16,
                9.0e16,
            ]
        ),
    )

    assert np.array_equal(
        temperature,
        np.asarray(
            [
                1.0,
                6.0,
            ]
        ),
    )


def test_production_benchmark_requires_clean_repository(
    monkeypatch,
):
    monkeypatch.setattr(
        benchmark,
        "_git_state",
        lambda: (
            "abc123",
            False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="clean committed repository",
    ):
        benchmark.run_phase4g_speed_benchmark()
