import numpy as np

from plasma_ai.physics.benchmark import (
    power_sweep,
    pressure_sweep,
)


def test_pressure_sweep_all_points_converge():
    points = pressure_sweep()

    assert all(point.converged for point in points)


def test_power_sweep_all_points_converge():
    points = power_sweep()

    assert all(point.converged for point in points)


def test_electron_density_increases_with_pressure():
    points = pressure_sweep()

    densities = np.array(
        [point.electron_density_m3 for point in points]
    )

    assert np.all(np.diff(densities) > 0.0)


def test_electron_temperature_decreases_with_pressure():
    points = pressure_sweep()

    temperatures = np.array(
        [point.electron_temperature_eV for point in points]
    )

    assert np.all(np.diff(temperatures) < 0.0)


def test_electron_density_increases_with_absorbed_power():
    points = power_sweep()

    densities = np.array(
        [point.electron_density_m3 for point in points]
    )

    assert np.all(np.diff(densities) > 0.0)


def test_all_benchmark_outputs_are_physical():
    points = pressure_sweep() + power_sweep()

    for point in points:
        assert point.electron_density_m3 > 0.0
        assert point.neutral_density_m3 > 0.0
        assert point.electron_temperature_eV > 0.0
        assert np.isfinite(point.max_relative_rate_s)


def test_maxwellian_model_temperature_is_reasonable():
    points = pressure_sweep()

    for point in points:
        assert 0.5 < point.electron_temperature_eV < 10.0
