import numpy as np
import pytest

from plasma_ai.physics.energy_losses import (
    electron_energy_density_j_m3,
)
from plasma_ai.physics.gas_flow import (
    effective_pumping_speed_m3_s,
    mtorr_to_pa,
    neutral_density_m3_from_pressure_pa,
)
from plasma_ai.physics.global_model import GlobalModelParameters
from plasma_ai.physics.solver import solve_global_model


RADIUS_M = 0.17
LENGTH_M = 0.25
FLOW_SCCM = 20.0
PRESSURE_MTORR = 30.0
ABSORBED_POWER_W = 50.0
GAS_TEMPERATURE_K = 300.0


def _environment():
    pressure_pa = mtorr_to_pa(PRESSURE_MTORR)

    neutral_density = neutral_density_m3_from_pressure_pa(
        pressure_pa,
        GAS_TEMPERATURE_K,
    )

    pumping_speed = effective_pumping_speed_m3_s(
        FLOW_SCCM,
        pressure_pa,
        GAS_TEMPERATURE_K,
    )

    return neutral_density, pumping_speed


def _solve(
    initial_ne_m3,
    initial_te_eV,
    cross_section_m2=1.0e-18,
):
    neutral_density, pumping_speed = _environment()

    state = np.array(
        [
            neutral_density,
            initial_ne_m3,
            electron_energy_density_j_m3(
                initial_ne_m3,
                initial_te_eV,
            ),
        ]
    )

    params = GlobalModelParameters(
        radius_m=RADIUS_M,
        length_m=LENGTH_M,
        absorbed_power_W=ABSORBED_POWER_W,
        flow_sccm=FLOW_SCCM,
        pumping_speed_m3_s=pumping_speed,
        gas_temperature_K=GAS_TEMPERATURE_K,
        ion_neutral_cross_section_m2=cross_section_m2,
    )

    return solve_global_model(
        state,
        params,
        end_time_s=20.0,
    )


def test_different_initial_conditions_reach_same_state():
    cases = [
        _solve(1.0e13, 1.5),
        _solve(1.0e15, 3.0),
        _solve(1.0e17, 6.0),
    ]

    assert all(solution.converged for solution in cases)

    final_ne = np.array(
        [solution.ion_density_m3[-1] for solution in cases]
    )

    final_te = np.array(
        [
            solution.electron_temperature_eV[-1]
            for solution in cases
        ]
    )

    assert np.ptp(final_ne) / np.mean(final_ne) < 1.0e-5
    assert np.ptp(final_te) / np.mean(final_te) < 1.0e-5


def test_cross_section_sweep_all_points_converge():
    cross_sections = [
        0.7e-18,
        1.0e-18,
        1.3e-18,
    ]

    solutions = [
        _solve(1.0e15, 3.0, sigma)
        for sigma in cross_sections
    ]

    assert all(solution.converged for solution in solutions)


def test_cross_section_changes_predicted_plasma_state():
    cross_sections = [
        0.7e-18,
        1.0e-18,
        1.3e-18,
    ]

    solutions = [
        _solve(1.0e15, 3.0, sigma)
        for sigma in cross_sections
    ]

    densities = np.array(
        [solution.ion_density_m3[-1] for solution in solutions]
    )

    temperatures = np.array(
        [
            solution.electron_temperature_eV[-1]
            for solution in solutions
        ]
    )

    assert np.all(np.diff(densities) > 0.0)
    assert np.all(np.diff(temperatures) < 0.0)


def test_cross_section_sensitivity_remains_physical():
    for sigma in [0.7e-18, 1.0e-18, 1.3e-18]:
        solution = _solve(
            1.0e15,
            3.0,
            sigma,
        )

        n0 = solution.neutral_density_m3[-1]
        ne = solution.ion_density_m3[-1]
        te = solution.electron_temperature_eV[-1]

        assert 0.0 < ne < n0
        assert 0.5 < te < 10.0


def test_nonpositive_cross_section_is_rejected():
    _, pumping_speed = _environment()

    with pytest.raises(ValueError):
        GlobalModelParameters(
            radius_m=RADIUS_M,
            length_m=LENGTH_M,
            absorbed_power_W=ABSORBED_POWER_W,
            flow_sccm=FLOW_SCCM,
            pumping_speed_m3_s=pumping_speed,
            gas_temperature_K=GAS_TEMPERATURE_K,
            ion_neutral_cross_section_m2=0.0,
        )
