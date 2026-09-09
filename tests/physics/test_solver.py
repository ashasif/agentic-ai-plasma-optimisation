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
from plasma_ai.physics.global_model import (
    GlobalModelParameters,
)
from plasma_ai.physics.solver import (
    maximum_relative_rate_s,
    solve_global_model,
)


RADIUS_M = 0.17
LENGTH_M = 0.25
FLOW_SCCM = 20.0
GAS_TEMPERATURE_K = 300.0
ABSORBED_POWER_W = 50.0

PRESSURE_PA = mtorr_to_pa(10.0)

NEUTRAL_DENSITY_M3 = neutral_density_m3_from_pressure_pa(
    PRESSURE_PA,
    GAS_TEMPERATURE_K,
)

PUMPING_SPEED_M3_S = effective_pumping_speed_m3_s(
    FLOW_SCCM,
    PRESSURE_PA,
    GAS_TEMPERATURE_K,
)

PARAMS = GlobalModelParameters(
    radius_m=RADIUS_M,
    length_m=LENGTH_M,
    absorbed_power_W=ABSORBED_POWER_W,
    flow_sccm=FLOW_SCCM,
    pumping_speed_m3_s=PUMPING_SPEED_M3_S,
    gas_temperature_K=GAS_TEMPERATURE_K,
)

INITIAL_ELECTRON_DENSITY_M3 = 1.0e15
INITIAL_TE_EV = 3.0

INITIAL_ENERGY_DENSITY_J_M3 = (
    electron_energy_density_j_m3(
        INITIAL_ELECTRON_DENSITY_M3,
        INITIAL_TE_EV,
    )
)

INITIAL_STATE = np.array(
    [
        NEUTRAL_DENSITY_M3,
        INITIAL_ELECTRON_DENSITY_M3,
        INITIAL_ENERGY_DENSITY_J_M3,
    ],
    dtype=float,
)


def test_solver_completes_successfully():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.success
    assert solution.time_s.shape == (301,)
    assert solution.states.shape == (3, 301)


def test_all_integrated_states_remain_positive():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert np.all(solution.states > 0.0)
    assert np.all(solution.electron_temperature_eV > 0.0)


def test_solution_reaches_steady_state_threshold():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.converged
    assert solution.max_relative_rate_s < 1.0e-6


def test_final_electron_temperature_reference_value():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.electron_temperature_eV[-1] == pytest.approx(
        2.074523232660581,
        rel=1e-6,
    )


def test_final_ion_density_reference_value():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.ion_density_m3[-1] == pytest.approx(
        5.510682259378814e16,
        rel=1e-6,
    )


def test_final_neutral_density_remains_near_target():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.neutral_density_m3[-1] == pytest.approx(
        3.21883218e20,
        rel=1e-6,
    )


def test_final_energy_density_reference_value():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.electron_energy_density_j_m3[
        -1
    ] == pytest.approx(
        2.74742171e-2,
        rel=1e-6,
    )


def test_initial_temperature_is_preserved_at_first_output():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    assert solution.electron_temperature_eV[0] == pytest.approx(
        INITIAL_TE_EV,
        rel=1e-12,
    )


def test_maximum_relative_rate_is_finite():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    rate = maximum_relative_rate_s(
        solution.final_state,
        PARAMS,
    )

    assert np.isfinite(rate)
    assert rate > 0.0


@pytest.mark.parametrize(
    "initial_state",
    [
        [0.0, 1.0e15, 0.01],
        [1.0e20, 0.0, 0.01],
        [1.0e20, 1.0e15, 0.0],
        [1.0e20, 1.0e15],
        [np.nan, 1.0e15, 0.01],
    ],
)
def test_invalid_initial_states_are_rejected(initial_state):
    with pytest.raises(ValueError):
        solve_global_model(
            initial_state,
            PARAMS,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end_time_s": 0.0},
        {"end_time_s": -1.0},
        {"output_points": 1},
        {"relative_tolerance": 0.0},
        {"absolute_tolerance": 0.0},
        {"convergence_threshold_s": 0.0},
    ],
)
def test_invalid_solver_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_global_model(
            INITIAL_STATE,
            PARAMS,
            **kwargs,
        )
