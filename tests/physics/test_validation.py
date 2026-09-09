import numpy as np

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
    solve_global_model,
)
from plasma_ai.physics.validation import (
    steady_state_balance_residuals,
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

INITIAL_NE_M3 = 1.0e15
INITIAL_TE_EV = 3.0

INITIAL_STATE = np.array(
    [
        NEUTRAL_DENSITY_M3,
        INITIAL_NE_M3,
        electron_energy_density_j_m3(
            INITIAL_NE_M3,
            INITIAL_TE_EV,
        ),
    ]
)


def test_converged_solution_has_small_balance_residuals():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    residuals = steady_state_balance_residuals(
        solution.final_state,
        PARAMS,
    )

    assert residuals[
        "neutral_particle_balance"
    ] < 1.0e-6

    assert residuals[
        "ion_particle_balance"
    ] < 1.0e-6

    assert residuals[
        "electron_energy_balance"
    ] < 1.0e-6

    assert residuals[
        "max_balance_residual"
    ] < 1.0e-6


def test_total_particle_identity_is_numerically_closed():
    solution = solve_global_model(
        INITIAL_STATE,
        PARAMS,
    )

    residuals = steady_state_balance_residuals(
        solution.final_state,
        PARAMS,
    )

    assert residuals[
        "total_particle_identity"
    ] < 1.0e-12


def test_initial_state_is_not_mistaken_for_steady_state():
    residuals = steady_state_balance_residuals(
        INITIAL_STATE,
        PARAMS,
    )

    assert residuals[
        "max_balance_residual"
    ] > 1.0e-2
