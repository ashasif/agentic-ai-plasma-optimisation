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
    electron_temperature_from_state_eV,
    global_model_rhs,
    global_model_terms,
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

ION_DENSITY_M3 = 1.0e17
TE_EV = 3.0

ENERGY_DENSITY_J_M3 = electron_energy_density_j_m3(
    ION_DENSITY_M3,
    TE_EV,
)

STATE = np.array(
    [
        NEUTRAL_DENSITY_M3,
        ION_DENSITY_M3,
        ENERGY_DENSITY_J_M3,
    ]
)

PARAMS = GlobalModelParameters(
    radius_m=RADIUS_M,
    length_m=LENGTH_M,
    absorbed_power_W=ABSORBED_POWER_W,
    flow_sccm=FLOW_SCCM,
    pumping_speed_m3_s=PUMPING_SPEED_M3_S,
    gas_temperature_K=GAS_TEMPERATURE_K,
)


def test_temperature_is_recovered_from_state():
    assert electron_temperature_from_state_eV(
        STATE
    ) == pytest.approx(
        3.0,
        rel=1e-12,
    )


def test_global_model_term_reference_values():
    terms = global_model_terms(
        STATE,
        PARAMS,
    )

    assert terms[
        "charged_wall_loss_rate_s"
    ] == pytest.approx(
        7340.554664381889,
        rel=1e-10,
    )

    assert terms[
        "neutral_feed_density_rate_m3_s"
    ] == pytest.approx(
        3.9456916797574853e20,
        rel=1e-10,
    )

    assert terms[
        "ionization_density_rate_m3_s"
    ] == pytest.approx(
        8.172879460839276e21,
        rel=1e-10,
    )

    assert terms[
        "charged_wall_loss_density_rate_m3_s"
    ] == pytest.approx(
        7.340554664381889e20,
        rel=1e-10,
    )

    assert terms[
        "neutral_pumping_density_rate_m3_s"
    ] == pytest.approx(
        3.9456916797574853e20,
        rel=1e-10,
    )


def test_global_model_rhs_reference_values():
    rhs = global_model_rhs(
        0.0,
        STATE,
        PARAMS,
    )

    assert rhs[0] == pytest.approx(
        -7.438823994401087e21,
        rel=1e-10,
    )

    assert rhs[1] == pytest.approx(
        7.438823994401087e21,
        rel=1e-10,
    )

    assert rhs[2] == pytest.approx(
        -28135.485893636964,
        rel=1e-10,
    )


def test_internal_particle_processes_cancel():
    terms = global_model_terms(
        STATE,
        PARAMS,
    )

    rhs = global_model_rhs(
        0.0,
        STATE,
        PARAMS,
    )

    expected_external_change = (
        terms["neutral_feed_density_rate_m3_s"]
        - terms["neutral_pumping_density_rate_m3_s"]
    )

    total_particle_change = (
        rhs[0]
        + rhs[1]
    )

    assert total_particle_change == pytest.approx(
        expected_external_change,
        abs=1.0e7,
    )


def test_absorbed_power_density_reference_value():
    terms = global_model_terms(
        STATE,
        PARAMS,
    )

    assert terms[
        "absorbed_power_density_W_m3"
    ] == pytest.approx(
        2202.8365825867863,
        rel=1e-10,
    )


def test_energy_derivative_is_power_balance():
    terms = global_model_terms(
        STATE,
        PARAMS,
    )

    rhs = global_model_rhs(
        0.0,
        STATE,
        PARAMS,
    )

    expected = (
        terms["absorbed_power_density_W_m3"]
        - terms["collisional_power_loss_density_W_m3"]
        - terms["wall_power_loss_density_W_m3"]
    )

    assert rhs[2] == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_global_model_returns_three_derivatives():
    rhs = global_model_rhs(
        0.0,
        STATE,
        PARAMS,
    )

    assert rhs.shape == (3,)
    assert np.all(np.isfinite(rhs))


@pytest.mark.parametrize(
    "state",
    [
        [0.0, 1.0e17, 0.1],
        [1.0e20, 0.0, 0.1],
        [1.0e20, 1.0e17, 0.0],
        [1.0e20, 1.0e17],
        [np.nan, 1.0e17, 0.1],
    ],
)
def test_invalid_states_are_rejected(state):
    with pytest.raises(ValueError):
        global_model_rhs(
            0.0,
            state,
            PARAMS,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius_m": 0.0},
        {"length_m": -1.0},
        {"absorbed_power_W": -1.0},
        {"flow_sccm": -1.0},
        {"pumping_speed_m3_s": -1.0},
        {"gas_temperature_K": 0.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    base = {
        "radius_m": RADIUS_M,
        "length_m": LENGTH_M,
        "absorbed_power_W": ABSORBED_POWER_W,
        "flow_sccm": FLOW_SCCM,
        "pumping_speed_m3_s": PUMPING_SPEED_M3_S,
        "gas_temperature_K": GAS_TEMPERATURE_K,
    }

    base.update(kwargs)

    with pytest.raises(ValueError):
        GlobalModelParameters(**base)
