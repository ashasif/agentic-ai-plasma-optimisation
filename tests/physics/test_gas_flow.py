import pytest

from plasma_ai.physics.gas_flow import (
    effective_pumping_speed_m3_s,
    gas_throughput_pa_m3_s,
    mtorr_to_pa,
    neutral_density_m3_from_pressure_pa,
    pa_to_mtorr,
    pressure_pa_from_neutral_density_m3,
    pumping_loss_rate_s,
    sccm_to_particles_per_s,
)
from plasma_ai.physics.plasma_geometry import cylindrical_volume_m3


FLOW_SCCM = 20.0
GAS_TEMPERATURE_K = 300.0
PRESSURE_MTORR = 10.0
RADIUS_M = 0.17
LENGTH_M = 0.25


def test_one_sccm_particle_flow_reference_value():
    assert sccm_to_particles_per_s(1.0) == pytest.approx(
        4.477966852997407e17,
        rel=1e-12,
    )


def test_twenty_sccm_particle_flow_reference_value():
    assert sccm_to_particles_per_s(FLOW_SCCM) == pytest.approx(
        8.955933705994813e18,
        rel=1e-12,
    )


def test_mtorr_to_pa_reference_value():
    assert mtorr_to_pa(PRESSURE_MTORR) == pytest.approx(
        1.3332236842105263,
        rel=1e-12,
    )


def test_pressure_unit_conversion_round_trip():
    pressure_pa = mtorr_to_pa(37.5)

    assert pa_to_mtorr(pressure_pa) == pytest.approx(
        37.5,
        rel=1e-12,
    )


def test_neutral_density_reference_value():
    pressure_pa = mtorr_to_pa(PRESSURE_MTORR)

    assert neutral_density_m3_from_pressure_pa(
        pressure_pa,
        GAS_TEMPERATURE_K,
    ) == pytest.approx(
        3.2188332786743194e20,
        rel=1e-12,
    )


def test_pressure_density_round_trip():
    pressure_pa = mtorr_to_pa(PRESSURE_MTORR)

    density = neutral_density_m3_from_pressure_pa(
        pressure_pa,
        GAS_TEMPERATURE_K,
    )

    recovered_pressure = pressure_pa_from_neutral_density_m3(
        density,
        GAS_TEMPERATURE_K,
    )

    assert recovered_pressure == pytest.approx(
        pressure_pa,
        rel=1e-12,
    )


def test_gas_throughput_reference_value():
    assert gas_throughput_pa_m3_s(
        FLOW_SCCM,
        GAS_TEMPERATURE_K,
    ) == pytest.approx(
        0.0370950027457441,
        rel=1e-12,
    )


def test_effective_pumping_speed_reference_value():
    pressure_pa = mtorr_to_pa(PRESSURE_MTORR)

    assert effective_pumping_speed_m3_s(
        FLOW_SCCM,
        pressure_pa,
        GAS_TEMPERATURE_K,
    ) == pytest.approx(
        0.02782354017938862,
        rel=1e-12,
    )


def test_pumping_loss_rate_reference_value():
    pressure_pa = mtorr_to_pa(PRESSURE_MTORR)

    pumping_speed = effective_pumping_speed_m3_s(
        FLOW_SCCM,
        pressure_pa,
        GAS_TEMPERATURE_K,
    )

    volume = cylindrical_volume_m3(
        RADIUS_M,
        LENGTH_M,
    )

    assert pumping_loss_rate_s(
        pumping_speed,
        volume,
    ) == pytest.approx(
        1.2258142432846113,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    "function,args",
    [
        (sccm_to_particles_per_s, (-1.0,)),
        (neutral_density_m3_from_pressure_pa, (0.0, 300.0)),
        (neutral_density_m3_from_pressure_pa, (1.0, 0.0)),
        (pressure_pa_from_neutral_density_m3, (0.0, 300.0)),
        (mtorr_to_pa, (0.0,)),
        (pa_to_mtorr, (-1.0,)),
        (effective_pumping_speed_m3_s, (20.0, 0.0, 300.0)),
        (pumping_loss_rate_s, (1.0, 0.0)),
    ],
)
def test_invalid_inputs_are_rejected(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_higher_pressure_requires_lower_pumping_speed_at_fixed_flow():
    low_pressure = mtorr_to_pa(10.0)
    high_pressure = mtorr_to_pa(50.0)

    speed_at_low_pressure = effective_pumping_speed_m3_s(
        FLOW_SCCM,
        low_pressure,
        GAS_TEMPERATURE_K,
    )

    speed_at_high_pressure = effective_pumping_speed_m3_s(
        FLOW_SCCM,
        high_pressure,
        GAS_TEMPERATURE_K,
    )

    assert speed_at_high_pressure < speed_at_low_pressure
