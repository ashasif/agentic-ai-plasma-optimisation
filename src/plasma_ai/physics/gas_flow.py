"""Neutral-gas flow, pressure, and pumping relations.

The reduced-order model treats argon feed using particle flow derived
from sccm and represents chamber evacuation through an effective
lumped pumping speed.

For sccm conversion, this project defines standard conditions as:
    T_std = 273.15 K
    p_std = 101325 Pa

These reference conditions are an explicit modelling convention.
"""

from __future__ import annotations

import math

from scipy.constants import Boltzmann, torr


STANDARD_TEMPERATURE_K = 273.15
STANDARD_PRESSURE_PA = 101325.0

CUBIC_CENTIMETRE_M3 = 1.0e-6
SECONDS_PER_MINUTE = 60.0


def _validate_positive_finite(value: float, name: str) -> float:
    """Return a finite, strictly positive scalar."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")

    return value


def _validate_nonnegative_finite(value: float, name: str) -> float:
    """Return a finite scalar greater than or equal to zero."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def sccm_to_particles_per_s(flow_sccm: float) -> float:
    """Convert argon flow in sccm to particles per second."""
    flow = _validate_nonnegative_finite(flow_sccm, "flow_sccm")

    standard_volume_flow_m3_s = (
        flow
        * CUBIC_CENTIMETRE_M3
        / SECONDS_PER_MINUTE
    )

    return (
        STANDARD_PRESSURE_PA
        * standard_volume_flow_m3_s
        / (Boltzmann * STANDARD_TEMPERATURE_K)
    )


def neutral_density_m3_from_pressure_pa(
    pressure_pa: float,
    gas_temperature_k: float,
) -> float:
    """Return neutral number density from the ideal-gas relation."""
    pressure = _validate_positive_finite(
        pressure_pa,
        "pressure_pa",
    )
    temperature = _validate_positive_finite(
        gas_temperature_k,
        "gas_temperature_k",
    )

    return pressure / (Boltzmann * temperature)


def pressure_pa_from_neutral_density_m3(
    neutral_density_m3: float,
    gas_temperature_k: float,
) -> float:
    """Return pressure from neutral density using the ideal-gas law."""
    density = _validate_positive_finite(
        neutral_density_m3,
        "neutral_density_m3",
    )
    temperature = _validate_positive_finite(
        gas_temperature_k,
        "gas_temperature_k",
    )

    return density * Boltzmann * temperature


def mtorr_to_pa(pressure_mtorr: float) -> float:
    """Convert pressure from mTorr to pascals."""
    pressure = _validate_positive_finite(
        pressure_mtorr,
        "pressure_mtorr",
    )

    return pressure * 1.0e-3 * torr


def pa_to_mtorr(pressure_pa: float) -> float:
    """Convert pressure from pascals to mTorr."""
    pressure = _validate_positive_finite(
        pressure_pa,
        "pressure_pa",
    )

    return pressure / torr * 1.0e3


def gas_throughput_pa_m3_s(
    flow_sccm: float,
    gas_temperature_k: float,
) -> float:
    """Return chamber gas throughput in Pa m^3 s^-1.

    Particle feed is converted to chamber throughput using k_B*T_g.
    """
    temperature = _validate_positive_finite(
        gas_temperature_k,
        "gas_temperature_k",
    )

    particle_flow = sccm_to_particles_per_s(flow_sccm)

    return particle_flow * Boltzmann * temperature


def effective_pumping_speed_m3_s(
    flow_sccm: float,
    target_pressure_pa: float,
    gas_temperature_k: float,
) -> float:
    """Infer effective pumping speed from Q = p*S."""
    pressure = _validate_positive_finite(
        target_pressure_pa,
        "target_pressure_pa",
    )

    throughput = gas_throughput_pa_m3_s(
        flow_sccm,
        gas_temperature_k,
    )

    return throughput / pressure


def pumping_loss_rate_s(
    pumping_speed_m3_s: float,
    reactor_volume_m3: float,
) -> float:
    """Return the first-order neutral pumping loss rate S/V in s^-1."""
    pumping_speed = _validate_nonnegative_finite(
        pumping_speed_m3_s,
        "pumping_speed_m3_s",
    )
    volume = _validate_positive_finite(
        reactor_volume_m3,
        "reactor_volume_m3",
    )

    return pumping_speed / volume
