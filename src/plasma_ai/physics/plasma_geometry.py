"""Geometry and wall-loss relations for the cylindrical argon global model."""

from __future__ import annotations

import math

from plasma_ai.physics.constants import (
    ARGON_ION_MASS_KG,
    ARGON_ION_NEUTRAL_CROSS_SECTION_M2,
    ELEMENTARY_CHARGE_C,
)


def _validate_positive_finite(value: float, name: str) -> float:
    """Validate that a scalar physical quantity is finite and positive."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")

    return value


def cylindrical_volume_m3(radius_m: float, length_m: float) -> float:
    """Return the volume of a cylindrical reactor in m^3."""
    radius = _validate_positive_finite(radius_m, "radius_m")
    length = _validate_positive_finite(length_m, "length_m")

    return math.pi * radius**2 * length


def bohm_velocity_m_s(
    te_eV: float,
    ion_mass_kg: float = ARGON_ION_MASS_KG,
) -> float:
    """Return the classical Bohm velocity for a cold-ion plasma.

    The electron temperature is supplied in eV.
    """
    te = _validate_positive_finite(te_eV, "te_eV")
    ion_mass = _validate_positive_finite(ion_mass_kg, "ion_mass_kg")

    return math.sqrt(ELEMENTARY_CHARGE_C * te / ion_mass)


def ion_mean_free_path_m(
    neutral_density_m3: float,
    cross_section_m2: float = ARGON_ION_NEUTRAL_CROSS_SECTION_M2,
) -> float:
    """Return the effective ion-neutral mean free path in metres."""
    neutral_density = _validate_positive_finite(
        neutral_density_m3,
        "neutral_density_m3",
    )
    cross_section = _validate_positive_finite(
        cross_section_m2,
        "cross_section_m2",
    )

    return 1.0 / (neutral_density * cross_section)


def axial_edge_factor(
    length_m: float,
    mean_free_path_m: float,
) -> float:
    """Return the axial edge-to-centre density factor h_L."""
    length = _validate_positive_finite(length_m, "length_m")
    mean_free_path = _validate_positive_finite(
        mean_free_path_m,
        "mean_free_path_m",
    )

    return 0.86 * (3.0 + length / (2.0 * mean_free_path)) ** -0.5


def radial_edge_factor(
    radius_m: float,
    mean_free_path_m: float,
) -> float:
    """Return the radial edge-to-centre density factor h_R."""
    radius = _validate_positive_finite(radius_m, "radius_m")
    mean_free_path = _validate_positive_finite(
        mean_free_path_m,
        "mean_free_path_m",
    )

    return 0.80 * (4.0 + radius / (2.0 * mean_free_path)) ** -0.5


def effective_loss_area_m2(
    radius_m: float,
    length_m: float,
    h_l: float,
    h_r: float,
) -> float:
    """Return the effective ion-loss area for a cylindrical reactor."""
    radius = _validate_positive_finite(radius_m, "radius_m")
    length = _validate_positive_finite(length_m, "length_m")
    h_l = _validate_positive_finite(h_l, "h_l")
    h_r = _validate_positive_finite(h_r, "h_r")

    return 2.0 * math.pi * (
        radius**2 * h_l
        + radius * length * h_r
    )


def wall_loss_rate_s(
    te_eV: float,
    neutral_density_m3: float,
    radius_m: float,
    length_m: float,
    cross_section_m2: float = ARGON_ION_NEUTRAL_CROSS_SECTION_M2,
) -> float:
    """Return the effective charged-particle wall-loss frequency in s^-1."""
    mean_free_path = ion_mean_free_path_m(
        neutral_density_m3,
        cross_section_m2,
    )

    h_l = axial_edge_factor(length_m, mean_free_path)
    h_r = radial_edge_factor(radius_m, mean_free_path)

    area = effective_loss_area_m2(
        radius_m,
        length_m,
        h_l,
        h_r,
    )

    volume = cylindrical_volume_m3(radius_m, length_m)
    u_b = bohm_velocity_m_s(te_eV)

    return u_b * area / volume
