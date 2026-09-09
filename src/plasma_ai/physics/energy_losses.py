"""Electron-energy and wall-loss relations for the argon global model."""

from __future__ import annotations

import math

from plasma_ai.physics.argon_rates import (
    elastic_rate_m3_s,
    excitation_rate_m3_s,
    ionization_rate_m3_s,
)
from plasma_ai.physics.constants import (
    ARGON_EXCITATION_ENERGY_EV,
    ARGON_IONIZATION_ENERGY_EV,
    ARGON_ION_MASS_KG,
    ELEMENTARY_CHARGE_C,
    ELECTRON_MASS_KG,
)


def _validate_positive_finite(value: float, name: str) -> float:
    """Validate a finite, strictly positive scalar."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")

    return value


def _validate_nonnegative_finite(value: float, name: str) -> float:
    """Validate a finite scalar greater than or equal to zero."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def sheath_potential_drop_v(
    te_eV: float,
    ion_mass_kg: float = ARGON_ION_MASS_KG,
) -> float:
    """Return the classical floating-wall sheath potential drop in volts.

    For Te supplied in eV:

        Vs = (Te / 2) * ln(mi / (2*pi*me))

    This assumes Maxwellian electrons, cold ions, and a simple
    electropositive plasma adjacent to a floating wall.
    """
    te = _validate_positive_finite(te_eV, "te_eV")
    ion_mass = _validate_positive_finite(ion_mass_kg, "ion_mass_kg")

    mass_ratio_term = ion_mass / (
        2.0 * math.pi * ELECTRON_MASS_KG
    )

    if mass_ratio_term <= 1.0:
        raise ValueError(
            "ion_mass_kg must produce a physically meaningful "
            "ion-to-electron mass ratio."
        )

    return 0.5 * te * math.log(mass_ratio_term)


def electron_wall_energy_loss_eV(te_eV: float) -> float:
    """Mean kinetic energy lost per electron reaching the wall."""
    te = _validate_positive_finite(te_eV, "te_eV")

    return 2.0 * te


def ion_wall_energy_loss_eV(
    te_eV: float,
    ion_mass_kg: float = ARGON_ION_MASS_KG,
) -> float:
    """Mean energy lost per ion reaching the wall, in eV."""
    te = _validate_positive_finite(te_eV, "te_eV")

    sheath_drop = sheath_potential_drop_v(
        te,
        ion_mass_kg,
    )

    return 0.5 * te + sheath_drop


def wall_pair_energy_loss_eV(
    te_eV: float,
    ion_mass_kg: float = ARGON_ION_MASS_KG,
) -> float:
    """Total wall energy lost per electron-ion pair."""
    return (
        electron_wall_energy_loss_eV(te_eV)
        + ion_wall_energy_loss_eV(te_eV, ion_mass_kg)
    )


def collisional_energy_loss_per_pair_eV(
    te_eV: float,
    ion_mass_kg: float = ARGON_ION_MASS_KG,
) -> float:
    """Return collisional energy loss per generated electron-ion pair.

    Includes ionisation, excitation, and elastic-collision terms.
    """
    te = _validate_positive_finite(te_eV, "te_eV")
    ion_mass = _validate_positive_finite(ion_mass_kg, "ion_mass_kg")

    k_iz = ionization_rate_m3_s(te)
    k_exc = excitation_rate_m3_s(te)
    k_elas = elastic_rate_m3_s(te)

    if k_iz <= 0.0:
        raise ValueError(
            "Ionisation rate underflowed to zero; Te is outside "
            "the numerically supported model range."
        )

    excitation_loss = (
        k_exc / k_iz
    ) * ARGON_EXCITATION_ENERGY_EV

    elastic_loss = (
        3.0
        * ELECTRON_MASS_KG
        / ion_mass
        * (k_elas / k_iz)
        * te
    )

    return (
        ARGON_IONIZATION_ENERGY_EV
        + excitation_loss
        + elastic_loss
    )


def electron_energy_density_j_m3(
    electron_density_m3: float,
    te_eV: float,
) -> float:
    """Return electron thermal energy density in J m^-3."""
    ne = _validate_nonnegative_finite(
        electron_density_m3,
        "electron_density_m3",
    )
    te = _validate_positive_finite(te_eV, "te_eV")

    return 1.5 * ELEMENTARY_CHARGE_C * ne * te


def electron_temperature_eV_from_energy_density(
    energy_density_j_m3: float,
    electron_density_m3: float,
) -> float:
    """Recover Te in eV from electron energy density and density."""
    energy_density = _validate_positive_finite(
        energy_density_j_m3,
        "energy_density_j_m3",
    )
    ne = _validate_positive_finite(
        electron_density_m3,
        "electron_density_m3",
    )

    return (
        2.0
        * energy_density
        / (3.0 * ELEMENTARY_CHARGE_C * ne)
    )


def collisional_power_loss_density_w_m3(
    te_eV: float,
    neutral_density_m3: float,
    electron_density_m3: float,
) -> float:
    """Return electron collisional power-loss density in W m^-3."""
    n0 = _validate_nonnegative_finite(
        neutral_density_m3,
        "neutral_density_m3",
    )
    ne = _validate_nonnegative_finite(
        electron_density_m3,
        "electron_density_m3",
    )

    if n0 == 0.0 or ne == 0.0:
        return 0.0

    epsilon_c = collisional_energy_loss_per_pair_eV(te_eV)
    k_iz = ionization_rate_m3_s(te_eV)

    return (
        ELEMENTARY_CHARGE_C
        * epsilon_c
        * k_iz
        * n0
        * ne
    )


def wall_power_loss_density_w_m3(
    te_eV: float,
    electron_density_m3: float,
    wall_loss_rate_s: float,
) -> float:
    """Return electron-ion wall-loss power density in W m^-3."""
    ne = _validate_nonnegative_finite(
        electron_density_m3,
        "electron_density_m3",
    )
    loss_rate = _validate_nonnegative_finite(
        wall_loss_rate_s,
        "wall_loss_rate_s",
    )

    if ne == 0.0 or loss_rate == 0.0:
        return 0.0

    epsilon_wall = wall_pair_energy_loss_eV(te_eV)

    return (
        ELEMENTARY_CHARGE_C
        * epsilon_wall
        * loss_rate
        * ne
    )
