"""Coupled zero-dimensional argon ICP global-model equations.

State vector
------------
y[0] = neutral argon density n0 [m^-3]
y[1] = positive-ion density ni [m^-3]
y[2] = electron energy density Pe [J m^-3]

For the simplified electropositive bulk plasma, quasineutrality gives:

    ne = ni

The model combines neutral-particle, charged-particle, and
electron-energy balances.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plasma_ai.physics.argon_rates import ionization_rate_m3_s
from plasma_ai.physics.energy_losses import (
    collisional_power_loss_density_w_m3,
    electron_temperature_eV_from_energy_density,
    wall_power_loss_density_w_m3,
)
from plasma_ai.physics.gas_flow import (
    pumping_loss_rate_s,
    sccm_to_particles_per_s,
)
from plasma_ai.physics.plasma_geometry import (
    cylindrical_volume_m3,
    wall_loss_rate_s,
)


@dataclass(frozen=True)
class GlobalModelParameters:
    """Fixed operating/model parameters for one simulation."""

    radius_m: float
    length_m: float
    absorbed_power_W: float
    flow_sccm: float
    pumping_speed_m3_s: float
    gas_temperature_K: float = 300.0

    def __post_init__(self) -> None:
        positive_fields = {
            "radius_m": self.radius_m,
            "length_m": self.length_m,
            "gas_temperature_K": self.gas_temperature_K,
        }

        nonnegative_fields = {
            "absorbed_power_W": self.absorbed_power_W,
            "flow_sccm": self.flow_sccm,
            "pumping_speed_m3_s": self.pumping_speed_m3_s,
        }

        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and strictly positive."
                )

        for name, value in nonnegative_fields.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )


def _validated_state(state: ArrayLike) -> NDArray[np.float64]:
    """Return a validated physical-state vector."""
    y = np.asarray(state, dtype=float)

    if y.shape != (3,):
        raise ValueError(
            "State must contain exactly [n0, ni, Pe]."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError("State must contain only finite values.")

    if np.any(y <= 0.0):
        raise ValueError(
            "n0, ni, and Pe must all be strictly positive."
        )

    return y


def electron_temperature_from_state_eV(
    state: ArrayLike,
) -> float:
    """Calculate electron temperature from ni and Pe."""
    y = _validated_state(state)

    ni = float(y[1])
    pe = float(y[2])

    # Quasineutrality: ne = ni
    return electron_temperature_eV_from_energy_density(
        pe,
        ni,
    )


def global_model_terms(
    state: ArrayLike,
    params: GlobalModelParameters,
) -> dict[str, float]:
    """Return individual physical source/loss terms.

    Exposing the terms separately makes conservation checks,
    diagnostics, and later fault reasoning traceable.
    """
    y = _validated_state(state)

    n0 = float(y[0])
    ni = float(y[1])
    pe = float(y[2])

    # Quasineutrality in the electropositive bulk plasma.
    ne = ni

    volume = cylindrical_volume_m3(
        params.radius_m,
        params.length_m,
    )

    te = electron_temperature_eV_from_energy_density(
        pe,
        ne,
    )

    k_iz = ionization_rate_m3_s(te)

    charged_wall_loss_rate = wall_loss_rate_s(
        te,
        n0,
        params.radius_m,
        params.length_m,
    )

    neutral_feed_density_rate = (
        sccm_to_particles_per_s(params.flow_sccm)
        / volume
    )

    ionization_density_rate = (
        k_iz
        * n0
        * ne
    )

    charged_wall_loss_density_rate = (
        charged_wall_loss_rate
        * ni
    )

    neutral_pumping_rate = pumping_loss_rate_s(
        params.pumping_speed_m3_s,
        volume,
    )

    neutral_pumping_density_rate = (
        neutral_pumping_rate
        * n0
    )

    absorbed_power_density = (
        params.absorbed_power_W
        / volume
    )

    collisional_power_loss_density = (
        collisional_power_loss_density_w_m3(
            te,
            n0,
            ne,
        )
    )

    wall_power_loss_density = wall_power_loss_density_w_m3(
        te,
        ne,
        charged_wall_loss_rate,
    )

    return {
        "volume_m3": volume,
        "electron_density_m3": ne,
        "electron_temperature_eV": te,
        "ionization_rate_coefficient_m3_s": k_iz,
        "charged_wall_loss_rate_s": charged_wall_loss_rate,
        "neutral_feed_density_rate_m3_s": neutral_feed_density_rate,
        "ionization_density_rate_m3_s": ionization_density_rate,
        "charged_wall_loss_density_rate_m3_s": (
            charged_wall_loss_density_rate
        ),
        "neutral_pumping_density_rate_m3_s": (
            neutral_pumping_density_rate
        ),
        "absorbed_power_density_W_m3": absorbed_power_density,
        "collisional_power_loss_density_W_m3": (
            collisional_power_loss_density
        ),
        "wall_power_loss_density_W_m3": (
            wall_power_loss_density
        ),
    }


def global_model_rhs(
    time_s: float,
    state: ArrayLike,
    params: GlobalModelParameters,
) -> NDArray[np.float64]:
    """Return [dn0/dt, dni/dt, dPe/dt] for the global model."""
    del time_s  # Autonomous system; kept for solve_ivp compatibility.

    terms = global_model_terms(
        state,
        params,
    )

    feed = terms["neutral_feed_density_rate_m3_s"]
    ionization = terms["ionization_density_rate_m3_s"]
    wall_particles = terms[
        "charged_wall_loss_density_rate_m3_s"
    ]
    pumping = terms[
        "neutral_pumping_density_rate_m3_s"
    ]

    absorbed_power = terms["absorbed_power_density_W_m3"]
    collisional_loss = terms[
        "collisional_power_loss_density_W_m3"
    ]
    wall_power_loss = terms[
        "wall_power_loss_density_W_m3"
    ]

    dn0_dt = (
        feed
        + wall_particles
        - ionization
        - pumping
    )

    dni_dt = (
        ionization
        - wall_particles
    )

    dpe_dt = (
        absorbed_power
        - collisional_loss
        - wall_power_loss
    )

    return np.array(
        [dn0_dt, dni_dt, dpe_dt],
        dtype=float,
    )
