"""Independent validation diagnostics for global-model solutions."""

from __future__ import annotations

from plasma_ai.physics.global_model import (
    GlobalModelParameters,
    global_model_rhs,
    global_model_terms,
)


def _relative_balance_residual(
    net_value: float,
    *component_magnitudes: float,
) -> float:
    """Return a dimensionless residual normalised by term magnitudes."""
    denominator = sum(abs(value) for value in component_magnitudes)

    if denominator == 0.0:
        return 0.0 if net_value == 0.0 else float("inf")

    return abs(net_value) / denominator


def steady_state_balance_residuals(
    state,
    params: GlobalModelParameters,
) -> dict[str, float]:
    """Calculate independent dimensionless balance residuals."""
    terms = global_model_terms(
        state,
        params,
    )

    rhs = global_model_rhs(
        0.0,
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

    absorbed_power = terms[
        "absorbed_power_density_W_m3"
    ]
    collisional_loss = terms[
        "collisional_power_loss_density_W_m3"
    ]
    wall_power_loss = terms[
        "wall_power_loss_density_W_m3"
    ]

    neutral_residual = _relative_balance_residual(
        rhs[0],
        feed,
        wall_particles,
        ionization,
        pumping,
    )

    ion_residual = _relative_balance_residual(
        rhs[1],
        ionization,
        wall_particles,
    )

    energy_residual = _relative_balance_residual(
        rhs[2],
        absorbed_power,
        collisional_loss,
        wall_power_loss,
    )

    expected_external_particle_change = (
        feed - pumping
    )

    actual_total_particle_change = (
        rhs[0] + rhs[1]
    )

    total_particle_identity_error = (
        actual_total_particle_change
        - expected_external_particle_change
    )

    total_particle_identity_residual = (
        _relative_balance_residual(
            total_particle_identity_error,
            feed,
            pumping,
        )
    )

    return {
        "neutral_particle_balance": neutral_residual,
        "ion_particle_balance": ion_residual,
        "electron_energy_balance": energy_residual,
        "total_particle_identity": (
            total_particle_identity_residual
        ),
        "max_balance_residual": max(
            neutral_residual,
            ion_residual,
            energy_residual,
        ),
    }
