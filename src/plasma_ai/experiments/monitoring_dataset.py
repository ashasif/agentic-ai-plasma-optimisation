"""Canonical monitoring-row schema for Phase-3 synthetic monitoring."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
)
from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
)
from plasma_ai.experiments.monitoring_observation import (
    MonitoringObservation,
)
from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
)
from plasma_ai.experiments.monitoring_process import (
    MonitoringProcessState,
)


APPROVED_MONITORING_FEATURES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
    "nominal_flow_sccm",
    "measured_absorbed_power_W",
    "measured_flow_sccm",
    "measured_pressure_mTorr",
)

APPROVED_BINARY_TARGET = (
    "fault_effect_active"
)

APPROVED_MULTICLASS_TARGET = (
    "active_fault_family"
)


@dataclass(frozen=True)
class MonitoringRow:
    """One canonical synthetic monitoring record.

    The row contains both observation-layer quantities and protected
    ground truth. Downstream ML code must use the feature manifest
    rather than auto-selecting numeric columns.
    """

    monitoring_id: str
    episode_id: str
    split: str
    step_index: int
    steps_per_episode: int

    recipe_design_index: int
    recipe_design_seed: int

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float
    nominal_flow_sccm: float
    nominal_pumping_speed_m3_s: float

    measured_absorbed_power_W: float
    measured_flow_sccm: float
    measured_pressure_mTorr: float

    fault_present: bool
    fault_domain: str
    fault_family: str
    fault_profile: str
    fault_severity: str
    fault_direction: str
    fault_magnitude_fraction: float
    fault_onset_step: int | None

    fault_started: bool
    fault_progress: float
    signed_fault_fraction: float
    fault_factor: float

    fault_effect_active: bool
    active_fault_domain: str
    active_fault_family: str

    normal_power_coupling_factor: float
    normal_flow_delivery_factor: float
    normal_pumping_effectiveness_factor: float

    applied_power_factor: float
    applied_flow_factor: float
    applied_pumping_factor: float

    true_absorbed_power_W: float
    true_flow_sccm: float
    true_pumping_speed_m3_s: float

    true_neutral_density_m3: float
    true_ion_density_m3: float
    true_electron_density_m3: float
    true_electron_energy_density_J_m3: float
    true_electron_temperature_eV: float
    true_pressure_mTorr: float

    true_pressure_relative_deviation_from_target: float

    ionization_rate_coefficient_m3_s: float
    ionization_density_rate_m3_s: float
    charged_wall_loss_rate_s: float
    charged_wall_loss_density_rate_m3_s: float

    ionization_fraction: float
    residence_time_proxy_s: float

    collisional_power_loss_W: float
    wall_power_loss_W: float

    power_measurement_noise_factor: float
    flow_measurement_noise_factor: float
    pressure_measurement_noise_factor: float

    power_measurement_relative_noise: float
    flow_measurement_relative_noise: float
    pressure_measurement_relative_noise: float

    pressure_sensor_bias_factor: float
    pressure_sensor_bias_fraction: float

    integration_success: bool
    converged: bool
    max_relative_rate_s: float

    neutral_particle_balance: float
    ion_particle_balance: float
    electron_energy_balance: float
    total_particle_identity: float
    max_balance_residual: float

    physical_state_valid: bool
    balance_valid: bool

    domain_status: str
    model_validity_status: str
    qualification_valid: bool
    is_ml_eligible: bool

    error_message: str


def monitoring_schema_fields() -> tuple[str, ...]:
    """Return canonical schema order."""
    return tuple(
        field.name
        for field in fields(
            MonitoringRow
        )
    )


def classify_true_domain_status(
    *,
    true_absorbed_power_W: float,
    true_pressure_mTorr: float,
    base_config: BaseDatasetConfig,
) -> str:
    """Classify true physical state against the Phase-3B envelope.

    Domain classification uses true absorbed power and true pressure,
    not noisy observations. A sensor bias therefore cannot turn an
    otherwise supported physical state into physics OOD.
    """
    if (
        not math.isfinite(
            true_absorbed_power_W
        )
        or not math.isfinite(
            true_pressure_mTorr
        )
    ):
        return "ood"

    power_min = (
        base_config.absorbed_power_min_W
    )
    power_max = (
        base_config.absorbed_power_max_W
    )

    pressure_min = (
        base_config.target_pressure_min_mTorr
    )
    pressure_max = (
        base_config.target_pressure_max_mTorr
    )

    if (
        true_absorbed_power_W < power_min
        or true_absorbed_power_W > power_max
        or true_pressure_mTorr < pressure_min
        or true_pressure_mTorr > pressure_max
    ):
        return "ood"

    fraction = (
        base_config.near_boundary_fraction
    )

    power_normalized = (
        (
            true_absorbed_power_W
            - power_min
        )
        / (
            power_max
            - power_min
        )
    )

    pressure_normalized = (
        (
            true_pressure_mTorr
            - pressure_min
        )
        / (
            pressure_max
            - pressure_min
        )
    )

    if (
        power_normalized <= fraction
        or power_normalized >= 1.0 - fraction
        or pressure_normalized <= fraction
        or pressure_normalized >= 1.0 - fraction
    ):
        return "near_boundary"

    return "supported"


def build_monitoring_row(
    *,
    config: MonitoringDatasetConfig,
    base_config: BaseDatasetConfig,
    plan: MonitoringEpisodePlan,
    state: MonitoringProcessState,
    observation: MonitoringObservation,
) -> MonitoringRow:
    """Combine plan, latent truth, diagnostics, and observations."""
    if (
        state.episode_id != plan.episode_id
        or observation.episode_id != plan.episode_id
    ):
        raise ValueError(
            "Plan, process state, and observation episode IDs differ."
        )

    if (
        state.monitoring_id
        != observation.monitoring_id
    ):
        raise ValueError(
            "Process state and observation monitoring IDs differ."
        )

    if (
        state.step_index
        != observation.step_index
    ):
        raise ValueError(
            "Process state and observation step indices differ."
        )

    if (
        state.split != plan.split
        or observation.split != plan.split
    ):
        raise ValueError(
            "Plan, process state, and observation splits differ."
        )

    effect_active = bool(
        abs(
            state.signed_fault_fraction
        )
        > 0.0
    )

    if effect_active:
        active_fault_domain = (
            plan.fault_domain
        )
        active_fault_family = (
            plan.fault_family
        )
    else:
        active_fault_domain = "none"
        active_fault_family = "none"

    domain_status = (
        classify_true_domain_status(
            true_absorbed_power_W=(
                state.true_absorbed_power_W
            ),
            true_pressure_mTorr=(
                state.true_pressure_mTorr
            ),
            base_config=base_config,
        )
        if state.integration_success
        else "ood"
    )

    is_ml_eligible = bool(
        state.qualification_valid
        and domain_status
        in {
            "supported",
            "near_boundary",
        }
    )

    return MonitoringRow(
        monitoring_id=(
            state.monitoring_id
        ),
        episode_id=(
            plan.episode_id
        ),
        split=(
            plan.split
        ),
        step_index=(
            state.step_index
        ),
        steps_per_episode=(
            config.steps_per_episode
        ),
        recipe_design_index=(
            plan.recipe_design_index
        ),
        recipe_design_seed=(
            plan.recipe_design_seed
        ),
        nominal_absorbed_power_W=(
            plan.nominal_absorbed_power_W
        ),
        target_pressure_mTorr=(
            plan.target_pressure_mTorr
        ),
        nominal_flow_sccm=(
            plan.nominal_flow_sccm
        ),
        nominal_pumping_speed_m3_s=(
            state.nominal_pumping_speed_m3_s
        ),
        measured_absorbed_power_W=(
            observation.measured_absorbed_power_W
        ),
        measured_flow_sccm=(
            observation.measured_flow_sccm
        ),
        measured_pressure_mTorr=(
            observation.measured_pressure_mTorr
        ),
        fault_present=bool(
            plan.fault_present
        ),
        fault_domain=(
            plan.fault_domain
        ),
        fault_family=(
            plan.fault_family
        ),
        fault_profile=(
            plan.fault_profile
        ),
        fault_severity=(
            plan.fault_severity
        ),
        fault_direction=(
            plan.fault_direction
        ),
        fault_magnitude_fraction=float(
            plan.fault_magnitude_fraction
        ),
        fault_onset_step=(
            plan.fault_onset_step
        ),
        fault_started=bool(
            state.fault_started
        ),
        fault_progress=float(
            state.fault_progress
        ),
        signed_fault_fraction=float(
            state.signed_fault_fraction
        ),
        fault_factor=float(
            state.fault_factor
        ),
        fault_effect_active=(
            effect_active
        ),
        active_fault_domain=(
            active_fault_domain
        ),
        active_fault_family=(
            active_fault_family
        ),
        normal_power_coupling_factor=float(
            state.normal_power_coupling_factor
        ),
        normal_flow_delivery_factor=float(
            state.normal_flow_delivery_factor
        ),
        normal_pumping_effectiveness_factor=float(
            state.normal_pumping_effectiveness_factor
        ),
        applied_power_factor=float(
            state.applied_power_factor
        ),
        applied_flow_factor=float(
            state.applied_flow_factor
        ),
        applied_pumping_factor=float(
            state.applied_pumping_factor
        ),
        true_absorbed_power_W=float(
            state.true_absorbed_power_W
        ),
        true_flow_sccm=float(
            state.true_flow_sccm
        ),
        true_pumping_speed_m3_s=float(
            state.true_pumping_speed_m3_s
        ),
        true_neutral_density_m3=float(
            state.true_neutral_density_m3
        ),
        true_ion_density_m3=float(
            state.true_ion_density_m3
        ),
        true_electron_density_m3=float(
            state.true_electron_density_m3
        ),
        true_electron_energy_density_J_m3=float(
            state.true_electron_energy_density_J_m3
        ),
        true_electron_temperature_eV=float(
            state.true_electron_temperature_eV
        ),
        true_pressure_mTorr=float(
            state.true_pressure_mTorr
        ),
        true_pressure_relative_deviation_from_target=float(
            state.true_pressure_relative_deviation_from_target
        ),
        ionization_rate_coefficient_m3_s=float(
            state.ionization_rate_coefficient_m3_s
        ),
        ionization_density_rate_m3_s=float(
            state.ionization_density_rate_m3_s
        ),
        charged_wall_loss_rate_s=float(
            state.charged_wall_loss_rate_s
        ),
        charged_wall_loss_density_rate_m3_s=float(
            state.charged_wall_loss_density_rate_m3_s
        ),
        ionization_fraction=float(
            state.ionization_fraction
        ),
        residence_time_proxy_s=float(
            state.residence_time_proxy_s
        ),
        collisional_power_loss_W=float(
            state.collisional_power_loss_W
        ),
        wall_power_loss_W=float(
            state.wall_power_loss_W
        ),
        power_measurement_noise_factor=float(
            observation.power_measurement_noise_factor
        ),
        flow_measurement_noise_factor=float(
            observation.flow_measurement_noise_factor
        ),
        pressure_measurement_noise_factor=float(
            observation.pressure_measurement_noise_factor
        ),
        power_measurement_relative_noise=float(
            observation.power_measurement_relative_noise
        ),
        flow_measurement_relative_noise=float(
            observation.flow_measurement_relative_noise
        ),
        pressure_measurement_relative_noise=float(
            observation.pressure_measurement_relative_noise
        ),
        pressure_sensor_bias_factor=float(
            observation.pressure_sensor_bias_factor
        ),
        pressure_sensor_bias_fraction=float(
            observation.pressure_sensor_bias_fraction
        ),
        integration_success=bool(
            state.integration_success
        ),
        converged=bool(
            state.converged
        ),
        max_relative_rate_s=float(
            state.max_relative_rate_s
        ),
        neutral_particle_balance=float(
            state.neutral_particle_balance
        ),
        ion_particle_balance=float(
            state.ion_particle_balance
        ),
        electron_energy_balance=float(
            state.electron_energy_balance
        ),
        total_particle_identity=float(
            state.total_particle_identity
        ),
        max_balance_residual=float(
            state.max_balance_residual
        ),
        physical_state_valid=bool(
            state.physical_state_valid
        ),
        balance_valid=bool(
            state.balance_valid
        ),
        domain_status=(
            domain_status
        ),
        model_validity_status=(
            state.model_validity_status
        ),
        qualification_valid=bool(
            state.qualification_valid
        ),
        is_ml_eligible=(
            is_ml_eligible
        ),
        error_message=(
            state.error_message
        ),
    )
