"""True-process and quasi-steady plasma layer for Phase-3 monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
)
from plasma_ai.experiments.monitoring_faults import (
    fault_progress,
    fault_started,
    multiplicative_fault_factor,
    signed_fault_fraction,
)
from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
)
from plasma_ai.experiments.monitoring_variability import (
    sample_process_variability,
)
from plasma_ai.physics.energy_losses import (
    electron_energy_density_j_m3,
)
from plasma_ai.physics.gas_flow import (
    effective_pumping_speed_m3_s,
    mtorr_to_pa,
    neutral_density_m3_from_pressure_pa,
    pa_to_mtorr,
    pressure_pa_from_neutral_density_m3,
)
from plasma_ai.physics.global_model import (
    GlobalModelParameters,
    global_model_terms,
)
from plasma_ai.physics.solver import (
    solve_global_model,
)
from plasma_ai.physics.validation import (
    steady_state_balance_residuals,
)


@dataclass(frozen=True)
class TrueProcessInputs:
    """Latent true operating inputs for one monitoring step."""

    nominal_pumping_speed_m3_s: float

    normal_power_coupling_factor: float
    normal_flow_delivery_factor: float
    normal_pumping_effectiveness_factor: float

    fault_started: bool
    fault_progress: float
    signed_fault_fraction: float
    fault_factor: float

    applied_power_factor: float
    applied_flow_factor: float
    applied_pumping_factor: float

    true_absorbed_power_W: float
    true_flow_sccm: float
    true_pumping_speed_m3_s: float


@dataclass(frozen=True)
class MonitoringProcessState:
    """Quasi-steady latent plasma/process truth for one monitoring step."""

    monitoring_id: str
    episode_id: str
    split: str
    step_index: int

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float
    nominal_flow_sccm: float
    nominal_pumping_speed_m3_s: float

    normal_power_coupling_factor: float
    normal_flow_delivery_factor: float
    normal_pumping_effectiveness_factor: float

    fault_started: bool
    fault_progress: float
    signed_fault_fraction: float
    fault_factor: float

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
    model_validity_status: str
    qualification_valid: bool

    error_message: str


def _monitoring_id(
    plan: MonitoringEpisodePlan,
    step_index: int,
) -> str:
    """Return a stable row identifier."""
    return (
        f"{plan.episode_id}_"
        f"step_{step_index:03d}"
    )


def build_true_process_inputs(
    config: MonitoringDatasetConfig,
    plan: MonitoringEpisodePlan,
    step_index: int,
) -> TrueProcessInputs:
    """Combine normal variability and process faults.

    Sensor faults deliberately do not modify the latent physical process.
    """
    if (
        step_index < 0
        or step_index >= config.steps_per_episode
    ):
        raise ValueError(
            "step_index lies outside the monitoring episode."
        )

    target_pressure_pa = mtorr_to_pa(
        plan.target_pressure_mTorr
    )

    nominal_pumping_speed_m3_s = (
        effective_pumping_speed_m3_s(
            plan.nominal_flow_sccm,
            target_pressure_pa,
            config.fixed_model_parameters.gas_temperature_K,
        )
    )

    variability = sample_process_variability(
        config,
        plan,
        step_index,
    )

    started = fault_started(
        plan,
        step_index,
        config.steps_per_episode,
    )

    progress = fault_progress(
        plan,
        step_index,
        config.steps_per_episode,
    )

    signed_fraction = signed_fault_fraction(
        plan,
        step_index,
        config.steps_per_episode,
    )

    fault_factor = multiplicative_fault_factor(
        plan,
        step_index,
        config.steps_per_episode,
    )

    power_factor = (
        variability.power_coupling_factor
    )

    flow_factor = (
        variability.flow_delivery_factor
    )

    pumping_factor = (
        variability.pumping_effectiveness_factor
    )

    if plan.fault_present:
        if plan.fault_domain == "process":
            if plan.fault_family == "power_coupling":
                power_factor *= fault_factor

            elif plan.fault_family == "flow_delivery":
                flow_factor *= fault_factor

            elif plan.fault_family == "pumping_effectiveness":
                pumping_factor *= fault_factor

            else:
                raise ValueError(
                    "Unknown process fault family: "
                    f"{plan.fault_family!r}"
                )

        elif plan.fault_domain == "sensor":
            if plan.fault_family != "pressure_sensor_bias":
                raise ValueError(
                    "Unknown sensor fault family: "
                    f"{plan.fault_family!r}"
                )

            # Sensor bias changes only the later observation layer.

        else:
            raise ValueError(
                "Faulty episode must have process or sensor domain."
            )

    true_absorbed_power_W = (
        plan.nominal_absorbed_power_W
        * power_factor
    )

    true_flow_sccm = (
        plan.nominal_flow_sccm
        * flow_factor
    )

    true_pumping_speed_m3_s = (
        nominal_pumping_speed_m3_s
        * pumping_factor
    )

    for name, value in {
        "true_absorbed_power_W": true_absorbed_power_W,
        "true_flow_sccm": true_flow_sccm,
        "true_pumping_speed_m3_s": true_pumping_speed_m3_s,
    }.items():
        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{name} must remain finite and positive."
            )

    return TrueProcessInputs(
        nominal_pumping_speed_m3_s=float(
            nominal_pumping_speed_m3_s
        ),
        normal_power_coupling_factor=float(
            variability.power_coupling_factor
        ),
        normal_flow_delivery_factor=float(
            variability.flow_delivery_factor
        ),
        normal_pumping_effectiveness_factor=float(
            variability.pumping_effectiveness_factor
        ),
        fault_started=bool(
            started
        ),
        fault_progress=float(
            progress
        ),
        signed_fault_fraction=float(
            signed_fraction
        ),
        fault_factor=float(
            fault_factor
        ),
        applied_power_factor=float(
            power_factor
        ),
        applied_flow_factor=float(
            flow_factor
        ),
        applied_pumping_factor=float(
            pumping_factor
        ),
        true_absorbed_power_W=float(
            true_absorbed_power_W
        ),
        true_flow_sccm=float(
            true_flow_sccm
        ),
        true_pumping_speed_m3_s=float(
            true_pumping_speed_m3_s
        ),
    )


def _failed_process_state(
    plan: MonitoringEpisodePlan,
    step_index: int,
    inputs: TrueProcessInputs,
    error_message: str,
) -> MonitoringProcessState:
    """Preserve an integration failure instead of resampling it."""
    nan = float("nan")

    return MonitoringProcessState(
        monitoring_id=_monitoring_id(
            plan,
            step_index,
        ),
        episode_id=plan.episode_id,
        split=plan.split,
        step_index=step_index,
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
            inputs.nominal_pumping_speed_m3_s
        ),
        normal_power_coupling_factor=(
            inputs.normal_power_coupling_factor
        ),
        normal_flow_delivery_factor=(
            inputs.normal_flow_delivery_factor
        ),
        normal_pumping_effectiveness_factor=(
            inputs.normal_pumping_effectiveness_factor
        ),
        fault_started=inputs.fault_started,
        fault_progress=inputs.fault_progress,
        signed_fault_fraction=(
            inputs.signed_fault_fraction
        ),
        fault_factor=inputs.fault_factor,
        applied_power_factor=(
            inputs.applied_power_factor
        ),
        applied_flow_factor=(
            inputs.applied_flow_factor
        ),
        applied_pumping_factor=(
            inputs.applied_pumping_factor
        ),
        true_absorbed_power_W=(
            inputs.true_absorbed_power_W
        ),
        true_flow_sccm=(
            inputs.true_flow_sccm
        ),
        true_pumping_speed_m3_s=(
            inputs.true_pumping_speed_m3_s
        ),
        true_neutral_density_m3=nan,
        true_ion_density_m3=nan,
        true_electron_density_m3=nan,
        true_electron_energy_density_J_m3=nan,
        true_electron_temperature_eV=nan,
        true_pressure_mTorr=nan,
        true_pressure_relative_deviation_from_target=nan,
        ionization_rate_coefficient_m3_s=nan,
        ionization_density_rate_m3_s=nan,
        charged_wall_loss_rate_s=nan,
        charged_wall_loss_density_rate_m3_s=nan,
        ionization_fraction=nan,
        residence_time_proxy_s=nan,
        collisional_power_loss_W=nan,
        wall_power_loss_W=nan,
        integration_success=False,
        converged=False,
        max_relative_rate_s=nan,
        neutral_particle_balance=nan,
        ion_particle_balance=nan,
        electron_energy_balance=nan,
        total_particle_identity=nan,
        max_balance_residual=nan,
        physical_state_valid=False,
        balance_valid=False,
        model_validity_status="integration_failed",
        qualification_valid=False,
        error_message=str(
            error_message
        ),
    )


def simulate_monitoring_process_step(
    config: MonitoringDatasetConfig,
    plan: MonitoringEpisodePlan,
    step_index: int,
) -> MonitoringProcessState:
    """Solve one independently cold-started quasi-steady monitoring step."""
    inputs = build_true_process_inputs(
        config,
        plan,
        step_index,
    )

    fixed = (
        config.fixed_model_parameters
    )

    # Physics-informed independent cold-start policy.
    #
    # Each monitoring row is still solved independently and does not
    # inherit state from the preceding episode step.  However, when
    # true delivered flow and true pumping differ from their nominal
    # values, the initial neutral density is placed near the pressure
    # implied by that row's Q/S ratio.  This avoids interpreting slow
    # neutral-inventory relaxation from the commanded pressure as a
    # failure of the quasi-steady operating point.
    flow_ratio = (
        inputs.true_flow_sccm
        / plan.nominal_flow_sccm
    )

    pumping_ratio = (
        inputs.true_pumping_speed_m3_s
        / inputs.nominal_pumping_speed_m3_s
    )

    inferred_initial_pressure_mTorr = (
        plan.target_pressure_mTorr
        * flow_ratio
        / pumping_ratio
    )

    initial_neutral_density_m3 = (
        neutral_density_m3_from_pressure_pa(
            mtorr_to_pa(
                inferred_initial_pressure_mTorr
            ),
            fixed.gas_temperature_K,
        )
    )

    initial_energy_density_J_m3 = (
        electron_energy_density_j_m3(
            fixed.initial_electron_density_m3,
            fixed.initial_electron_temperature_eV,
        )
    )

    initial_state = np.array(
        [
            initial_neutral_density_m3,
            fixed.initial_electron_density_m3,
            initial_energy_density_J_m3,
        ],
        dtype=float,
    )

    params = GlobalModelParameters(
        radius_m=fixed.radius_m,
        length_m=fixed.length_m,
        absorbed_power_W=(
            inputs.true_absorbed_power_W
        ),
        flow_sccm=(
            inputs.true_flow_sccm
        ),
        pumping_speed_m3_s=(
            inputs.true_pumping_speed_m3_s
        ),
        gas_temperature_K=(
            fixed.gas_temperature_K
        ),
        ion_neutral_cross_section_m2=(
            fixed.ion_neutral_cross_section_m2
        ),
    )

    try:
        solution = solve_global_model(
            initial_state,
            params,
            end_time_s=(
                fixed.solver_end_time_s
            ),
            convergence_threshold_s=(
                fixed.convergence_threshold_s
            ),
        )
    except RuntimeError as exc:
        return _failed_process_state(
            plan,
            step_index,
            inputs,
            str(exc),
        )

    final_state = solution.final_state

    n0 = float(
        final_state[0]
    )

    ni = float(
        final_state[1]
    )

    pe = float(
        final_state[2]
    )

    # Frozen electropositive quasineutral bulk assumption.
    ne = ni

    te = float(
        solution.electron_temperature_eV[-1]
    )

    true_pressure_pa = (
        pressure_pa_from_neutral_density_m3(
            n0,
            fixed.gas_temperature_K,
        )
    )

    true_pressure_mTorr = float(
        pa_to_mtorr(
            true_pressure_pa
        )
    )

    pressure_relative_deviation = (
        (
            true_pressure_mTorr
            - plan.target_pressure_mTorr
        )
        / plan.target_pressure_mTorr
    )

    terms = global_model_terms(
        final_state,
        params,
    )

    residuals = steady_state_balance_residuals(
        final_state,
        params,
    )

    volume_m3 = float(
        terms["volume_m3"]
    )

    ionization_rate_coefficient_m3_s = float(
        terms[
            "ionization_rate_coefficient_m3_s"
        ]
    )

    ionization_density_rate_m3_s = float(
        terms[
            "ionization_density_rate_m3_s"
        ]
    )

    charged_wall_loss_rate_s = float(
        terms[
            "charged_wall_loss_rate_s"
        ]
    )

    charged_wall_loss_density_rate_m3_s = float(
        terms[
            "charged_wall_loss_density_rate_m3_s"
        ]
    )

    collisional_power_loss_W = float(
        terms[
            "collisional_power_loss_density_W_m3"
        ]
        * volume_m3
    )

    wall_power_loss_W = float(
        terms[
            "wall_power_loss_density_W_m3"
        ]
        * volume_m3
    )

    ionization_fraction = float(
        ni
        / (
            n0
            + ni
        )
    )

    # Global V/S engineering proxy only.
    residence_time_proxy_s = float(
        volume_m3
        / inputs.true_pumping_speed_m3_s
    )

    values_requiring_finiteness = (
        n0,
        ni,
        ne,
        pe,
        te,
        true_pressure_mTorr,
        float(
            solution.max_relative_rate_s
        ),
        ionization_rate_coefficient_m3_s,
        ionization_density_rate_m3_s,
        charged_wall_loss_rate_s,
        charged_wall_loss_density_rate_m3_s,
        ionization_fraction,
        residence_time_proxy_s,
        collisional_power_loss_W,
        wall_power_loss_W,
    )

    finite_values = all(
        math.isfinite(value)
        for value in values_requiring_finiteness
    )

    physical_state_valid = bool(
        finite_values
        and n0 > 0.0
        and 0.0 < ne < n0
        and pe > 0.0
        and (
            fixed.minimum_reasonable_electron_temperature_eV
            < te
            < fixed.maximum_reasonable_electron_temperature_eV
        )
        and true_pressure_mTorr > 0.0
        and inputs.true_pumping_speed_m3_s > 0.0
        and ionization_rate_coefficient_m3_s >= 0.0
        and ionization_density_rate_m3_s >= 0.0
        and charged_wall_loss_rate_s >= 0.0
        and charged_wall_loss_density_rate_m3_s >= 0.0
        and 0.0 < ionization_fraction < 1.0
        and residence_time_proxy_s > 0.0
        and collisional_power_loss_W >= 0.0
        and wall_power_loss_W >= 0.0
    )

    balance_valid = bool(
        residuals[
            "neutral_particle_balance"
        ]
        < fixed.balance_residual_limit
        and residuals[
            "ion_particle_balance"
        ]
        < fixed.balance_residual_limit
        and residuals[
            "electron_energy_balance"
        ]
        < fixed.balance_residual_limit
        and residuals[
            "max_balance_residual"
        ]
        < fixed.balance_residual_limit
        and residuals[
            "total_particle_identity"
        ]
        < fixed.total_particle_identity_limit
    )

    if not solution.converged:
        model_validity_status = (
            "nonconverged"
        )
    elif not physical_state_valid:
        model_validity_status = (
            "invalid_physical_state"
        )
    elif not balance_valid:
        model_validity_status = (
            "invalid_balance"
        )
    else:
        model_validity_status = (
            "valid"
        )

    qualification_valid = bool(
        bool(
            solution.success
        )
        and bool(
            solution.converged
        )
        and physical_state_valid
        and balance_valid
    )

    return MonitoringProcessState(
        monitoring_id=_monitoring_id(
            plan,
            step_index,
        ),
        episode_id=plan.episode_id,
        split=plan.split,
        step_index=step_index,
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
            inputs.nominal_pumping_speed_m3_s
        ),
        normal_power_coupling_factor=(
            inputs.normal_power_coupling_factor
        ),
        normal_flow_delivery_factor=(
            inputs.normal_flow_delivery_factor
        ),
        normal_pumping_effectiveness_factor=(
            inputs.normal_pumping_effectiveness_factor
        ),
        fault_started=inputs.fault_started,
        fault_progress=inputs.fault_progress,
        signed_fault_fraction=(
            inputs.signed_fault_fraction
        ),
        fault_factor=inputs.fault_factor,
        applied_power_factor=(
            inputs.applied_power_factor
        ),
        applied_flow_factor=(
            inputs.applied_flow_factor
        ),
        applied_pumping_factor=(
            inputs.applied_pumping_factor
        ),
        true_absorbed_power_W=(
            inputs.true_absorbed_power_W
        ),
        true_flow_sccm=(
            inputs.true_flow_sccm
        ),
        true_pumping_speed_m3_s=(
            inputs.true_pumping_speed_m3_s
        ),
        true_neutral_density_m3=n0,
        true_ion_density_m3=ni,
        true_electron_density_m3=ne,
        true_electron_energy_density_J_m3=pe,
        true_electron_temperature_eV=te,
        true_pressure_mTorr=(
            true_pressure_mTorr
        ),
        true_pressure_relative_deviation_from_target=float(
            pressure_relative_deviation
        ),
        ionization_rate_coefficient_m3_s=(
            ionization_rate_coefficient_m3_s
        ),
        ionization_density_rate_m3_s=(
            ionization_density_rate_m3_s
        ),
        charged_wall_loss_rate_s=(
            charged_wall_loss_rate_s
        ),
        charged_wall_loss_density_rate_m3_s=(
            charged_wall_loss_density_rate_m3_s
        ),
        ionization_fraction=(
            ionization_fraction
        ),
        residence_time_proxy_s=(
            residence_time_proxy_s
        ),
        collisional_power_loss_W=(
            collisional_power_loss_W
        ),
        wall_power_loss_W=(
            wall_power_loss_W
        ),
        integration_success=bool(
            solution.success
        ),
        converged=bool(
            solution.converged
        ),
        max_relative_rate_s=float(
            solution.max_relative_rate_s
        ),
        neutral_particle_balance=float(
            residuals[
                "neutral_particle_balance"
            ]
        ),
        ion_particle_balance=float(
            residuals[
                "ion_particle_balance"
            ]
        ),
        electron_energy_balance=float(
            residuals[
                "electron_energy_balance"
            ]
        ),
        total_particle_identity=float(
            residuals[
                "total_particle_identity"
            ]
        ),
        max_balance_residual=float(
            residuals[
                "max_balance_residual"
            ]
        ),
        physical_state_valid=(
            physical_state_valid
        ),
        balance_valid=(
            balance_valid
        ),
        model_validity_status=(
            model_validity_status
        ),
        qualification_valid=(
            qualification_valid
        ),
        error_message="",
    )
