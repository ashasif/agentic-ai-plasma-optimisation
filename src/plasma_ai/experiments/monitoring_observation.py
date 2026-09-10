"""Synthetic sensor-observation layer for Phase-3 monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import math

from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
)
from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
)
from plasma_ai.experiments.monitoring_process import (
    MonitoringProcessState,
)
from plasma_ai.experiments.monitoring_random import (
    deterministic_relative_factor,
)


@dataclass(frozen=True)
class MonitoringObservation:
    """Synthetic measured quantities for one monitoring step."""

    monitoring_id: str
    episode_id: str
    split: str
    step_index: int

    power_measurement_noise_factor: float
    flow_measurement_noise_factor: float
    pressure_measurement_noise_factor: float

    power_measurement_relative_noise: float
    flow_measurement_relative_noise: float
    pressure_measurement_relative_noise: float

    pressure_sensor_bias_factor: float
    pressure_sensor_bias_fraction: float

    measured_absorbed_power_W: float
    measured_flow_sccm: float
    measured_pressure_mTorr: float


def build_monitoring_observation(
    config: MonitoringDatasetConfig,
    plan: MonitoringEpisodePlan,
    state: MonitoringProcessState,
) -> MonitoringObservation:
    """Create reproducible noisy sensor observations.

    Measurement noise is applied after the latent physical process has
    been solved. Pressure-sensor bias is an observation-layer effect
    only and therefore does not modify the plasma state.
    """
    if state.episode_id != plan.episode_id:
        raise ValueError(
            "Process state and episode plan have different episode IDs."
        )

    if state.split != plan.split:
        raise ValueError(
            "Process state and episode plan have different splits."
        )

    if (
        state.step_index < 0
        or state.step_index >= config.steps_per_episode
    ):
        raise ValueError(
            "Process-state step index lies outside the episode."
        )

    noise = config.measurement_noise
    seed = config.seeds.measurement_noise

    power_noise_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=state.step_index,
        channel="measurement_absorbed_power",
        relative_sigma=(
            noise.absorbed_power_relative_sigma
        ),
        truncation_sigma=(
            noise.truncation_sigma
        ),
    )

    flow_noise_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=state.step_index,
        channel="measurement_flow",
        relative_sigma=(
            noise.flow_relative_sigma
        ),
        truncation_sigma=(
            noise.truncation_sigma
        ),
    )

    pressure_noise_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=state.step_index,
        channel="measurement_pressure",
        relative_sigma=(
            noise.pressure_relative_sigma
        ),
        truncation_sigma=(
            noise.truncation_sigma
        ),
    )

    pressure_sensor_bias_factor = 1.0

    if plan.fault_present and plan.fault_domain == "sensor":
        if plan.fault_family != "pressure_sensor_bias":
            raise ValueError(
                "Unknown sensor fault family: "
                f"{plan.fault_family!r}"
            )

        pressure_sensor_bias_factor = float(
            state.fault_factor
        )

    measured_absorbed_power_W = (
        state.true_absorbed_power_W
        * power_noise_factor
    )

    measured_flow_sccm = (
        state.true_flow_sccm
        * flow_noise_factor
    )

    measured_pressure_mTorr = (
        state.true_pressure_mTorr
        * pressure_sensor_bias_factor
        * pressure_noise_factor
    )

    # Successful process solves must produce finite positive
    # synthetic observations.  Integration failures are preserved
    # explicitly instead of causing dataset generation to abort; their
    # unavailable measured values therefore remain NaN and the final
    # monitoring row is marked ineligible for ML.
    if state.integration_success:
        for name, value in {
            "measured_absorbed_power_W": (
                measured_absorbed_power_W
            ),
            "measured_flow_sccm": (
                measured_flow_sccm
            ),
            "measured_pressure_mTorr": (
                measured_pressure_mTorr
            ),
        }.items():
            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must remain finite and positive."
                )

    return MonitoringObservation(
        monitoring_id=state.monitoring_id,
        episode_id=state.episode_id,
        split=state.split,
        step_index=state.step_index,
        power_measurement_noise_factor=float(
            power_noise_factor
        ),
        flow_measurement_noise_factor=float(
            flow_noise_factor
        ),
        pressure_measurement_noise_factor=float(
            pressure_noise_factor
        ),
        power_measurement_relative_noise=float(
            power_noise_factor
            - 1.0
        ),
        flow_measurement_relative_noise=float(
            flow_noise_factor
            - 1.0
        ),
        pressure_measurement_relative_noise=float(
            pressure_noise_factor
            - 1.0
        ),
        pressure_sensor_bias_factor=float(
            pressure_sensor_bias_factor
        ),
        pressure_sensor_bias_fraction=float(
            pressure_sensor_bias_factor
            - 1.0
        ),
        measured_absorbed_power_W=float(
            measured_absorbed_power_W
        ),
        measured_flow_sccm=float(
            measured_flow_sccm
        ),
        measured_pressure_mTorr=float(
            measured_pressure_mTorr
        ),
    )
