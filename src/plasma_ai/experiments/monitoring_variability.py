"""Normal latent process variability for Phase-3 monitoring data."""

from __future__ import annotations

from dataclasses import dataclass

from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
)
from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
)
from plasma_ai.experiments.monitoring_random import (
    deterministic_relative_factor,
)


@dataclass(frozen=True)
class ProcessVariabilitySample:
    """Normal process-variability factors for one monitoring step."""

    power_coupling_factor: float
    flow_delivery_factor: float
    pumping_effectiveness_factor: float

    power_relative_deviation: float
    flow_relative_deviation: float
    pumping_relative_deviation: float


def sample_process_variability(
    config: MonitoringDatasetConfig,
    plan: MonitoringEpisodePlan,
    step_index: int,
) -> ProcessVariabilitySample:
    """Generate reproducible normal process variability for one step.

    These factors represent normal synthetic process variability only.
    Fault perturbations are deliberately applied in a separate layer.
    """
    if (
        step_index < 0
        or step_index >= config.steps_per_episode
    ):
        raise ValueError(
            "step_index lies outside the monitoring episode."
        )

    variability = (
        config.process_variability
    )

    seed = (
        config.seeds.process_variability
    )

    power_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=step_index,
        channel="process_power_coupling",
        relative_sigma=(
            variability.absorbed_power_relative_sigma
        ),
        truncation_sigma=(
            variability.truncation_sigma
        ),
    )

    flow_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=step_index,
        channel="process_flow_delivery",
        relative_sigma=(
            variability.flow_relative_sigma
        ),
        truncation_sigma=(
            variability.truncation_sigma
        ),
    )

    pumping_factor = deterministic_relative_factor(
        base_seed=seed,
        episode_id=plan.episode_id,
        step_index=step_index,
        channel="process_pumping_effectiveness",
        relative_sigma=(
            variability.pumping_relative_sigma
        ),
        truncation_sigma=(
            variability.truncation_sigma
        ),
    )

    return ProcessVariabilitySample(
        power_coupling_factor=float(
            power_factor
        ),
        flow_delivery_factor=float(
            flow_factor
        ),
        pumping_effectiveness_factor=float(
            pumping_factor
        ),
        power_relative_deviation=float(
            power_factor
            - 1.0
        ),
        flow_relative_deviation=float(
            flow_factor
            - 1.0
        ),
        pumping_relative_deviation=float(
            pumping_factor
            - 1.0
        ),
    )
