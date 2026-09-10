"""Deterministic episode planning for Phase-3 monitoring data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from plasma_ai.experiments.monitoring_config import (
    FAULT_FAMILIES,
    SEVERITY_ORDER,
    MonitoringDatasetConfig,
    SPLIT_ORDER,
)


@dataclass(frozen=True)
class MonitoringEpisodePlan:
    """Frozen plan for one 64-step quasi-steady monitoring episode."""

    episode_id: str
    split: str
    split_episode_index: int
    recipe_design_index: int
    recipe_design_seed: int

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float
    nominal_flow_sccm: float

    fault_present: bool
    fault_domain: str
    fault_family: str
    fault_profile: str
    fault_severity: str
    fault_direction: str
    fault_magnitude_fraction: float
    fault_onset_step: int | None


def _shuffled(
    values: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Return a deterministic shuffled copy."""
    order = rng.permutation(
        len(values)
    )

    return [
        values[int(index)]
        for index in order
    ]


def _fault_family_pool(
    config: MonitoringDatasetConfig,
    split_name: str,
) -> list[str]:
    """Return the exact frozen fault-family allocation for a split."""
    split = config.splits[
        split_name
    ]

    pool = (
        ["none"] * split.normal
        + ["power_coupling"] * split.power_coupling
        + ["flow_delivery"] * split.flow_delivery
        + ["pumping_effectiveness"]
        * split.pumping_effectiveness
        + ["pressure_sensor_bias"]
        * split.pressure_sensor_bias
    )

    if len(pool) != split.episodes:
        raise RuntimeError(
            "Fault-family allocation does not match split size."
        )

    return pool


def _severity_pool(
    count: int,
) -> list[str]:
    """Return a severity-balanced pool for a frozen family count."""
    if count == 6:
        return [
            "mild",
            "mild",
            "moderate",
            "moderate",
            "severe",
            "severe",
        ]

    if count == 3:
        return [
            "mild",
            "moderate",
            "severe",
        ]

    raise ValueError(
        "Frozen Phase-3E fault-family counts must be 3 or 6."
    )


def _profile_pool(
    split_name: str,
    count: int,
) -> list[str]:
    """Return a profile-balanced pool.

    Across train + validation + test, each fault family receives
    exactly six step and six drift episodes.
    """
    if split_name == "train" and count == 6:
        return [
            "step",
            "step",
            "step",
            "drift",
            "drift",
            "drift",
        ]

    if split_name == "validation" and count == 3:
        return [
            "step",
            "step",
            "drift",
        ]

    if split_name == "test" and count == 3:
        return [
            "step",
            "drift",
            "drift",
        ]

    raise ValueError(
        "Unexpected split/count combination for profile allocation."
    )


def _direction_pool(
    family: str,
    split_name: str,
    count: int,
) -> list[str]:
    """Return the frozen direction allocation for one fault family."""
    if family == "power_coupling":
        return [
            "negative"
            for _ in range(count)
        ]

    if split_name == "train" and count == 6:
        return [
            "negative",
            "negative",
            "negative",
            "positive",
            "positive",
            "positive",
        ]

    if split_name == "validation" and count == 3:
        return [
            "negative",
            "negative",
            "positive",
        ]

    if split_name == "test" and count == 3:
        return [
            "negative",
            "positive",
            "positive",
        ]

    raise ValueError(
        "Unexpected split/count combination for direction allocation."
    )


def _nominal_recipe_design(
    config: MonitoringDatasetConfig,
) -> np.ndarray:
    """Return the reproducible 64-point inner-envelope Sobol design."""
    sampler = qmc.Sobol(
        d=2,
        scramble=True,
        seed=config.seeds.recipe_design,
    )

    unit_design = sampler.random_base2(
        m=6,
    )

    return qmc.scale(
        unit_design,
        l_bounds=[
            config.nominal_recipe.absorbed_power_min_W,
            config.nominal_recipe.target_pressure_min_mTorr,
        ],
        u_bounds=[
            config.nominal_recipe.absorbed_power_max_W,
            config.nominal_recipe.target_pressure_max_mTorr,
        ],
    )


def generate_monitoring_episode_plan(
    config: MonitoringDatasetConfig,
) -> list[MonitoringEpisodePlan]:
    """Generate the complete deterministic 64-episode monitoring plan."""
    recipe_design = _nominal_recipe_design(
        config
    )

    rng = np.random.default_rng(
        config.seeds.episode_fault_plan
    )

    plans: list[MonitoringEpisodePlan] = []

    recipe_index = 0

    for split_name in SPLIT_ORDER:
        split = config.splits[
            split_name
        ]

        family_pool = _shuffled(
            _fault_family_pool(
                config,
                split_name,
            ),
            rng,
        )

        family_attributes: dict[
            str,
            dict[str, list[str]],
        ] = {}

        for family in FAULT_FAMILIES:
            count = getattr(
                split,
                family,
            )

            family_attributes[
                family
            ] = {
                "severity": _shuffled(
                    _severity_pool(
                        count
                    ),
                    rng,
                ),
                "profile": _shuffled(
                    _profile_pool(
                        split_name,
                        count,
                    ),
                    rng,
                ),
                "direction": _shuffled(
                    _direction_pool(
                        family,
                        split_name,
                        count,
                    ),
                    rng,
                ),
            }

        family_offsets = defaultdict(
            int
        )

        for split_episode_index, family in enumerate(
            family_pool,
            start=1,
        ):
            power = float(
                recipe_design[
                    recipe_index,
                    0,
                ]
            )

            pressure = float(
                recipe_design[
                    recipe_index,
                    1,
                ]
            )

            recipe_design_index = (
                recipe_index
                + 1
            )

            episode_id = (
                f"monitor_{split_name}_"
                f"{split_episode_index:03d}"
            )

            if family == "none":
                plan = MonitoringEpisodePlan(
                    episode_id=episode_id,
                    split=split_name,
                    split_episode_index=(
                        split_episode_index
                    ),
                    recipe_design_index=(
                        recipe_design_index
                    ),
                    recipe_design_seed=(
                        config.seeds.recipe_design
                    ),
                    nominal_absorbed_power_W=power,
                    target_pressure_mTorr=pressure,
                    nominal_flow_sccm=(
                        config.nominal_recipe.flow_sccm
                    ),
                    fault_present=False,
                    fault_domain="none",
                    fault_family="none",
                    fault_profile="none",
                    fault_severity="none",
                    fault_direction="none",
                    fault_magnitude_fraction=0.0,
                    fault_onset_step=None,
                )
            else:
                offset = family_offsets[
                    family
                ]

                attributes = family_attributes[
                    family
                ]

                severity = attributes[
                    "severity"
                ][offset]

                profile = attributes[
                    "profile"
                ][offset]

                direction = attributes[
                    "direction"
                ][offset]

                family_offsets[
                    family
                ] += 1

                magnitude = (
                    config.faults[
                        family
                    ].magnitude_for_severity(
                        severity
                    )
                )

                onset = int(
                    rng.integers(
                        config.fault_onset.minimum_step,
                        config.fault_onset.maximum_step
                        + 1,
                    )
                )

                plan = MonitoringEpisodePlan(
                    episode_id=episode_id,
                    split=split_name,
                    split_episode_index=(
                        split_episode_index
                    ),
                    recipe_design_index=(
                        recipe_design_index
                    ),
                    recipe_design_seed=(
                        config.seeds.recipe_design
                    ),
                    nominal_absorbed_power_W=power,
                    target_pressure_mTorr=pressure,
                    nominal_flow_sccm=(
                        config.nominal_recipe.flow_sccm
                    ),
                    fault_present=True,
                    fault_domain=(
                        config.faults[
                            family
                        ].domain
                    ),
                    fault_family=family,
                    fault_profile=profile,
                    fault_severity=severity,
                    fault_direction=direction,
                    fault_magnitude_fraction=float(
                        magnitude
                    ),
                    fault_onset_step=onset,
                )

            plans.append(
                plan
            )

            recipe_index += 1

    if recipe_index != config.total_episodes:
        raise RuntimeError(
            "Monitoring recipe allocation did not consume "
            "exactly 64 recipes."
        )

    return plans
