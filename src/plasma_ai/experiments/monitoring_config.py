"""Configuration for the Phase-3 synthetic monitoring environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


SPLIT_ORDER = (
    "train",
    "validation",
    "test",
)

FAULT_FAMILIES = (
    "power_coupling",
    "flow_delivery",
    "pumping_effectiveness",
    "pressure_sensor_bias",
)

SEVERITY_ORDER = (
    "mild",
    "moderate",
    "severe",
)

ALLOWED_PROFILES = (
    "step",
    "drift",
)


@dataclass(frozen=True)
class MonitoringSplitConfig:
    """Episode allocation for one dataset split."""

    episodes: int
    normal: int
    power_coupling: int
    flow_delivery: int
    pumping_effectiveness: int
    pressure_sensor_bias: int

    @property
    def faulty_episodes(self) -> int:
        """Return the number of faulty episodes."""
        return (
            self.power_coupling
            + self.flow_delivery
            + self.pumping_effectiveness
            + self.pressure_sensor_bias
        )

    @property
    def allocated_episodes(self) -> int:
        """Return normal plus all fault-family allocations."""
        return (
            self.normal
            + self.faulty_episodes
        )


@dataclass(frozen=True)
class NominalRecipeConfig:
    """Inner nominal recipe envelope for monitoring episodes."""

    absorbed_power_min_W: float
    absorbed_power_max_W: float
    target_pressure_min_mTorr: float
    target_pressure_max_mTorr: float
    flow_sccm: float


@dataclass(frozen=True)
class FixedModelConfig:
    """Frozen reduced-order model settings."""

    gas_temperature_K: float
    ion_neutral_cross_section_m2: float
    radius_m: float
    length_m: float

    initial_electron_density_m3: float
    initial_electron_temperature_eV: float

    solver_end_time_s: float
    convergence_threshold_s: float
    balance_residual_limit: float
    total_particle_identity_limit: float

    minimum_reasonable_electron_temperature_eV: float
    maximum_reasonable_electron_temperature_eV: float


@dataclass(frozen=True)
class ProcessVariabilityConfig:
    """Synthetic latent run-to-run variability."""

    absorbed_power_relative_sigma: float
    flow_relative_sigma: float
    pumping_relative_sigma: float
    truncation_sigma: float


@dataclass(frozen=True)
class MeasurementNoiseConfig:
    """Synthetic sensor-noise assumptions."""

    absorbed_power_relative_sigma: float
    flow_relative_sigma: float
    pressure_relative_sigma: float
    truncation_sigma: float


@dataclass(frozen=True)
class FaultOnsetConfig:
    """Allowed discrete fault-onset range."""

    minimum_step: int
    maximum_step: int


@dataclass(frozen=True)
class FaultConfig:
    """Definition of one synthetic single-fault family."""

    domain: str
    directions: tuple[str, ...]
    mild_fraction: float
    moderate_fraction: float
    severe_fraction: float

    def magnitude_for_severity(
        self,
        severity: str,
    ) -> float:
        """Return the positive fractional magnitude for one severity."""
        if severity == "mild":
            return self.mild_fraction

        if severity == "moderate":
            return self.moderate_fraction

        if severity == "severe":
            return self.severe_fraction

        raise ValueError(
            f"Unknown fault severity: {severity!r}"
        )


@dataclass(frozen=True)
class MonitoringSeeds:
    """Independent reproducibility seeds."""

    recipe_design: int
    episode_fault_plan: int
    process_variability: int
    measurement_noise: int


@dataclass(frozen=True)
class MonitoringDatasetConfig:
    """Complete frozen monitoring-environment configuration."""

    dataset_name: str
    dataset_version: str
    interpretation: str
    steps_per_episode: int

    nominal_recipe: NominalRecipeConfig
    fixed_model_parameters: FixedModelConfig

    splits: dict[str, MonitoringSplitConfig]

    process_variability: ProcessVariabilityConfig
    measurement_noise: MeasurementNoiseConfig

    fault_onset: FaultOnsetConfig
    fault_profiles: tuple[str, ...]

    faults: dict[str, FaultConfig]

    seeds: MonitoringSeeds

    @property
    def total_episodes(self) -> int:
        """Return configured episode count."""
        return sum(
            self.splits[name].episodes
            for name in SPLIT_ORDER
        )

    @property
    def total_rows(self) -> int:
        """Return configured monitoring-row count."""
        return (
            self.total_episodes
            * self.steps_per_episode
        )


def _positive_finite(
    value: float,
    name: str,
) -> None:
    """Require a positive finite scalar."""
    if (
        not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and strictly positive."
        )


def _validate_relative_sigma(
    value: float,
    name: str,
) -> None:
    """Require a finite non-negative fractional sigma."""
    if (
        not math.isfinite(value)
        or value < 0.0
        or value >= 1.0
    ):
        raise ValueError(
            f"{name} must be finite and lie in [0, 1)."
        )


def validate_monitoring_config(
    config: MonitoringDatasetConfig,
) -> None:
    """Validate the frozen Phase-3 monitoring configuration."""
    if config.steps_per_episode != 64:
        raise ValueError(
            "steps_per_episode must remain frozen at 64."
        )

    if set(config.splits) != set(SPLIT_ORDER):
        raise ValueError(
            "splits must contain exactly train, validation, and test."
        )

    expected_episode_counts = {
        "train": 32,
        "validation": 16,
        "test": 16,
    }

    for split_name in SPLIT_ORDER:
        split = config.splits[split_name]

        if split.episodes != expected_episode_counts[split_name]:
            raise ValueError(
                f"{split_name} episode count does not match "
                "the frozen Phase-3E design."
            )

        if split.allocated_episodes != split.episodes:
            raise ValueError(
                f"{split_name} episode allocation does not sum "
                "to the configured split total."
            )

        for value in (
            split.normal,
            split.power_coupling,
            split.flow_delivery,
            split.pumping_effectiveness,
            split.pressure_sensor_bias,
        ):
            if value < 0:
                raise ValueError(
                    "Episode allocations cannot be negative."
                )

    if config.total_episodes != 64:
        raise ValueError(
            "Monitoring dataset must contain exactly 64 episodes."
        )

    if config.total_rows != 4096:
        raise ValueError(
            "Monitoring dataset must contain exactly 4096 rows."
        )

    recipe = config.nominal_recipe

    if (
        recipe.absorbed_power_min_W
        >= recipe.absorbed_power_max_W
    ):
        raise ValueError(
            "Nominal absorbed-power bounds are invalid."
        )

    if (
        recipe.target_pressure_min_mTorr
        >= recipe.target_pressure_max_mTorr
    ):
        raise ValueError(
            "Nominal pressure bounds are invalid."
        )

    for name, value in {
        "absorbed_power_min_W": recipe.absorbed_power_min_W,
        "absorbed_power_max_W": recipe.absorbed_power_max_W,
        "target_pressure_min_mTorr": (
            recipe.target_pressure_min_mTorr
        ),
        "target_pressure_max_mTorr": (
            recipe.target_pressure_max_mTorr
        ),
        "flow_sccm": recipe.flow_sccm,
    }.items():
        _positive_finite(
            value,
            name,
        )

    fixed = config.fixed_model_parameters

    for name, value in {
        "gas_temperature_K": fixed.gas_temperature_K,
        "ion_neutral_cross_section_m2": (
            fixed.ion_neutral_cross_section_m2
        ),
        "radius_m": fixed.radius_m,
        "length_m": fixed.length_m,
        "initial_electron_density_m3": (
            fixed.initial_electron_density_m3
        ),
        "initial_electron_temperature_eV": (
            fixed.initial_electron_temperature_eV
        ),
        "solver_end_time_s": fixed.solver_end_time_s,
        "convergence_threshold_s": (
            fixed.convergence_threshold_s
        ),
        "balance_residual_limit": (
            fixed.balance_residual_limit
        ),
        "total_particle_identity_limit": (
            fixed.total_particle_identity_limit
        ),
        "minimum_reasonable_electron_temperature_eV": (
            fixed.minimum_reasonable_electron_temperature_eV
        ),
        "maximum_reasonable_electron_temperature_eV": (
            fixed.maximum_reasonable_electron_temperature_eV
        ),
    }.items():
        _positive_finite(
            value,
            name,
        )

    if (
        fixed.minimum_reasonable_electron_temperature_eV
        >= fixed.maximum_reasonable_electron_temperature_eV
    ):
        raise ValueError(
            "Electron-temperature validity bounds are invalid."
        )

    process = config.process_variability

    for name, value in {
        "absorbed_power_relative_sigma": (
            process.absorbed_power_relative_sigma
        ),
        "flow_relative_sigma": process.flow_relative_sigma,
        "pumping_relative_sigma": (
            process.pumping_relative_sigma
        ),
    }.items():
        _validate_relative_sigma(
            value,
            name,
        )

    _positive_finite(
        process.truncation_sigma,
        "process variability truncation_sigma",
    )

    measurement = config.measurement_noise

    for name, value in {
        "absorbed_power_relative_sigma": (
            measurement.absorbed_power_relative_sigma
        ),
        "flow_relative_sigma": (
            measurement.flow_relative_sigma
        ),
        "pressure_relative_sigma": (
            measurement.pressure_relative_sigma
        ),
    }.items():
        _validate_relative_sigma(
            value,
            name,
        )

    _positive_finite(
        measurement.truncation_sigma,
        "measurement noise truncation_sigma",
    )

    onset = config.fault_onset

    if (
        onset.minimum_step != 16
        or onset.maximum_step != 31
    ):
        raise ValueError(
            "Fault onset must remain frozen to steps 16 through 31."
        )

    if onset.maximum_step >= config.steps_per_episode:
        raise ValueError(
            "Fault onset must occur before the episode ends."
        )

    if tuple(config.fault_profiles) != ALLOWED_PROFILES:
        raise ValueError(
            "Fault profiles must remain ('step', 'drift')."
        )

    if set(config.faults) != set(FAULT_FAMILIES):
        raise ValueError(
            "Fault definitions must contain exactly the four "
            "frozen single-fault families."
        )

    expected_domains = {
        "power_coupling": "process",
        "flow_delivery": "process",
        "pumping_effectiveness": "process",
        "pressure_sensor_bias": "sensor",
    }

    expected_directions = {
        "power_coupling": (
            "negative",
        ),
        "flow_delivery": (
            "negative",
            "positive",
        ),
        "pumping_effectiveness": (
            "negative",
            "positive",
        ),
        "pressure_sensor_bias": (
            "negative",
            "positive",
        ),
    }

    for family in FAULT_FAMILIES:
        fault = config.faults[family]

        if fault.domain != expected_domains[family]:
            raise ValueError(
                f"{family} has an invalid fault domain."
            )

        if fault.directions != expected_directions[family]:
            raise ValueError(
                f"{family} has invalid directions."
            )

        magnitudes = (
            fault.mild_fraction,
            fault.moderate_fraction,
            fault.severe_fraction,
        )

        for magnitude in magnitudes:
            if (
                not math.isfinite(magnitude)
                or magnitude <= 0.0
                or magnitude >= 1.0
            ):
                raise ValueError(
                    f"{family} magnitudes must lie in (0, 1)."
                )

        if not (
            fault.mild_fraction
            < fault.moderate_fraction
            < fault.severe_fraction
        ):
            raise ValueError(
                f"{family} severities must increase "
                "mild < moderate < severe."
            )

    seeds = (
        config.seeds.recipe_design,
        config.seeds.episode_fault_plan,
        config.seeds.process_variability,
        config.seeds.measurement_noise,
    )

    if len(set(seeds)) != 4:
        raise ValueError(
            "Monitoring reproducibility seeds must be distinct."
        )

    if any(
        seed < 0
        for seed in seeds
    ):
        raise ValueError(
            "Monitoring seeds must be non-negative."
        )


def load_monitoring_config(
    path: str | Path,
) -> MonitoringDatasetConfig:
    """Load and validate a monitoring-dataset JSON configuration."""
    raw = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    recipe_raw = raw[
        "nominal_recipe"
    ]

    fixed_raw = raw[
        "fixed_model_parameters"
    ]

    process_raw = raw[
        "process_variability"
    ]

    measurement_raw = raw[
        "measurement_noise"
    ]

    onset_raw = raw[
        "fault_onset"
    ]

    seed_raw = raw[
        "seeds"
    ]

    splits = {
        name: MonitoringSplitConfig(
            episodes=int(values["episodes"]),
            normal=int(values["normal"]),
            power_coupling=int(
                values["power_coupling"]
            ),
            flow_delivery=int(
                values["flow_delivery"]
            ),
            pumping_effectiveness=int(
                values["pumping_effectiveness"]
            ),
            pressure_sensor_bias=int(
                values["pressure_sensor_bias"]
            ),
        )
        for name, values in raw[
            "splits"
        ].items()
    }

    faults = {
        name: FaultConfig(
            domain=str(values["domain"]),
            directions=tuple(
                str(direction)
                for direction in values[
                    "directions"
                ]
            ),
            mild_fraction=float(
                values["mild_fraction"]
            ),
            moderate_fraction=float(
                values["moderate_fraction"]
            ),
            severe_fraction=float(
                values["severe_fraction"]
            ),
        )
        for name, values in raw[
            "faults"
        ].items()
    }

    config = MonitoringDatasetConfig(
        dataset_name=str(
            raw["dataset_name"]
        ),
        dataset_version=str(
            raw["dataset_version"]
        ),
        interpretation=str(
            raw["interpretation"]
        ),
        steps_per_episode=int(
            raw["steps_per_episode"]
        ),
        nominal_recipe=NominalRecipeConfig(
            absorbed_power_min_W=float(
                recipe_raw[
                    "absorbed_power_min_W"
                ]
            ),
            absorbed_power_max_W=float(
                recipe_raw[
                    "absorbed_power_max_W"
                ]
            ),
            target_pressure_min_mTorr=float(
                recipe_raw[
                    "target_pressure_min_mTorr"
                ]
            ),
            target_pressure_max_mTorr=float(
                recipe_raw[
                    "target_pressure_max_mTorr"
                ]
            ),
            flow_sccm=float(
                recipe_raw[
                    "flow_sccm"
                ]
            ),
        ),
        fixed_model_parameters=FixedModelConfig(
            gas_temperature_K=float(
                fixed_raw[
                    "gas_temperature_K"
                ]
            ),
            ion_neutral_cross_section_m2=float(
                fixed_raw[
                    "ion_neutral_cross_section_m2"
                ]
            ),
            radius_m=float(
                fixed_raw["radius_m"]
            ),
            length_m=float(
                fixed_raw["length_m"]
            ),
            initial_electron_density_m3=float(
                fixed_raw[
                    "initial_electron_density_m3"
                ]
            ),
            initial_electron_temperature_eV=float(
                fixed_raw[
                    "initial_electron_temperature_eV"
                ]
            ),
            solver_end_time_s=float(
                fixed_raw[
                    "solver_end_time_s"
                ]
            ),
            convergence_threshold_s=float(
                fixed_raw[
                    "convergence_threshold_s"
                ]
            ),
            balance_residual_limit=float(
                fixed_raw[
                    "balance_residual_limit"
                ]
            ),
            total_particle_identity_limit=float(
                fixed_raw[
                    "total_particle_identity_limit"
                ]
            ),
            minimum_reasonable_electron_temperature_eV=float(
                fixed_raw[
                    "minimum_reasonable_electron_temperature_eV"
                ]
            ),
            maximum_reasonable_electron_temperature_eV=float(
                fixed_raw[
                    "maximum_reasonable_electron_temperature_eV"
                ]
            ),
        ),
        splits=splits,
        process_variability=ProcessVariabilityConfig(
            absorbed_power_relative_sigma=float(
                process_raw[
                    "absorbed_power_relative_sigma"
                ]
            ),
            flow_relative_sigma=float(
                process_raw[
                    "flow_relative_sigma"
                ]
            ),
            pumping_relative_sigma=float(
                process_raw[
                    "pumping_relative_sigma"
                ]
            ),
            truncation_sigma=float(
                process_raw[
                    "truncation_sigma"
                ]
            ),
        ),
        measurement_noise=MeasurementNoiseConfig(
            absorbed_power_relative_sigma=float(
                measurement_raw[
                    "absorbed_power_relative_sigma"
                ]
            ),
            flow_relative_sigma=float(
                measurement_raw[
                    "flow_relative_sigma"
                ]
            ),
            pressure_relative_sigma=float(
                measurement_raw[
                    "pressure_relative_sigma"
                ]
            ),
            truncation_sigma=float(
                measurement_raw[
                    "truncation_sigma"
                ]
            ),
        ),
        fault_onset=FaultOnsetConfig(
            minimum_step=int(
                onset_raw[
                    "minimum_step"
                ]
            ),
            maximum_step=int(
                onset_raw[
                    "maximum_step"
                ]
            ),
        ),
        fault_profiles=tuple(
            str(profile)
            for profile in raw[
                "fault_profiles"
            ]
        ),
        faults=faults,
        seeds=MonitoringSeeds(
            recipe_design=int(
                seed_raw[
                    "recipe_design"
                ]
            ),
            episode_fault_plan=int(
                seed_raw[
                    "episode_fault_plan"
                ]
            ),
            process_variability=int(
                seed_raw[
                    "process_variability"
                ]
            ),
            measurement_noise=int(
                seed_raw[
                    "measurement_noise"
                ]
            ),
        ),
    )

    validate_monitoring_config(
        config
    )

    return config
