"""Configuration models for reproducible Phase-3 datasets."""

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


@dataclass(frozen=True)
class SplitConfig:
    """Sampling configuration for one frozen dataset split."""

    rows: int
    seed: int


@dataclass(frozen=True)
class BaseDatasetConfig:
    """Frozen configuration for the deterministic base dataset."""

    dataset_name: str
    dataset_version: str
    sampling_method: str

    absorbed_power_min_W: float
    absorbed_power_max_W: float
    target_pressure_min_mTorr: float
    target_pressure_max_mTorr: float

    fixed_flow_sccm: float
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

    near_boundary_fraction: float

    splits: dict[str, SplitConfig]

    @property
    def total_rows(self) -> int:
        """Return the total configured row count."""
        return sum(
            self.splits[name].rows
            for name in SPLIT_ORDER
        )


def _validate_positive_finite(
    value: float,
    name: str,
) -> None:
    """Require a finite strictly positive configuration value."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{name} must be finite and strictly positive."
        )


def validate_base_dataset_config(
    config: BaseDatasetConfig,
) -> None:
    """Validate the frozen deterministic base-dataset configuration."""
    if config.sampling_method != "scrambled_sobol":
        raise ValueError(
            "sampling_method must be 'scrambled_sobol'."
        )

    if set(config.splits) != set(SPLIT_ORDER):
        raise ValueError(
            "splits must contain exactly train, validation, and test."
        )

    if (
        config.absorbed_power_min_W
        >= config.absorbed_power_max_W
    ):
        raise ValueError(
            "absorbed-power bounds are invalid."
        )

    if (
        config.target_pressure_min_mTorr
        >= config.target_pressure_max_mTorr
    ):
        raise ValueError(
            "pressure bounds are invalid."
        )

    positive_values = {
        "absorbed_power_min_W": config.absorbed_power_min_W,
        "absorbed_power_max_W": config.absorbed_power_max_W,
        "target_pressure_min_mTorr": (
            config.target_pressure_min_mTorr
        ),
        "target_pressure_max_mTorr": (
            config.target_pressure_max_mTorr
        ),
        "fixed_flow_sccm": config.fixed_flow_sccm,
        "gas_temperature_K": config.gas_temperature_K,
        "ion_neutral_cross_section_m2": (
            config.ion_neutral_cross_section_m2
        ),
        "radius_m": config.radius_m,
        "length_m": config.length_m,
        "initial_electron_density_m3": (
            config.initial_electron_density_m3
        ),
        "initial_electron_temperature_eV": (
            config.initial_electron_temperature_eV
        ),
        "solver_end_time_s": config.solver_end_time_s,
        "convergence_threshold_s": (
            config.convergence_threshold_s
        ),
        "balance_residual_limit": (
            config.balance_residual_limit
        ),
        "total_particle_identity_limit": (
            config.total_particle_identity_limit
        ),
        "minimum_reasonable_electron_temperature_eV": (
            config.minimum_reasonable_electron_temperature_eV
        ),
        "maximum_reasonable_electron_temperature_eV": (
            config.maximum_reasonable_electron_temperature_eV
        ),
    }

    for name, value in positive_values.items():
        _validate_positive_finite(
            value,
            name,
        )

    if not (
        0.0
        < config.near_boundary_fraction
        < 0.5
    ):
        raise ValueError(
            "near_boundary_fraction must lie strictly between 0 and 0.5."
        )

    if (
        config.minimum_reasonable_electron_temperature_eV
        >= config.maximum_reasonable_electron_temperature_eV
    ):
        raise ValueError(
            "electron-temperature validity bounds are invalid."
        )

    for split_name in SPLIT_ORDER:
        split = config.splits[split_name]

        if split.rows <= 0:
            raise ValueError(
                f"{split_name} rows must be positive."
            )

        if split.rows & (split.rows - 1):
            raise ValueError(
                f"{split_name} rows must be a power of two "
                "for Sobol random_base2 sampling."
            )

        if split.seed < 0:
            raise ValueError(
                f"{split_name} seed must be non-negative."
            )


def load_base_dataset_config(
    path: str | Path,
) -> BaseDatasetConfig:
    """Load and validate a Phase-3 base-dataset JSON configuration."""
    config_path = Path(path)

    raw = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    splits = {
        name: SplitConfig(
            rows=int(values["rows"]),
            seed=int(values["seed"]),
        )
        for name, values in raw["splits"].items()
    }

    config = BaseDatasetConfig(
        dataset_name=str(raw["dataset_name"]),
        dataset_version=str(raw["dataset_version"]),
        sampling_method=str(raw["sampling_method"]),
        absorbed_power_min_W=float(
            raw["absorbed_power_min_W"]
        ),
        absorbed_power_max_W=float(
            raw["absorbed_power_max_W"]
        ),
        target_pressure_min_mTorr=float(
            raw["target_pressure_min_mTorr"]
        ),
        target_pressure_max_mTorr=float(
            raw["target_pressure_max_mTorr"]
        ),
        fixed_flow_sccm=float(
            raw["fixed_flow_sccm"]
        ),
        gas_temperature_K=float(
            raw["gas_temperature_K"]
        ),
        ion_neutral_cross_section_m2=float(
            raw["ion_neutral_cross_section_m2"]
        ),
        radius_m=float(raw["radius_m"]),
        length_m=float(raw["length_m"]),
        initial_electron_density_m3=float(
            raw["initial_electron_density_m3"]
        ),
        initial_electron_temperature_eV=float(
            raw["initial_electron_temperature_eV"]
        ),
        solver_end_time_s=float(
            raw["solver_end_time_s"]
        ),
        convergence_threshold_s=float(
            raw["convergence_threshold_s"]
        ),
        balance_residual_limit=float(
            raw["balance_residual_limit"]
        ),
        total_particle_identity_limit=float(
            raw["total_particle_identity_limit"]
        ),
        minimum_reasonable_electron_temperature_eV=float(
            raw[
                "minimum_reasonable_electron_temperature_eV"
            ]
        ),
        maximum_reasonable_electron_temperature_eV=float(
            raw[
                "maximum_reasonable_electron_temperature_eV"
            ]
        ),
        near_boundary_fraction=float(
            raw["near_boundary_fraction"]
        ),
        splits=splits,
    )

    validate_base_dataset_config(config)

    return config
