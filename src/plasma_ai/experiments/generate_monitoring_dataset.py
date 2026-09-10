"""Production artifact generation for Phase-3 monitoring data."""

from __future__ import annotations

from collections import Counter
import json
import math
import platform
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import scipy

from plasma_ai.experiments.dataset_artifacts import (
    file_sha256,
    physics_source_sha256,
)
from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
    load_base_dataset_config,
)
from plasma_ai.experiments.monitoring_artifacts import (
    canonical_monitoring_csv_bytes,
    load_monitoring_feature_manifest,
    monitoring_config_sha256,
    monitoring_dataset_sha256,
)
from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
    SPLIT_ORDER,
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_dataset import (
    MonitoringRow,
    build_monitoring_row,
)
from plasma_ai.experiments.monitoring_observation import (
    build_monitoring_observation,
)
from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
    generate_monitoring_episode_plan,
)
from plasma_ai.experiments.monitoring_process import (
    simulate_monitoring_process_step,
)


DEFAULT_MONITORING_CONFIG_PATH = Path(
    "configs/phase3/monitoring_dataset.json"
)

DEFAULT_BASE_CONFIG_PATH = Path(
    "configs/phase3/base_dataset.json"
)

DEFAULT_FEATURE_MANIFEST_PATH = Path(
    "configs/phase3/monitoring_feature_manifest.json"
)

DEFAULT_PHYSICS_ROOT = Path(
    "src/plasma_ai/physics"
)

DEFAULT_EXPERIMENTS_ROOT = Path(
    "src/plasma_ai/experiments"
)

DEFAULT_DATASET_PATH = Path(
    "data/synthetic/phase3/monitoring_episodes.csv"
)

DEFAULT_MANIFEST_PATH = Path(
    "artifacts/phase3/monitoring_manifest.json"
)

DEFAULT_SUMMARY_PATH = Path(
    "artifacts/phase3/monitoring_summary.json"
)


def _json_bytes(
    payload: dict[str, object],
) -> bytes:
    """Serialize a JSON artifact deterministically."""
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )

    return (
        text
        + "\n"
    ).encode(
        "utf-8"
    )


def git_head() -> str:
    """Return the current Git HEAD."""
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def generate_monitoring_row(
    *,
    config: MonitoringDatasetConfig,
    base_config: BaseDatasetConfig,
    plan: MonitoringEpisodePlan,
    step_index: int,
) -> MonitoringRow:
    """Generate one complete monitoring row."""
    state = simulate_monitoring_process_step(
        config,
        plan,
        step_index,
    )

    observation = build_monitoring_observation(
        config,
        plan,
        state,
    )

    return build_monitoring_row(
        config=config,
        base_config=base_config,
        plan=plan,
        state=state,
        observation=observation,
    )


def _finite_statistics(
    values: list[float],
) -> dict[str, float] | None:
    """Return deterministic statistics over finite values only."""
    array = np.asarray(
        values,
        dtype=float,
    )

    finite = array[
        np.isfinite(
            array
        )
    ]

    if finite.size == 0:
        return None

    return {
        "min": float(
            np.min(
                finite
            )
        ),
        "max": float(
            np.max(
                finite
            )
        ),
        "mean": float(
            np.mean(
                finite
            )
        ),
        "std": float(
            np.std(
                finite,
                ddof=0,
            )
        ),
    }


def build_monitoring_summary(
    rows: list[MonitoringRow],
) -> dict[str, object]:
    """Return deterministic monitoring-data QA statistics."""
    if not rows:
        raise ValueError(
            "Cannot summarize an empty monitoring dataset."
        )

    split_row_counts = Counter(
        row.split
        for row in rows
    )

    domain_counts = Counter(
        row.domain_status
        for row in rows
    )

    validity_counts = Counter(
        row.model_validity_status
        for row in rows
    )

    active_family_counts = Counter(
        row.active_fault_family
        for row in rows
    )

    active_domain_counts = Counter(
        row.active_fault_domain
        for row in rows
    )

    episode_first_rows: dict[
        str,
        MonitoringRow,
    ] = {}

    for row in rows:
        episode_first_rows.setdefault(
            row.episode_id,
            row,
        )

    episode_rows = list(
        episode_first_rows.values()
    )

    split_episode_counts = Counter(
        row.split
        for row in episode_rows
    )

    episode_fault_family_counts = Counter(
        row.fault_family
        for row in episode_rows
    )

    faulty_episode_rows = [
        row
        for row in episode_rows
        if row.fault_present
    ]

    fault_severity_episode_counts = Counter(
        row.fault_severity
        for row in faulty_episode_rows
    )

    fault_profile_episode_counts = Counter(
        row.fault_profile
        for row in faulty_episode_rows
    )

    fault_direction_episode_counts = Counter(
        row.fault_direction
        for row in faulty_episode_rows
    )

    valid_rows = [
        row
        for row in rows
        if row.qualification_valid
    ]

    ml_rows = [
        row
        for row in rows
        if row.is_ml_eligible
    ]

    numerical_fields = (
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
        "true_absorbed_power_W",
        "true_flow_sccm",
        "true_pumping_speed_m3_s",
        "true_pressure_mTorr",
        "true_electron_density_m3",
        "true_electron_temperature_eV",
        "measured_absorbed_power_W",
        "measured_flow_sccm",
        "measured_pressure_mTorr",
        "fault_progress",
        "signed_fault_fraction",
        "max_relative_rate_s",
        "max_balance_residual",
        "total_particle_identity",
    )

    statistics: dict[
        str,
        object,
    ] = {}

    for field_name in numerical_fields:
        result = _finite_statistics(
            [
                float(
                    getattr(
                        row,
                        field_name,
                    )
                )
                for row in rows
            ]
        )

        if result is not None:
            statistics[
                field_name
            ] = result

    return {
        "rows_total": len(
            rows
        ),
        "episodes_total": len(
            episode_rows
        ),
        "rows_qualification_valid": len(
            valid_rows
        ),
        "rows_ml_eligible": len(
            ml_rows
        ),
        "rows_invalid": (
            len(
                rows
            )
            - len(
                valid_rows
            )
        ),
        "rows_fault_episode": sum(
            row.fault_present
            for row in rows
        ),
        "rows_fault_started": sum(
            row.fault_started
            for row in rows
        ),
        "rows_fault_effect_active": sum(
            row.fault_effect_active
            for row in rows
        ),
        "split_row_counts": dict(
            sorted(
                split_row_counts.items()
            )
        ),
        "split_episode_counts": dict(
            sorted(
                split_episode_counts.items()
            )
        ),
        "domain_status_counts": dict(
            sorted(
                domain_counts.items()
            )
        ),
        "model_validity_status_counts": dict(
            sorted(
                validity_counts.items()
            )
        ),
        "active_fault_family_row_counts": dict(
            sorted(
                active_family_counts.items()
            )
        ),
        "active_fault_domain_row_counts": dict(
            sorted(
                active_domain_counts.items()
            )
        ),
        "episode_fault_family_counts": dict(
            sorted(
                episode_fault_family_counts.items()
            )
        ),
        "fault_severity_episode_counts": dict(
            sorted(
                fault_severity_episode_counts.items()
            )
        ),
        "fault_profile_episode_counts": dict(
            sorted(
                fault_profile_episode_counts.items()
            )
        ),
        "fault_direction_episode_counts": dict(
            sorted(
                fault_direction_episode_counts.items()
            )
        ),
        "statistics": statistics,
    }


def build_monitoring_manifest(
    *,
    config: MonitoringDatasetConfig,
    base_config: BaseDatasetConfig,
    rows: list[MonitoringRow],
    config_path: str | Path,
    base_config_path: str | Path,
    feature_manifest_path: str | Path,
    physics_root: str | Path,
    experiments_root: str | Path,
) -> dict[str, object]:
    """Return deterministic monitoring provenance metadata."""
    feature_manifest = (
        load_monitoring_feature_manifest(
            feature_manifest_path
        )
    )

    split_row_counts = Counter(
        row.split
        for row in rows
    )

    split_episode_sets: dict[
        str,
        set[str],
    ] = {
        name: set()
        for name in SPLIT_ORDER
    }

    for row in rows:
        if row.split in split_episode_sets:
            split_episode_sets[
                row.split
            ].add(
                row.episode_id
            )

    return {
        "dataset_name": (
            config.dataset_name
        ),
        "dataset_version": (
            config.dataset_version
        ),
        "interpretation": (
            config.interpretation
        ),
        "row_count": len(
            rows
        ),
        "episode_count": len(
            {
                row.episode_id
                for row in rows
            }
        ),
        "steps_per_episode": (
            config.steps_per_episode
        ),
        "actual_split_row_counts": {
            name: int(
                split_row_counts.get(
                    name,
                    0,
                )
            )
            for name in SPLIT_ORDER
        },
        "actual_split_episode_counts": {
            name: len(
                split_episode_sets[
                    name
                ]
            )
            for name in SPLIT_ORDER
        },
        "expected_design": {
            "episodes": (
                config.total_episodes
            ),
            "rows": (
                config.total_rows
            ),
            "split_episode_counts": {
                name: (
                    config.splits[
                        name
                    ].episodes
                )
                for name in SPLIT_ORDER
            },
            "split_row_counts": {
                name: (
                    config.splits[
                        name
                    ].episodes
                    * config.steps_per_episode
                )
                for name in SPLIT_ORDER
            },
        },
        "nominal_recipe_envelope": {
            "absorbed_power_W": [
                config.nominal_recipe.absorbed_power_min_W,
                config.nominal_recipe.absorbed_power_max_W,
            ],
            "target_pressure_mTorr": [
                config.nominal_recipe.target_pressure_min_mTorr,
                config.nominal_recipe.target_pressure_max_mTorr,
            ],
            "flow_sccm": (
                config.nominal_recipe.flow_sccm
            ),
        },
        "qualified_base_envelope": {
            "absorbed_power_W": [
                base_config.absorbed_power_min_W,
                base_config.absorbed_power_max_W,
            ],
            "target_pressure_mTorr": [
                base_config.target_pressure_min_mTorr,
                base_config.target_pressure_max_mTorr,
            ],
        },
        "process_variability": {
            "absorbed_power_relative_sigma": (
                config.process_variability.absorbed_power_relative_sigma
            ),
            "flow_relative_sigma": (
                config.process_variability.flow_relative_sigma
            ),
            "pumping_relative_sigma": (
                config.process_variability.pumping_relative_sigma
            ),
            "truncation_sigma": (
                config.process_variability.truncation_sigma
            ),
        },
        "measurement_noise": {
            "absorbed_power_relative_sigma": (
                config.measurement_noise.absorbed_power_relative_sigma
            ),
            "flow_relative_sigma": (
                config.measurement_noise.flow_relative_sigma
            ),
            "pressure_relative_sigma": (
                config.measurement_noise.pressure_relative_sigma
            ),
            "truncation_sigma": (
                config.measurement_noise.truncation_sigma
            ),
        },
        "fault_design": {
            "single_fault_only": True,
            "families": sorted(
                config.faults
            ),
            "profiles": list(
                config.fault_profiles
            ),
            "onset_step_range": [
                config.fault_onset.minimum_step,
                config.fault_onset.maximum_step,
            ],
        },
        "seeds": {
            "recipe_design": (
                config.seeds.recipe_design
            ),
            "episode_fault_plan": (
                config.seeds.episode_fault_plan
            ),
            "process_variability": (
                config.seeds.process_variability
            ),
            "measurement_noise": (
                config.seeds.measurement_noise
            ),
        },
        "solver_and_validation": {
            "solver_end_time_s": (
                config.fixed_model_parameters.solver_end_time_s
            ),
            "convergence_threshold_s": (
                config.fixed_model_parameters.convergence_threshold_s
            ),
            "balance_residual_limit": (
                config.fixed_model_parameters.balance_residual_limit
            ),
            "total_particle_identity_limit": (
                config.fixed_model_parameters.total_particle_identity_limit
            ),
            "cold_start_policy": (
                "independent physics-informed Q/S neutral initialization"
            ),
        },
        "feature_contract": {
            "features": list(
                feature_manifest[
                    "features"
                ]
            ),
            "binary_target": (
                feature_manifest[
                    "binary_target"
                ]
            ),
            "multiclass_target": (
                feature_manifest[
                    "multiclass_target"
                ]
            ),
            "group_columns": list(
                feature_manifest[
                    "group_columns"
                ]
            ),
            "ordering_column": (
                feature_manifest[
                    "ordering_column"
                ]
            ),
        },
        "provenance": {
            "git_head": git_head(),
            "monitoring_config_sha256": (
                monitoring_config_sha256(
                    config
                )
            ),
            "monitoring_config_file_sha256": (
                file_sha256(
                    config_path
                )
            ),
            "base_config_file_sha256": (
                file_sha256(
                    base_config_path
                )
            ),
            "feature_manifest_sha256": (
                file_sha256(
                    feature_manifest_path
                )
            ),
            "physics_source_sha256": (
                physics_source_sha256(
                    physics_root
                )
            ),
            "experiments_source_sha256": (
                physics_source_sha256(
                    experiments_root
                )
            ),
            "dataset_sha256": (
                monitoring_dataset_sha256(
                    rows
                )
            ),
        },
        "software": {
            "python": (
                platform.python_version()
            ),
            "numpy": (
                np.__version__
            ),
            "scipy": (
                scipy.__version__
            ),
        },
        "scientific_scope": {
            "data_type": (
                "synthetic quasi-steady reduced-order "
                "argon plasma monitoring simulation"
            ),
            "ordered_steps_are_calibrated_physical_time": False,
            "experimental_data": False,
            "industrial_validation": False,
            "oipt_operating_range": False,
            "generator_power_equivalent": False,
            "reactive_etch_prediction": False,
            "compound_faults": False,
            "synthetic_scenario_assumptions": True,
            "fabricated_measured_electron_density": False,
            "fabricated_measured_electron_temperature": False,
        },
    }


def write_monitoring_artifacts(
    *,
    rows: list[MonitoringRow],
    config: MonitoringDatasetConfig,
    base_config: BaseDatasetConfig,
    config_path: str | Path,
    base_config_path: str | Path,
    feature_manifest_path: str | Path,
    physics_root: str | Path,
    experiments_root: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> dict[str, object]:
    """Write canonical monitoring CSV, manifest, and summary."""
    dataset_target = Path(
        dataset_path
    )

    manifest_target = Path(
        manifest_path
    )

    summary_target = Path(
        summary_path
    )

    dataset_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_bytes = (
        canonical_monitoring_csv_bytes(
            rows
        )
    )

    manifest = build_monitoring_manifest(
        config=config,
        base_config=base_config,
        rows=rows,
        config_path=config_path,
        base_config_path=(
            base_config_path
        ),
        feature_manifest_path=(
            feature_manifest_path
        ),
        physics_root=(
            physics_root
        ),
        experiments_root=(
            experiments_root
        ),
    )

    summary = build_monitoring_summary(
        rows
    )

    dataset_target.write_bytes(
        dataset_bytes
    )

    manifest_target.write_bytes(
        _json_bytes(
            manifest
        )
    )

    summary_target.write_bytes(
        _json_bytes(
            summary
        )
    )

    written_hash = file_sha256(
        dataset_target
    )

    expected_hash = manifest[
        "provenance"
    ][
        "dataset_sha256"
    ]

    if written_hash != expected_hash:
        raise RuntimeError(
            "Written monitoring dataset SHA-256 does not "
            "match the canonical in-memory hash."
        )

    return {
        "dataset_path": str(
            dataset_target
        ),
        "manifest_path": str(
            manifest_target
        ),
        "summary_path": str(
            summary_target
        ),
        "dataset_sha256": (
            written_hash
        ),
    }


def generate_production_monitoring_dataset(
    *,
    config_path: str | Path = (
        DEFAULT_MONITORING_CONFIG_PATH
    ),
    base_config_path: str | Path = (
        DEFAULT_BASE_CONFIG_PATH
    ),
    feature_manifest_path: str | Path = (
        DEFAULT_FEATURE_MANIFEST_PATH
    ),
    physics_root: str | Path = (
        DEFAULT_PHYSICS_ROOT
    ),
    experiments_root: str | Path = (
        DEFAULT_EXPERIMENTS_ROOT
    ),
    dataset_path: str | Path = (
        DEFAULT_DATASET_PATH
    ),
    manifest_path: str | Path = (
        DEFAULT_MANIFEST_PATH
    ),
    summary_path: str | Path = (
        DEFAULT_SUMMARY_PATH
    ),
    progress_every: int = 128,
) -> dict[str, object]:
    """Generate the complete frozen 4096-row monitoring dataset."""
    if progress_every <= 0:
        raise ValueError(
            "progress_every must be positive."
        )

    config = load_monitoring_config(
        config_path
    )

    base_config = (
        load_base_dataset_config(
            base_config_path
        )
    )

    # Validate feature contract before expensive simulation work.
    load_monitoring_feature_manifest(
        feature_manifest_path
    )

    plans = (
        generate_monitoring_episode_plan(
            config
        )
    )

    if len(
        plans
    ) != config.total_episodes:
        raise RuntimeError(
            "Episode plan does not match configured episode count."
        )

    rows: list[
        MonitoringRow
    ] = []

    start = perf_counter()

    for plan in plans:
        for step_index in range(
            config.steps_per_episode
        ):
            row = generate_monitoring_row(
                config=config,
                base_config=base_config,
                plan=plan,
                step_index=step_index,
            )

            rows.append(
                row
            )

            count = len(
                rows
            )

            if (
                count % progress_every == 0
                or count == config.total_rows
            ):
                elapsed = (
                    perf_counter()
                    - start
                )

                valid = sum(
                    item.qualification_valid
                    for item in rows
                )

                eligible = sum(
                    item.is_ml_eligible
                    for item in rows
                )

                print(
                    f"{count:>5}/{config.total_rows} complete | "
                    f"valid={valid:>5}/{count:<5} | "
                    f"ml={eligible:>5}/{count:<5} | "
                    f"elapsed={elapsed / 60.0:.2f} min",
                    flush=True,
                )

    if len(
        rows
    ) != config.total_rows:
        raise RuntimeError(
            "Monitoring generator did not produce exactly "
            f"{config.total_rows} rows."
        )

    elapsed = (
        perf_counter()
        - start
    )

    artifact_info = (
        write_monitoring_artifacts(
            rows=rows,
            config=config,
            base_config=base_config,
            config_path=config_path,
            base_config_path=(
                base_config_path
            ),
            feature_manifest_path=(
                feature_manifest_path
            ),
            physics_root=(
                physics_root
            ),
            experiments_root=(
                experiments_root
            ),
            dataset_path=(
                dataset_path
            ),
            manifest_path=(
                manifest_path
            ),
            summary_path=(
                summary_path
            ),
        )
    )

    artifact_info[
        "rows_total"
    ] = int(
        len(
            rows
        )
    )

    artifact_info[
        "rows_valid"
    ] = int(
        sum(
            row.qualification_valid
            for row in rows
        )
    )

    artifact_info[
        "rows_ml_eligible"
    ] = int(
        sum(
            row.is_ml_eligible
            for row in rows
        )
    )

    artifact_info[
        "episodes_total"
    ] = int(
        len(
            {
                row.episode_id
                for row in rows
            }
        )
    )

    artifact_info[
        "elapsed_seconds"
    ] = float(
        elapsed
    )

    return artifact_info


def main() -> None:
    """Generate the frozen Phase-3 monitoring dataset."""
    print(
        "PHASE 3E ? generating deterministic "
        "synthetic monitoring dataset",
        flush=True,
    )

    result = (
        generate_production_monitoring_dataset()
    )

    print()
    print(
        "Generation complete"
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
