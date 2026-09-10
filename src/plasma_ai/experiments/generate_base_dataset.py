"""Production generation of the deterministic Phase-3 base dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy

from plasma_ai.experiments.base_dataset import (
    BaseSimulationRow,
    generate_base_design,
    simulate_base_design_point,
)
from plasma_ai.experiments.dataset_artifacts import (
    base_dataset_sha256,
    canonical_base_csv_bytes,
    config_sha256,
    file_sha256,
    physics_source_sha256,
)
from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
    load_base_dataset_config,
)


DEFAULT_CONFIG_PATH = Path(
    "configs/phase3/base_dataset.json"
)

DEFAULT_FEATURE_MANIFEST_PATH = Path(
    "configs/phase3/base_feature_manifest.json"
)

DEFAULT_PHYSICS_ROOT = Path(
    "src/plasma_ai/physics"
)

DEFAULT_DATASET_PATH = Path(
    "data/synthetic/phase3/base_steady_state.csv"
)

DEFAULT_MANIFEST_PATH = Path(
    "artifacts/phase3/base_steady_state_manifest.json"
)

DEFAULT_SUMMARY_PATH = Path(
    "artifacts/phase3/base_steady_state_summary.json"
)


def git_head() -> str:
    """Return the current Git HEAD commit hash."""
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
    ).encode("utf-8")


def build_base_summary(
    rows: list[BaseSimulationRow],
) -> dict[str, object]:
    """Return deterministic summary statistics for generated rows."""
    if not rows:
        raise ValueError(
            "Cannot summarize an empty base dataset."
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

    split_counts = Counter(
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

    summary: dict[str, object] = {
        "rows_total": len(rows),
        "rows_qualification_valid": len(
            valid_rows
        ),
        "rows_ml_eligible": len(
            ml_rows
        ),
        "rows_invalid": (
            len(rows)
            - len(valid_rows)
        ),
        "split_counts": dict(
            sorted(
                split_counts.items()
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
    }

    if valid_rows:
        numerical_fields = {
            "nominal_absorbed_power_W": [
                row.nominal_absorbed_power_W
                for row in valid_rows
            ],
            "target_pressure_mTorr": [
                row.target_pressure_mTorr
                for row in valid_rows
            ],
            "derived_pumping_speed_m3_s": [
                row.derived_pumping_speed_m3_s
                for row in valid_rows
            ],
            "true_electron_density_m3": [
                row.true_electron_density_m3
                for row in valid_rows
            ],
            "true_electron_temperature_eV": [
                row.true_electron_temperature_eV
                for row in valid_rows
            ],
            "true_pressure_mTorr": [
                row.true_pressure_mTorr
                for row in valid_rows
            ],
            "ionization_fraction": [
                row.ionization_fraction
                for row in valid_rows
            ],
            "residence_time_proxy_s": [
                row.residence_time_proxy_s
                for row in valid_rows
            ],
            "collisional_power_loss_W": [
                row.collisional_power_loss_W
                for row in valid_rows
            ],
            "wall_power_loss_W": [
                row.wall_power_loss_W
                for row in valid_rows
            ],
            "max_relative_rate_s": [
                row.max_relative_rate_s
                for row in valid_rows
            ],
            "max_balance_residual": [
                row.max_balance_residual
                for row in valid_rows
            ],
            "total_particle_identity": [
                row.total_particle_identity
                for row in valid_rows
            ],
            "pressure_target_relative_error": [
                row.pressure_target_relative_error
                for row in valid_rows
            ],
        }

        statistics: dict[str, object] = {}

        for name, values in numerical_fields.items():
            array = np.asarray(
                values,
                dtype=float,
            )

            statistics[name] = {
                "min": float(
                    np.min(array)
                ),
                "max": float(
                    np.max(array)
                ),
                "mean": float(
                    np.mean(array)
                ),
                "std": float(
                    np.std(
                        array,
                        ddof=0,
                    )
                ),
            }

        summary["statistics"] = statistics

    return summary


def build_base_manifest(
    *,
    config: BaseDatasetConfig,
    rows: list[BaseSimulationRow],
    config_path: str | Path,
    feature_manifest_path: str | Path,
    physics_root: str | Path,
) -> dict[str, object]:
    """Return the deterministic provenance manifest."""
    dataset_hash = base_dataset_sha256(
        rows
    )

    return {
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "row_count": len(rows),
        "split_counts": {
            name: config.splits[name].rows
            for name in (
                "train",
                "validation",
                "test",
            )
        },
        "sampling_method": (
            config.sampling_method
        ),
        "sampling_dimensions": [
            "nominal_absorbed_power_W",
            "target_pressure_mTorr",
        ],
        "sampling_envelope": {
            "absorbed_power_W": [
                config.absorbed_power_min_W,
                config.absorbed_power_max_W,
            ],
            "target_pressure_mTorr": [
                config.target_pressure_min_mTorr,
                config.target_pressure_max_mTorr,
            ],
        },
        "fixed_parameters": {
            "flow_sccm": (
                config.fixed_flow_sccm
            ),
            "gas_temperature_K": (
                config.gas_temperature_K
            ),
            "ion_neutral_cross_section_m2": (
                config.ion_neutral_cross_section_m2
            ),
            "radius_m": config.radius_m,
            "length_m": config.length_m,
        },
        "split_seeds": {
            name: config.splits[name].seed
            for name in (
                "train",
                "validation",
                "test",
            )
        },
        "solver_and_validation": {
            "solver_end_time_s": (
                config.solver_end_time_s
            ),
            "convergence_threshold_s": (
                config.convergence_threshold_s
            ),
            "balance_residual_limit": (
                config.balance_residual_limit
            ),
            "total_particle_identity_limit": (
                config.total_particle_identity_limit
            ),
        },
        "provenance": {
            "git_head": git_head(),
            "config_sha256": (
                config_sha256(
                    config
                )
            ),
            "config_file_sha256": (
                file_sha256(
                    config_path
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
            "dataset_sha256": dataset_hash,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scientific_scope": {
            "data_type": (
                "synthetic reduced-order "
                "argon plasma simulation"
            ),
            "experimental_data": False,
            "industrial_validation": False,
            "oipt_operating_range": False,
            "generator_power_equivalent": False,
            "reactive_etch_prediction": False,
        },
    }


def write_base_artifacts(
    *,
    rows: list[BaseSimulationRow],
    config: BaseDatasetConfig,
    config_path: str | Path,
    feature_manifest_path: str | Path,
    physics_root: str | Path,
    dataset_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> dict[str, object]:
    """Write canonical dataset, manifest, and summary artifacts."""
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

    dataset_bytes = canonical_base_csv_bytes(
        rows
    )

    manifest = build_base_manifest(
        config=config,
        rows=rows,
        config_path=config_path,
        feature_manifest_path=(
            feature_manifest_path
        ),
        physics_root=physics_root,
    )

    summary = build_base_summary(
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
    ]["dataset_sha256"]

    if written_hash != expected_hash:
        raise RuntimeError(
            "Written dataset SHA-256 does not match "
            "the canonical in-memory dataset hash."
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
        "dataset_sha256": written_hash,
    }


def generate_production_base_dataset(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    feature_manifest_path: str | Path = (
        DEFAULT_FEATURE_MANIFEST_PATH
    ),
    physics_root: str | Path = DEFAULT_PHYSICS_ROOT,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    progress_every: int = 256,
) -> dict[str, object]:
    """Generate and write the full configured deterministic dataset."""
    if progress_every <= 0:
        raise ValueError(
            "progress_every must be positive."
        )

    config = load_base_dataset_config(
        config_path
    )

    design = generate_base_design(
        config
    )

    rows: list[BaseSimulationRow] = []

    start = perf_counter()

    for index, point in enumerate(
        design,
        start=1,
    ):
        row = simulate_base_design_point(
            point,
            config,
        )

        rows.append(
            row
        )

        if (
            index % progress_every == 0
            or index == len(design)
        ):
            elapsed = (
                perf_counter()
                - start
            )

            valid = sum(
                item.qualification_valid
                for item in rows
            )

            print(
                f"{index:>5}/{len(design)} complete | "
                f"valid={valid:>5}/{index:<5} | "
                f"elapsed={elapsed / 60.0:.2f} min",
                flush=True,
            )

    elapsed = (
        perf_counter()
        - start
    )

    artifact_info = write_base_artifacts(
        rows=rows,
        config=config,
        config_path=config_path,
        feature_manifest_path=(
            feature_manifest_path
        ),
        physics_root=physics_root,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )

    artifact_info[
        "rows_total"
    ] = len(rows)

    artifact_info[
        "rows_valid"
    ] = sum(
        row.qualification_valid
        for row in rows
    )

    artifact_info[
        "rows_ml_eligible"
    ] = sum(
        row.is_ml_eligible
        for row in rows
    )

    artifact_info[
        "elapsed_seconds"
    ] = elapsed

    return artifact_info


def main() -> None:
    """Generate the frozen production Phase-3 base dataset."""
    print(
        "PHASE 3D ? generating deterministic "
        "base steady-state dataset",
        flush=True,
    )

    result = generate_production_base_dataset()

    print()
    print("Generation complete")
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
