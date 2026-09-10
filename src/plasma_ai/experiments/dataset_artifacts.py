"""Canonical serialization and provenance utilities for Phase-3 data."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

from plasma_ai.experiments.base_dataset import (
    BaseSimulationRow,
)
from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
)


APPROVED_BASE_FEATURES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
)

APPROVED_PRIMARY_TARGETS = (
    "true_electron_density_m3",
    "true_electron_temperature_eV",
)


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 for one file's exact bytes."""
    return sha256_bytes(
        Path(path).read_bytes()
    )


def canonical_config_bytes(
    config: BaseDatasetConfig,
) -> bytes:
    """Serialize configuration semantics deterministically.

    The hash therefore does not depend on whitespace or key ordering
    in the source JSON file.
    """
    text = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

    return text.encode("utf-8")


def config_sha256(
    config: BaseDatasetConfig,
) -> str:
    """Return the semantic SHA-256 of a base-dataset configuration."""
    return sha256_bytes(
        canonical_config_bytes(config)
    )


def physics_source_sha256(
    physics_root: str | Path,
) -> str:
    """Hash the complete Python physics source tree deterministically."""
    root = Path(physics_root)

    paths = sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
    )

    if not paths:
        raise ValueError(
            f"No Python physics files found under {root}."
        )

    digest = hashlib.sha256()

    for path in paths:
        relative_path = path.relative_to(root).as_posix()

        digest.update(
            relative_path.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def _canonical_csv_value(value: object) -> str:
    """Convert one dataclass value to stable CSV text."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"

        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"

        return format(value, ".17g")

    return str(value)


def canonical_base_csv_bytes(
    rows: Sequence[BaseSimulationRow],
) -> bytes:
    """Serialize base rows using frozen schema order and CSV formatting."""
    column_names = [
        field.name
        for field in fields(BaseSimulationRow)
    ]

    stream = io.StringIO(
        newline=""
    )

    writer = csv.writer(
        stream,
        lineterminator="\n",
    )

    writer.writerow(
        column_names
    )

    for row in rows:
        writer.writerow(
            [
                _canonical_csv_value(
                    getattr(row, column_name)
                )
                for column_name in column_names
            ]
        )

    return stream.getvalue().encode("utf-8")


def base_dataset_sha256(
    rows: Sequence[BaseSimulationRow],
) -> str:
    """Return SHA-256 of the canonical base-dataset CSV bytes."""
    return sha256_bytes(
        canonical_base_csv_bytes(rows)
    )


def load_feature_manifest(
    path: str | Path,
) -> dict[str, object]:
    """Load and validate the frozen base feature manifest."""
    manifest = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    features = tuple(
        manifest["features"]
    )

    primary_targets = tuple(
        manifest["primary_targets"]
    )

    if features != APPROVED_BASE_FEATURES:
        raise ValueError(
            "Feature manifest does not match the frozen "
            "Phase-3 base feature set."
        )

    if primary_targets != APPROVED_PRIMARY_TARGETS:
        raise ValueError(
            "Primary targets do not match the frozen "
            "Phase-3 surrogate targets."
        )

    schema_fields = {
        field.name
        for field in fields(BaseSimulationRow)
    }

    requested_fields = (
        set(features)
        | set(primary_targets)
        | set(
            manifest.get(
                "secondary_targets",
                [],
            )
        )
    )

    missing = requested_fields - schema_fields

    if missing:
        raise ValueError(
            "Feature manifest references unknown base-dataset "
            f"columns: {sorted(missing)}"
        )

    if set(features) & set(primary_targets):
        raise ValueError(
            "Features and primary targets must be disjoint."
        )

    return manifest
