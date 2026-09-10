"""Canonical hashing and feature-manifest utilities for monitoring data."""

from __future__ import annotations

from dataclasses import asdict, fields
import csv
import io
import json
from pathlib import Path
from typing import Sequence

from plasma_ai.experiments.dataset_artifacts import (
    _canonical_csv_value,
    sha256_bytes,
)
from plasma_ai.experiments.monitoring_config import (
    MonitoringDatasetConfig,
)
from plasma_ai.experiments.monitoring_dataset import (
    APPROVED_BINARY_TARGET,
    APPROVED_MONITORING_FEATURES,
    APPROVED_MULTICLASS_TARGET,
    MonitoringRow,
)


def _monitoring_csv_value(
    value: object,
) -> str:
    """Convert one monitoring value to canonical CSV text."""
    if value is None:
        return ""

    return _canonical_csv_value(
        value
    )


def canonical_monitoring_csv_bytes(
    rows: Sequence[MonitoringRow],
) -> bytes:
    """Serialize monitoring rows in frozen dataclass field order."""
    column_names = [
        field.name
        for field in fields(
            MonitoringRow
        )
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
                _monitoring_csv_value(
                    getattr(
                        row,
                        column_name,
                    )
                )
                for column_name in column_names
            ]
        )

    return stream.getvalue().encode(
        "utf-8"
    )


def monitoring_dataset_sha256(
    rows: Sequence[MonitoringRow],
) -> str:
    """Return SHA-256 of canonical monitoring CSV bytes."""
    return sha256_bytes(
        canonical_monitoring_csv_bytes(
            rows
        )
    )


def canonical_monitoring_config_bytes(
    config: MonitoringDatasetConfig,
) -> bytes:
    """Serialize monitoring configuration semantics deterministically."""
    text = json.dumps(
        asdict(
            config
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
        allow_nan=False,
    )

    return text.encode(
        "utf-8"
    )


def monitoring_config_sha256(
    config: MonitoringDatasetConfig,
) -> str:
    """Return semantic SHA-256 for monitoring configuration."""
    return sha256_bytes(
        canonical_monitoring_config_bytes(
            config
        )
    )


def load_monitoring_feature_manifest(
    path: str | Path,
) -> dict[str, object]:
    """Load and validate the leakage-safe monitoring feature contract."""
    manifest = json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8"
        )
    )

    features = tuple(
        manifest[
            "features"
        ]
    )

    if (
        features
        != APPROVED_MONITORING_FEATURES
    ):
        raise ValueError(
            "Monitoring feature manifest does not match "
            "the frozen approved feature set."
        )

    binary_target = str(
        manifest[
            "binary_target"
        ]
    )

    multiclass_target = str(
        manifest[
            "multiclass_target"
        ]
    )

    if (
        binary_target
        != APPROVED_BINARY_TARGET
    ):
        raise ValueError(
            "Monitoring binary target does not match "
            "the frozen target."
        )

    if (
        multiclass_target
        != APPROVED_MULTICLASS_TARGET
    ):
        raise ValueError(
            "Monitoring multiclass target does not match "
            "the frozen target."
        )

    schema_fields = {
        field.name
        for field in fields(
            MonitoringRow
        )
    }

    referenced = (
        set(
            features
        )
        | {
            binary_target,
            multiclass_target,
        }
        | set(
            manifest.get(
                "group_columns",
                [],
            )
        )
        | {
            str(
                manifest[
                    "ordering_column"
                ]
            )
        }
        | set(
            manifest.get(
                "protected_ground_truth",
                [],
            )
        )
    )

    missing = (
        referenced
        - schema_fields
    )

    if missing:
        raise ValueError(
            "Monitoring feature manifest references unknown "
            f"schema columns: {sorted(missing)}"
        )

    protected = set(
        manifest.get(
            "protected_ground_truth",
            [],
        )
    )

    leaked_features = (
        set(
            features
        )
        & protected
    )

    if leaked_features:
        raise ValueError(
            "Protected monitoring ground truth appears in "
            f"the feature set: {sorted(leaked_features)}"
        )

    if (
        binary_target in features
        or multiclass_target in features
    ):
        raise ValueError(
            "Monitoring targets cannot also be model features."
        )

    return manifest
