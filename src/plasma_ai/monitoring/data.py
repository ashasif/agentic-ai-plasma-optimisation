"""Leakage-safe Phase 5 monitoring-data access.

This module implements the frozen Phase 5B development-data contract.

Only TRAIN and VALIDATION payloads are exposed. The canonical monitoring CSV
contains TEST rows, so the parser inspects only the split metadata required to
identify and skip those rows. TEST predictive features and TEST targets are not
interpreted, converted, returned, scored, or made available through the public
development API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import hashlib
import json

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROTOCOL_PATH = (
    _REPO_ROOT
    / "configs"
    / "phase5"
    / "phase5_protocol.json"
)

RAW_FEATURES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
    "nominal_flow_sccm",
    "measured_absorbed_power_W",
    "measured_flow_sccm",
    "measured_pressure_mTorr",
)

DERIVED_FEATURES = (
    "absorbed_power_relative_residual",
    "flow_relative_residual",
    "pressure_relative_residual",
)

MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES

BINARY_TARGET = "fault_effect_active"
MULTICLASS_TARGET = "active_fault_family"

_ALLOWED_DEVELOPMENT_SPLITS = (
    "train",
    "validation",
)


@dataclass(frozen=True)
class MonitoringDevelopmentSplit:
    """One immutable Phase 5 development split."""

    name: str
    X: np.ndarray
    binary_y: np.ndarray
    multiclass_y: np.ndarray
    episode_ids: np.ndarray
    step_indices: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.X.shape[0])

    @property
    def episode_count(self) -> int:
        return int(np.unique(self.episode_ids).size)


@dataclass(frozen=True)
class Phase5DevelopmentDataset:
    """TRAIN/VALIDATION-only monitoring dataset."""

    train: MonitoringDevelopmentSplit
    validation: MonitoringDevelopmentSplit
    feature_names: tuple[str, ...] = MODEL_FEATURES
    binary_target_name: str = BINARY_TARGET
    multiclass_target_name: str = MULTICLASS_TARGET

    def split(
        self,
        name: str,
    ) -> MonitoringDevelopmentSplit:
        """Return a permitted development split."""

        if name == "test":
            raise ValueError(
                "Phase 5 TEST payload is locked and unavailable "
                "through the development loader."
            )

        if name == "train":
            return self.train

        if name == "validation":
            return self.validation

        raise ValueError(
            "Phase 5 development split must be 'train' "
            "or 'validation'."
        )


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _resolve_repository_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return _REPO_ROOT / path


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _load_and_validate_protocol(
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)

    if protocol.get("phase") != "5":
        raise ValueError(
            "Expected frozen Phase 5 protocol."
        )

    if protocol.get("protocol_state") != "FROZEN":
        raise ValueError(
            "Phase 5 protocol must be FROZEN."
        )

    if protocol.get("implementation_started") is not False:
        raise ValueError(
            "Frozen Phase 5 protocol has unexpected "
            "implementation_started state."
        )

    test_discipline = protocol.get(
        "test_discipline",
        {},
    )

    if (
        test_discipline.get(
            "development_access_after_protocol_freeze"
        )
        is not False
    ):
        raise ValueError(
            "Phase 5 TEST development lock is not frozen."
        )

    if tuple(protocol.get("raw_features", ())) != RAW_FEATURES:
        raise ValueError(
            "Phase 5 raw-feature contract drift detected."
        )

    if (
        tuple(protocol.get("model_feature_order", ()))
        != MODEL_FEATURES
    ):
        raise ValueError(
            "Phase 5 model-feature order drift detected."
        )

    targets = protocol.get(
        "targets",
        {},
    )

    if targets.get("binary") != BINARY_TARGET:
        raise ValueError(
            "Phase 5 binary target contract drift detected."
        )

    if (
        targets.get("active_multiclass")
        != MULTICLASS_TARGET
    ):
        raise ValueError(
            "Phase 5 multiclass target contract drift detected."
        )

    return protocol


def _load_and_validate_feature_manifest(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    record = protocol[
        "upstream"
    ][
        "phase3_feature_manifest"
    ]

    path = _resolve_repository_path(
        record["path"]
    )

    expected_hash = record["sha256"]
    actual_hash = _sha256(path)

    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen Phase 3 monitoring feature-manifest "
            "hash mismatch."
        )

    manifest = _load_json(path)

    if tuple(manifest.get("features", ())) != RAW_FEATURES:
        raise ValueError(
            "Phase 3 / Phase 5 raw-feature contract mismatch."
        )

    if manifest.get("binary_target") != BINARY_TARGET:
        raise ValueError(
            "Phase 3 / Phase 5 binary-target mismatch."
        )

    if (
        manifest.get("multiclass_target")
        != MULTICLASS_TARGET
    ):
        raise ValueError(
            "Phase 3 / Phase 5 multiclass-target mismatch."
        )

    protected = set(
        manifest.get(
            "protected_ground_truth",
            (),
        )
    )

    overlap = protected.intersection(
        RAW_FEATURES
    )

    if overlap:
        raise ValueError(
            "Protected-ground-truth leakage detected in "
            f"raw features: {sorted(overlap)}"
        )

    group_columns = set(
        manifest.get(
            "group_columns",
            (),
        )
    )

    if group_columns.intersection(
        RAW_FEATURES
    ):
        raise ValueError(
            "Grouping metadata may not be predictive features."
        )

    if (
        manifest.get("ordering_column")
        in RAW_FEATURES
    ):
        raise ValueError(
            "Ordering metadata may not be a predictive feature."
        )

    return manifest


def _parse_boolean(
    value: str,
) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid boolean target value: {value!r}"
    )


def _feature_vector(
    row: dict[str, str],
) -> tuple[float, ...]:
    raw = tuple(
        float(row[name])
        for name in RAW_FEATURES
    )

    if not np.all(
        np.isfinite(
            np.asarray(
                raw,
                dtype=np.float64,
            )
        )
    ):
        raise ValueError(
            "Non-finite Phase 5 predictive feature encountered."
        )

    nominal_power = raw[0]
    target_pressure = raw[1]
    nominal_flow = raw[2]

    if nominal_power <= 0.0:
        raise ValueError(
            "nominal_absorbed_power_W must be positive."
        )

    if target_pressure <= 0.0:
        raise ValueError(
            "target_pressure_mTorr must be positive."
        )

    if nominal_flow <= 0.0:
        raise ValueError(
            "nominal_flow_sccm must be positive."
        )

    measured_power = raw[3]
    measured_flow = raw[4]
    measured_pressure = raw[5]

    derived = (
        measured_power / nominal_power - 1.0,
        measured_flow / nominal_flow - 1.0,
        measured_pressure / target_pressure - 1.0,
    )

    vector = raw + derived

    if not np.all(
        np.isfinite(
            np.asarray(
                vector,
                dtype=np.float64,
            )
        )
    ):
        raise ValueError(
            "Non-finite Phase 5 derived feature encountered."
        )

    return vector


def _make_read_only(
    array: np.ndarray,
) -> np.ndarray:
    array.setflags(
        write=False
    )
    return array


def _finalize_split(
    *,
    name: str,
    feature_rows: list[tuple[float, ...]],
    binary_values: list[bool],
    multiclass_values: list[str],
    episode_values: list[str],
    step_values: list[int],
    protocol: dict[str, Any],
) -> MonitoringDevelopmentSplit:
    X = np.asarray(
        feature_rows,
        dtype=np.float64,
    )

    binary_y = np.asarray(
        binary_values,
        dtype=np.bool_,
    )

    multiclass_y = np.asarray(
        multiclass_values,
        dtype=str,
    )

    episode_ids = np.asarray(
        episode_values,
        dtype=str,
    )

    step_indices = np.asarray(
        step_values,
        dtype=np.int64,
    )

    if X.ndim != 2:
        raise ValueError(
            f"{name} feature matrix must be two-dimensional."
        )

    if X.shape[1] != len(MODEL_FEATURES):
        raise ValueError(
            f"{name} feature matrix has wrong feature width."
        )

    row_count = X.shape[0]

    if not (
        binary_y.shape
        == multiclass_y.shape
        == episode_ids.shape
        == step_indices.shape
        == (row_count,)
    ):
        raise ValueError(
            f"{name} metadata/target shape mismatch."
        )

    expected = protocol[
        "splits"
    ][
        name
    ]

    if row_count != int(
        expected["rows"]
    ):
        raise ValueError(
            f"{name} row-count mismatch."
        )

    unique_episodes = np.unique(
        episode_ids
    )

    if unique_episodes.size != int(
        expected["episodes"]
    ):
        raise ValueError(
            f"{name} episode-count mismatch."
        )

    expected_steps = int(
        protocol[
            "upstream"
        ][
            "phase3_monitoring_dataset"
        ][
            "steps_per_episode"
        ]
    )

    expected_sequence = list(
        range(expected_steps)
    )

    for episode_id in unique_episodes:
        mask = episode_ids == episode_id

        steps = step_indices[
            mask
        ].tolist()

        if steps != expected_sequence:
            raise ValueError(
                "Episode ordering/step contract mismatch for "
                f"{episode_id!r} in {name}."
            )

    active_classes = set(
        protocol[
            "targets"
        ][
            "active_classes"
        ]
    )

    for active, family in zip(
        binary_y.tolist(),
        multiclass_y.tolist(),
        strict=True,
    ):
        if active:
            if family not in active_classes:
                raise ValueError(
                    "Active fault row has invalid active fault "
                    f"family: {family!r}"
                )
        elif family != "none":
            raise ValueError(
                "Inactive fault row must have "
                "active_fault_family == 'none'."
            )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            f"{name} feature matrix contains non-finite values."
        )

    return MonitoringDevelopmentSplit(
        name=name,
        X=_make_read_only(X),
        binary_y=_make_read_only(binary_y),
        multiclass_y=_make_read_only(multiclass_y),
        episode_ids=_make_read_only(episode_ids),
        step_indices=_make_read_only(step_indices),
    )


def load_phase5_development_dataset(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    dataset_path: str | Path | None = None,
) -> Phase5DevelopmentDataset:
    """Load leakage-safe TRAIN and VALIDATION monitoring data.

    TEST rows are identified only from split metadata and are skipped before
    predictive features or targets are interpreted.
    """

    protocol_file = Path(
        protocol_path
    )

    if not protocol_file.is_absolute():
        protocol_file = _resolve_repository_path(
            protocol_file
        )

    protocol = _load_and_validate_protocol(
        protocol_file
    )

    _load_and_validate_feature_manifest(
        protocol
    )

    dataset_record = protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ]

    if dataset_path is None:
        dataset_file = _resolve_repository_path(
            dataset_record["path"]
        )
    else:
        dataset_file = Path(
            dataset_path
        )

        if not dataset_file.is_absolute():
            dataset_file = _resolve_repository_path(
                dataset_file
            )

    expected_hash = dataset_record[
        "sha256"
    ]

    actual_hash = _sha256(
        dataset_file
    )

    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen Phase 3 monitoring dataset hash mismatch."
        )

    required_columns = set(
        RAW_FEATURES
    ).union(
        {
            BINARY_TARGET,
            MULTICLASS_TARGET,
            "episode_id",
            "split",
            "step_index",
        }
    )

    builders: dict[
        str,
        dict[str, list[Any]],
    ] = {
        name: {
            "features": [],
            "binary": [],
            "multiclass": [],
            "episodes": [],
            "steps": [],
        }
        for name in _ALLOWED_DEVELOPMENT_SPLITS
    }

    test_rows_skipped = 0

    with dataset_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Monitoring CSV header is missing."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Monitoring CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            # Critical TEST boundary:
            # inspect split metadata first and do not interpret
            # any TEST feature or target payload.
            split_name = row[
                "split"
            ].strip()

            if split_name == "test":
                test_rows_skipped += 1
                continue

            if (
                split_name
                not in _ALLOWED_DEVELOPMENT_SPLITS
            ):
                raise ValueError(
                    "Unexpected monitoring split: "
                    f"{split_name!r}"
                )

            builder = builders[
                split_name
            ]

            builder[
                "features"
            ].append(
                _feature_vector(
                    row
                )
            )

            builder[
                "binary"
            ].append(
                _parse_boolean(
                    row[
                        BINARY_TARGET
                    ]
                )
            )

            family = row[
                MULTICLASS_TARGET
            ].strip()

            if not family:
                raise ValueError(
                    "Empty active_fault_family encountered."
                )

            builder[
                "multiclass"
            ].append(
                family
            )

            episode_id = row[
                "episode_id"
            ].strip()

            if not episode_id:
                raise ValueError(
                    "Empty episode_id encountered."
                )

            builder[
                "episodes"
            ].append(
                episode_id
            )

            builder[
                "steps"
            ].append(
                int(
                    row[
                        "step_index"
                    ]
                )
            )

    expected_test_rows = int(
        protocol[
            "splits"
        ][
            "test"
        ][
            "rows"
        ]
    )

    if test_rows_skipped != expected_test_rows:
        raise ValueError(
            "TEST split row-count metadata mismatch while "
            "enforcing the development lock."
        )

    train = _finalize_split(
        name="train",
        feature_rows=builders["train"]["features"],
        binary_values=builders["train"]["binary"],
        multiclass_values=builders["train"]["multiclass"],
        episode_values=builders["train"]["episodes"],
        step_values=builders["train"]["steps"],
        protocol=protocol,
    )

    validation = _finalize_split(
        name="validation",
        feature_rows=builders["validation"]["features"],
        binary_values=builders["validation"]["binary"],
        multiclass_values=builders["validation"]["multiclass"],
        episode_values=builders["validation"]["episodes"],
        step_values=builders["validation"]["steps"],
        protocol=protocol,
    )

    train_episodes = set(
        train.episode_ids.tolist()
    )

    validation_episodes = set(
        validation.episode_ids.tolist()
    )

    overlap = train_episodes.intersection(
        validation_episodes
    )

    if overlap:
        raise ValueError(
            "Episode leakage detected between TRAIN and "
            f"VALIDATION: {sorted(overlap)}"
        )

    return Phase5DevelopmentDataset(
        train=train,
        validation=validation,
    )
