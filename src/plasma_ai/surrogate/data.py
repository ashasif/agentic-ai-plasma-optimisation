"""Controlled Phase 4 surrogate-dataset loading.

The loader enforces the frozen Phase 4 feature, target, dataset-hash,
and split contracts.

TEST targets are withheld by default. Access requires an explicit
``include_test_targets=True`` argument intended for the locked Phase 4F
evaluation only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_PROTOCOL_PATH = Path(
    "configs/phase4/surrogate_protocol.json"
)


@dataclass(frozen=True)
class SurrogateSplit:
    """One frozen dataset split."""

    X: np.ndarray
    y: np.ndarray | None


@dataclass(frozen=True)
class SurrogateDataset:
    """Phase 4 modelling arrays and frozen metadata."""

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    splits: Mapping[str, SurrogateSplit]

    def split(self, name: str) -> SurrogateSplit:
        """Return one named frozen split."""
        try:
            return self.splits[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown split {name!r}. "
                f"Available splits: {tuple(self.splits)}"
            ) from exc


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    source = Path(path)
    digest = hashlib.sha256()

    with source.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_surrogate_protocol(
    path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict:
    """Load and minimally validate the frozen Phase 4 protocol."""
    source = Path(path)

    protocol = json.loads(
        source.read_text(encoding="utf-8")
    )

    if protocol.get("status") != "frozen":
        raise ValueError(
            "Phase 4 surrogate protocol is not frozen."
        )

    if protocol.get("phase") != "4A":
        raise ValueError(
            "Unexpected Phase 4 protocol phase marker."
        )

    return protocol


def _validate_dataset_identity(
    protocol: dict,
) -> Path:
    dataset = protocol["datasets"]["base"]
    path = Path(dataset["path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen Phase 3 base dataset not found: {path}"
        )

    observed_sha256 = file_sha256(path)
    expected_sha256 = dataset["sha256"]

    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Frozen Phase 3 base dataset SHA-256 mismatch: "
            f"expected {expected_sha256}, "
            f"observed {observed_sha256}."
        )

    return path


def _as_finite_float(
    value: str,
    *,
    column: str,
    row_number: int,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric value in {column!r} "
            f"at CSV row {row_number}."
        ) from exc

    if not np.isfinite(parsed):
        raise ValueError(
            f"Non-finite value in {column!r} "
            f"at CSV row {row_number}."
        )

    return parsed


def load_phase4_dataset(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    *,
    include_test_targets: bool = False,
) -> SurrogateDataset:
    """Load the frozen Phase 3 base dataset for Phase 4.

    TRAIN and VALIDATION targets are loaded normally.

    TEST target values are deliberately withheld unless
    ``include_test_targets=True`` is explicitly requested.
    """
    protocol = load_surrogate_protocol(
        protocol_path
    )

    dataset_path = _validate_dataset_identity(
        protocol
    )

    features = tuple(
        protocol["feature_contract"]["features"]
    )

    targets = tuple(
        protocol[
            "target_contract"
        ]["primary_targets"]
    )

    split_column = protocol[
        "datasets"
    ]["base"]["split_column"]

    expected_counts = {
        name: int(count)
        for name, count in protocol[
            "datasets"
        ]["base"]["splits"].items()
    }

    expected_splits = tuple(
        expected_counts.keys()
    )

    feature_rows: dict[str, list[list[float]]] = {
        split: []
        for split in expected_splits
    }

    target_rows: dict[str, list[list[float]]] = {
        split: []
        for split in expected_splits
    }

    with dataset_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        fieldnames = tuple(
            reader.fieldnames or ()
        )

        required_columns = {
            split_column,
            *features,
            *targets,
        }

        missing = sorted(
            required_columns.difference(fieldnames)
        )

        if missing:
            raise ValueError(
                "Frozen base dataset is missing required "
                f"columns: {missing}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            split = row[split_column]

            if split not in expected_counts:
                raise ValueError(
                    f"Unexpected split {split!r} "
                    f"at CSV row {row_number}."
                )

            feature_rows[split].append([
                _as_finite_float(
                    row[name],
                    column=name,
                    row_number=row_number,
                )
                for name in features
            ])

            if (
                split != "test"
                or include_test_targets
            ):
                target_rows[split].append([
                    _as_finite_float(
                        row[name],
                        column=name,
                        row_number=row_number,
                    )
                    for name in targets
                ])

    observed_counts = {
        split: len(feature_rows[split])
        for split in expected_splits
    }

    if observed_counts != expected_counts:
        raise ValueError(
            "Frozen split-count mismatch: "
            f"expected {expected_counts}, "
            f"observed {observed_counts}."
        )

    splits: dict[str, SurrogateSplit] = {}

    for split in expected_splits:
        X = np.asarray(
            feature_rows[split],
            dtype=np.float64,
        )

        if split == "test" and not include_test_targets:
            y = None
        else:
            y = np.asarray(
                target_rows[split],
                dtype=np.float64,
            )

        splits[split] = SurrogateSplit(
            X=X,
            y=y,
        )

    return SurrogateDataset(
        feature_names=features,
        target_names=targets,
        splits=splits,
    )
