"""Tests for the Phase 5B leakage-safe monitoring loader."""

from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pytest

from plasma_ai.monitoring.data import (
    BINARY_TARGET,
    DEFAULT_PROTOCOL_PATH,
    MODEL_FEATURES,
    MULTICLASS_TARGET,
    RAW_FEATURES,
    load_phase5_development_dataset,
)


def _frozen_protocol():
    return json.loads(
        DEFAULT_PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )


def _sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def _write_csv(
    path,
    rows,
):
    fields = [
        "episode_id",
        "split",
        "step_index",
        *RAW_FEATURES,
        BINARY_TARGET,
        MULTICLASS_TARGET,
    ]

    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def _development_row(
    *,
    episode_id,
    split,
    active,
    family,
):
    return {
        "episode_id": episode_id,
        "split": split,
        "step_index": "0",
        "nominal_absorbed_power_W": "50.0",
        "target_pressure_mTorr": "30.0",
        "nominal_flow_sccm": "20.0",
        "measured_absorbed_power_W": "49.0",
        "measured_flow_sccm": "20.2",
        "measured_pressure_mTorr": "30.3",
        "fault_effect_active": (
            "true" if active else "false"
        ),
        "active_fault_family": family,
    }


def _write_fixture_protocol(
    tmp_path,
    dataset_path,
):
    protocol = deepcopy(
        _frozen_protocol()
    )

    protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ][
        "path"
    ] = str(
        Path(dataset_path).resolve()
    )

    protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ][
        "sha256"
    ] = _sha256(
        dataset_path
    )

    protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ][
        "rows"
    ] = 3

    protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ][
        "episodes"
    ] = 3

    protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ][
        "steps_per_episode"
    ] = 1

    protocol["splits"] = {
        "train": {
            "episodes": 1,
            "rows": 1,
        },
        "validation": {
            "episodes": 1,
            "rows": 1,
        },
        "test": {
            "episodes": 1,
            "rows": 1,
        },
        "assignment_unit": "episode",
        "episode_overlap_permitted": False,
    }

    protocol_path = (
        tmp_path
        / "phase5_protocol.json"
    )

    protocol_path.write_text(
        json.dumps(
            protocol,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return protocol_path


def test_frozen_phase5_feature_and_target_contract():
    protocol = _frozen_protocol()

    assert protocol["protocol_state"] == "FROZEN"

    assert tuple(
        protocol["raw_features"]
    ) == RAW_FEATURES

    assert tuple(
        protocol["model_feature_order"]
    ) == MODEL_FEATURES

    assert (
        protocol["targets"]["binary"]
        == BINARY_TARGET
    )

    assert (
        protocol["targets"]["active_multiclass"]
        == MULTICLASS_TARGET
    )

    assert (
        protocol["test_discipline"][
            "development_access_after_protocol_freeze"
        ]
        is False
    )


def test_real_development_loader_has_frozen_shapes_and_test_lock():
    dataset = load_phase5_development_dataset()

    assert dataset.feature_names == MODEL_FEATURES

    assert dataset.train.X.shape == (
        2048,
        9,
    )

    assert dataset.validation.X.shape == (
        1024,
        9,
    )

    assert dataset.train.episode_count == 32
    assert dataset.validation.episode_count == 16

    with pytest.raises(
        ValueError,
        match="TEST payload is locked",
    ):
        dataset.split("test")


def test_real_development_loader_preserves_episode_isolation_and_order():
    dataset = load_phase5_development_dataset()

    train_episodes = set(
        dataset.train.episode_ids.tolist()
    )

    validation_episodes = set(
        dataset.validation.episode_ids.tolist()
    )

    assert train_episodes.isdisjoint(
        validation_episodes
    )

    for split in (
        dataset.train,
        dataset.validation,
    ):
        for episode_id in np.unique(
            split.episode_ids
        ):
            steps = split.step_indices[
                split.episode_ids
                == episode_id
            ]

            np.testing.assert_array_equal(
                steps,
                np.arange(
                    64,
                    dtype=np.int64,
                ),
            )


def test_real_development_loader_builds_exact_residual_features():
    dataset = load_phase5_development_dataset()

    for split in (
        dataset.train,
        dataset.validation,
    ):
        X = split.X

        np.testing.assert_allclose(
            X[:, 6],
            X[:, 3] / X[:, 0] - 1.0,
            rtol=0.0,
            atol=1e-15,
        )

        np.testing.assert_allclose(
            X[:, 7],
            X[:, 4] / X[:, 2] - 1.0,
            rtol=0.0,
            atol=1e-15,
        )

        np.testing.assert_allclose(
            X[:, 8],
            X[:, 5] / X[:, 1] - 1.0,
            rtol=0.0,
            atol=1e-15,
        )


def test_real_development_arrays_are_finite_and_read_only():
    dataset = load_phase5_development_dataset()

    for split in (
        dataset.train,
        dataset.validation,
    ):
        assert np.all(
            np.isfinite(
                split.X
            )
        )

        assert split.X.flags.writeable is False
        assert split.binary_y.flags.writeable is False
        assert split.multiclass_y.flags.writeable is False
        assert split.episode_ids.flags.writeable is False
        assert split.step_indices.flags.writeable is False


def test_model_features_do_not_include_protected_ground_truth():
    protocol = _frozen_protocol()

    feature_record = protocol[
        "upstream"
    ][
        "phase3_feature_manifest"
    ]

    feature_path = (
        DEFAULT_PROTOCOL_PATH.parents[2]
        / feature_record["path"]
    )

    manifest = json.loads(
        feature_path.read_text(
            encoding="utf-8"
        )
    )

    protected = set(
        manifest[
            "protected_ground_truth"
        ]
    )

    assert protected.isdisjoint(
        MODEL_FEATURES
    )

    assert "episode_id" not in MODEL_FEATURES
    assert "split" not in MODEL_FEATURES
    assert "step_index" not in MODEL_FEATURES


def test_test_payload_is_skipped_before_feature_or_target_parsing(
    tmp_path,
):
    dataset_path = (
        tmp_path
        / "monitoring.csv"
    )

    train = _development_row(
        episode_id="train-001",
        split="train",
        active=False,
        family="none",
    )

    validation = _development_row(
        episode_id="validation-001",
        split="validation",
        active=True,
        family="flow_delivery",
    )

    poisoned_test = {
        "episode_id": "DO_NOT_INTERPRET",
        "split": "test",
        "step_index": "DO_NOT_INTERPRET",
        "nominal_absorbed_power_W": "DO_NOT_INTERPRET",
        "target_pressure_mTorr": "DO_NOT_INTERPRET",
        "nominal_flow_sccm": "DO_NOT_INTERPRET",
        "measured_absorbed_power_W": "DO_NOT_INTERPRET",
        "measured_flow_sccm": "DO_NOT_INTERPRET",
        "measured_pressure_mTorr": "DO_NOT_INTERPRET",
        "fault_effect_active": "DO_NOT_INTERPRET",
        "active_fault_family": "DO_NOT_INTERPRET",
    }

    _write_csv(
        dataset_path,
        [
            train,
            validation,
            poisoned_test,
        ],
    )

    protocol_path = _write_fixture_protocol(
        tmp_path,
        dataset_path,
    )

    dataset = load_phase5_development_dataset(
        protocol_path=protocol_path,
    )

    assert dataset.train.row_count == 1
    assert dataset.validation.row_count == 1

    with pytest.raises(
        ValueError,
        match="TEST payload is locked",
    ):
        dataset.split("test")


def test_loader_rejects_dataset_hash_drift(
    tmp_path,
):
    dataset_path = (
        tmp_path
        / "monitoring.csv"
    )

    rows = [
        _development_row(
            episode_id="train-001",
            split="train",
            active=False,
            family="none",
        ),
        _development_row(
            episode_id="validation-001",
            split="validation",
            active=True,
            family="flow_delivery",
        ),
        {
            **_development_row(
                episode_id="test-001",
                split="test",
                active=False,
                family="none",
            )
        },
    ]

    _write_csv(
        dataset_path,
        rows,
    )

    protocol_path = _write_fixture_protocol(
        tmp_path,
        dataset_path,
    )

    with dataset_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\n")

    with pytest.raises(
        ValueError,
        match="dataset hash mismatch",
    ):
        load_phase5_development_dataset(
            protocol_path=protocol_path,
        )
