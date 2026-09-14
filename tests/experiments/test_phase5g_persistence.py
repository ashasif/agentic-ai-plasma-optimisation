from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

import plasma_ai.monitoring.phase5g_persistence as persistence
from plasma_ai.monitoring.data import (
    RAW_FEATURES,
    _feature_vector,
)


def test_phase5g_protocol_is_frozen_and_test_access_is_prohibited():
    protocol = persistence.load_phase5g_protocol()

    assert protocol["phase"] == "5G"
    assert protocol["protocol_state"] == "FROZEN"

    assert (
        protocol["governance"][
            "phase5_test_permanently_consumed"
        ]
        is True
    )

    assert (
        protocol["governance"][
            "phase5_test_dataset_access_permitted"
        ]
        is False
    )

    assert (
        protocol["purpose"][
            "model_retraining"
        ]
        is False
    )

    assert (
        protocol["purpose"][
            "model_reserialization"
        ]
        is False
    )


def test_phase5g_feature_builder_matches_frozen_feature_vector_exactly():
    raw = np.asarray(
        [
            [
                50.0,
                25.0,
                20.0,
                45.0,
                18.6,
                27.5,
            ],
            [
                80.0,
                45.0,
                20.0,
                80.0,
                20.0,
                40.5,
            ],
        ],
        dtype=np.float64,
    )

    actual = persistence.build_phase5g_model_features(
        raw
    )

    expected_rows = []

    for row in raw:
        mapping = {
            name:
                str(
                    value
                )
            for name, value in zip(
                RAW_FEATURES,
                row,
                strict=True,
            )
        }

        expected_rows.append(
            _feature_vector(
                mapping
            )
        )

    expected = np.asarray(
        expected_rows,
        dtype=np.float64,
    )

    assert actual.shape == (
        2,
        9,
    )

    assert np.array_equal(
        actual,
        expected,
    )

    single = persistence.build_phase5g_model_features(
        raw[
            0
        ]
    )

    assert single.shape == (
        1,
        9,
    )

    assert np.array_equal(
        single[
            0
        ],
        expected[
            0
        ],
    )

    assert actual.flags.writeable is False

    with pytest.raises(
        ValueError
    ):
        persistence.build_phase5g_model_features(
            [1.0, 2.0]
        )

    with pytest.raises(
        ValueError
    ):
        persistence.build_phase5g_model_features(
            np.empty(
                (
                    0,
                    6,
                )
            )
        )

    invalid = raw.copy()
    invalid[
        0,
        3,
    ] = np.nan

    with pytest.raises(
        ValueError
    ):
        persistence.build_phase5g_model_features(
            invalid
        )

    invalid = raw.copy()
    invalid[
        0,
        0,
    ] = 0.0

    with pytest.raises(
        ValueError
    ):
        persistence.build_phase5g_model_features(
            invalid
        )


class _FakeDetector:
    threshold = 0.40

    def predict_active_probability(
        self,
        X,
    ):
        values = np.asarray(
            [
                0.20,
                0.40,
                0.90,
            ],
            dtype=np.float64,
        )

        return values[
            : X.shape[0]
        ]

    def predict(
        self,
        X,
    ):
        return (
            self.predict_active_probability(
                X
            )
            >= self.threshold
        )


class _FakeDiagnoser:
    def __init__(
        self,
    ):
        self.calls = []

    def predict(
        self,
        X,
    ):
        self.calls.append(
            np.array(
                X,
                copy=True,
            )
        )

        values = np.asarray(
            [
                "power_coupling",
                "pressure_path_anomaly",
            ],
            dtype=str,
        )

        return values[
            : X.shape[0]
        ]


def test_loaded_monitoring_runs_diagnoser_only_on_detector_positive_rows():
    diagnoser = _FakeDiagnoser()

    loaded = persistence.LoadedPhase5GMonitoring(
        detector=_FakeDetector(),
        diagnoser=diagnoser,
        manifest={
            "prediction_contract": {
                "inactive_state":
                    "none",

                "end_to_end_states": [
                    "none",
                    "flow_delivery",
                    "power_coupling",
                    "pressure_path_anomaly",
                ],
            }
        },
    )

    raw = np.asarray(
        [
            [
                50.0,
                30.0,
                20.0,
                50.0,
                20.0,
                30.0,
            ],
            [
                50.0,
                30.0,
                20.0,
                45.0,
                20.0,
                30.0,
            ],
            [
                50.0,
                30.0,
                20.0,
                50.0,
                18.6,
                30.0,
            ],
        ]
    )

    result = loaded.predict(
        raw
    )

    assert np.array_equal(
        result.active_probability,
        np.asarray(
            [
                0.20,
                0.40,
                0.90,
            ]
        ),
    )

    assert np.array_equal(
        result.fault_active,
        np.asarray(
            [
                False,
                True,
                True,
            ]
        ),
    )

    assert np.array_equal(
        result.diagnostic_state,
        np.asarray(
            [
                "none",
                "power_coupling",
                "pressure_path_anomaly",
            ]
        ),
    )

    assert len(
        diagnoser.calls
    ) == 1

    assert diagnoser.calls[
        0
    ].shape == (
        2,
        9,
    )

    assert result.active_probability.flags.writeable is False
    assert result.fault_active.flags.writeable is False
    assert result.diagnostic_state.flags.writeable is False


def test_verified_pickle_bytes_reject_tampered_artifact(tmp_path):
    path = tmp_path / "model.pkl"

    path.write_bytes(
        b"tampered-model-bytes"
    )

    expected = hashlib.sha256(
        b"different-model-bytes"
    ).hexdigest()

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        persistence._read_verified_pickle_bytes(
            path,
            expected_sha256=expected,
        )


def test_manifest_writer_refuses_existing_output_and_dirty_repository(
    tmp_path,
    monkeypatch,
):
    existing = tmp_path / "existing_manifest.json"

    existing.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError
    ):
        persistence.write_phase5g_manifest(
            output_path=existing
        )

    destination = tmp_path / "new_manifest.json"

    monkeypatch.setattr(
        persistence,
        "_git_state",
        lambda: (
            "synthetic-dirty-commit",
            False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="clean committed repository",
    ):
        persistence.write_phase5g_manifest(
            output_path=destination
        )

    assert not destination.exists()


def test_manifest_roundtrip_loads_exact_frozen_models_and_predicts_synthetic_rows(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "monitoring_manifest.json"

    monkeypatch.setattr(
        persistence,
        "_git_state",
        lambda: (
            "synthetic-clean-test-commit",
            True,
        ),
    )

    written = persistence.write_phase5g_manifest(
        output_path=destination
    )

    assert written == destination
    assert destination.exists()

    loaded = persistence.load_phase5g_monitoring(
        manifest_path=destination
    )

    assert (
        loaded.detector.candidate_id
        == "ERD4_extra_trees_threshold_040"
    )

    assert loaded.detector.threshold == 0.40
    assert loaded.detector.fit_rows == 3072

    assert (
        loaded.diagnoser.candidate_id
        == "ERG3_extra_trees"
    )

    assert loaded.diagnoser.fit_rows == 1451

    synthetic = np.asarray(
        [
            [
                50.0,
                30.0,
                20.0,
                50.0,
                20.0,
                30.0,
            ],
            [
                50.0,
                30.0,
                20.0,
                45.0,
                20.0,
                30.0,
            ],
            [
                80.0,
                45.0,
                20.0,
                80.0,
                18.6,
                49.5,
            ],
        ],
        dtype=np.float64,
    )

    result = loaded.predict(
        synthetic
    )

    assert result.active_probability.shape == (
        3,
    )

    assert result.fault_active.shape == (
        3,
    )

    assert result.diagnostic_state.shape == (
        3,
    )

    assert np.all(
        np.isfinite(
            result.active_probability
        )
    )

    assert set(
        result.diagnostic_state.tolist()
    ).issubset(
        {
            "none",
            "flow_delivery",
            "power_coupling",
            "pressure_path_anomaly",
        }
    )


def test_runtime_mismatch_is_rejected_before_model_deserialization(
    tmp_path,
    monkeypatch,
):
    protocol = persistence.load_phase5g_protocol()

    manifest = persistence._build_manifest(
        protocol,
        source_git_commit="synthetic-test-commit",
        repository_clean_state=True,
        artifact_creation_timestamp_utc="2026-09-14T00:00:00Z",
    )

    manifest[
        "provenance"
    ][
        "python_version"
    ] = "0.0.0"

    path = tmp_path / "bad_runtime_manifest.json"

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_pickle_deserialization_occurs(
        payload,
    ):
        raise AssertionError(
            "pickle.loads must not run after runtime mismatch."
        )

    monkeypatch.setattr(
        persistence.pickle,
        "loads",
        fail_if_pickle_deserialization_occurs,
    )

    with pytest.raises(
        RuntimeError,
        match="Persisted Phase 5G scientific runtime mismatch",
    ):
        persistence.load_phase5g_monitoring(
            manifest_path=path
        )
