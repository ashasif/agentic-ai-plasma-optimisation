"""Tests for the Phase 5F one-time locked TEST evaluator.

Real Phase 5 TEST predictive rows are not loaded by this test module.
Loader behaviour is qualified with a temporary synthetic CSV. The real
preflight test verifies only frozen JSON/model artifacts and does not open
the monitoring dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import plasma_ai.monitoring.phase5f_evaluation as phase5f

from plasma_ai.monitoring.data import (
    MODEL_FEATURES,
    MonitoringDevelopmentSplit,
)


class FakeDetector:
    candidate_id = (
        "ERD4_extra_trees_threshold_040"
    )

    fit_rows = 3072
    threshold = 0.40

    _probabilities = np.asarray(
        [
            0.10,
            0.90,
            0.80,
            0.70,
            0.20,
            0.85,
            0.30,
            0.10,
        ],
        dtype=np.float64,
    )

    def predict_active_probability(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(
            X[
                :,
                0,
            ],
            dtype=int,
        )

        return self._probabilities[
            indices
        ]

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return (
            self.predict_active_probability(
                X
            )
            >= self.threshold
        )


class FakeDiagnoser:
    candidate_id = (
        "ERG3_extra_trees"
    )

    fit_rows = 1451

    train_majority_class = (
        "pressure_path_anomaly"
    )

    _predictions = {
        0:
            "flow_delivery",

        1:
            "flow_delivery",

        2:
            "power_coupling",

        3:
            "flow_delivery",

        4:
            "pressure_path_anomaly",

        5:
            "pressure_path_anomaly",

        6:
            "pressure_path_anomaly",

        7:
            "pressure_path_anomaly",
    }

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(
            X[
                :,
                0,
            ],
            dtype=int,
        )

        return np.asarray(
            [
                self._predictions[
                    int(index)
                ]
                for index in indices
            ],
            dtype=str,
        )


def _write_synthetic_locked_dataset(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
]:
    protocol = json.loads(
        Path(
            "configs/phase5/phase5_protocol.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    dataset_path = (
        tmp_path
        / "synthetic_monitoring.csv"
    )

    fieldnames = (
        list(
            phase5f.RAW_FEATURES
        )
        + [
            phase5f.BINARY_TARGET,
            phase5f.MULTICLASS_TARGET,
            "episode_id",
            "split",
            "step_index",
        ]
    )

    def valid_row(
        *,
        episode_id: str,
        step_index: int,
        active: bool,
        family: str,
        measured_power: float,
        measured_flow: float,
        measured_pressure: float,
    ) -> dict[str, str]:
        return {
            "nominal_absorbed_power_W":
                "50.0",

            "target_pressure_mTorr":
                "30.0",

            "nominal_flow_sccm":
                "20.0",

            "measured_absorbed_power_W":
                str(
                    measured_power
                ),

            "measured_flow_sccm":
                str(
                    measured_flow
                ),

            "measured_pressure_mTorr":
                str(
                    measured_pressure
                ),

            phase5f.BINARY_TARGET:
                (
                    "true"
                    if active
                    else "false"
                ),

            phase5f.MULTICLASS_TARGET:
                family,

            "episode_id":
                episode_id,

            "split":
                "test",

            "step_index":
                str(
                    step_index
                ),
        }

    invalid_non_test = {
        name:
            "NOT_A_NUMBER"
        for name in phase5f.RAW_FEATURES
    }

    invalid_non_test.update(
        {
            phase5f.BINARY_TARGET:
                "NOT_A_BOOLEAN",

            phase5f.MULTICLASS_TARGET:
                "",

            "episode_id":
                "",

            "split":
                "train",

            "step_index":
                "NOT_AN_INTEGER",
        }
    )

    invalid_validation = dict(
        invalid_non_test
    )

    invalid_validation[
        "split"
    ] = "validation"

    rows = [
        invalid_non_test,
        invalid_validation,

        valid_row(
            episode_id="test_fixture_000",
            step_index=0,
            active=False,
            family="none",
            measured_power=50.0,
            measured_flow=20.0,
            measured_pressure=30.0,
        ),

        valid_row(
            episode_id="test_fixture_000",
            step_index=1,
            active=True,
            family="flow_delivery",
            measured_power=50.0,
            measured_flow=18.0,
            measured_pressure=30.0,
        ),

        valid_row(
            episode_id="test_fixture_000",
            step_index=2,
            active=True,
            family="power_coupling",
            measured_power=45.0,
            measured_flow=20.0,
            measured_pressure=30.0,
        ),

        valid_row(
            episode_id="test_fixture_001",
            step_index=0,
            active=False,
            family="none",
            measured_power=50.0,
            measured_flow=20.0,
            measured_pressure=30.0,
        ),

        valid_row(
            episode_id="test_fixture_001",
            step_index=1,
            active=True,
            family="pressure_sensor_bias",
            measured_power=50.0,
            measured_flow=20.0,
            measured_pressure=33.0,
        ),

        valid_row(
            episode_id="test_fixture_001",
            step_index=2,
            active=True,
            family="pumping_effectiveness",
            measured_power=50.0,
            measured_flow=20.0,
            measured_pressure=27.0,
        ),
    ]

    with dataset_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    dataset_hash = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()

    dataset_record = (
        protocol[
            "upstream"
        ][
            "phase3_monitoring_dataset"
        ]
    )

    dataset_record[
        "path"
    ] = str(
        dataset_path
    )

    dataset_record[
        "sha256"
    ] = dataset_hash

    dataset_record[
        "steps_per_episode"
    ] = 3

    protocol[
        "splits"
    ][
        "test"
    ][
        "rows"
    ] = 6

    protocol[
        "splits"
    ][
        "test"
    ][
        "episodes"
    ] = 2

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

    return (
        protocol_path,
        dataset_path,
    )


def _synthetic_test_split() -> MonitoringDevelopmentSplit:
    X = np.zeros(
        (
            8,
            len(
                MODEL_FEATURES
            ),
        ),
        dtype=np.float64,
    )

    X[
        :,
        0,
    ] = np.arange(
        8,
        dtype=np.float64,
    )

    binary = np.asarray(
        [
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
        ],
        dtype=np.bool_,
    )

    multiclass = np.asarray(
        [
            "none",
            "flow_delivery",
            "power_coupling",
            "none",
            "none",
            "pressure_sensor_bias",
            "pumping_effectiveness",
            "none",
        ],
        dtype=str,
    )

    episodes = np.asarray(
        [
            "episode_0",
            "episode_0",
            "episode_0",
            "episode_0",
            "episode_1",
            "episode_1",
            "episode_1",
            "episode_1",
        ],
        dtype=str,
    )

    steps = np.asarray(
        [
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
        ],
        dtype=np.int64,
    )

    return MonitoringDevelopmentSplit(
        name="test",
        X=X,
        binary_y=binary,
        multiclass_y=multiclass,
        episode_ids=episodes,
        step_indices=steps,
    )


def test_locked_test_loader_requires_explicit_unlock() -> None:
    with pytest.raises(
        RuntimeError,
        match="explicit",
    ):
        phase5f.load_phase5_locked_test_split()


def test_locked_test_loader_uses_only_test_payload(
    tmp_path: Path,
) -> None:
    (
        protocol_path,
        dataset_path,
    ) = _write_synthetic_locked_dataset(
        tmp_path
    )

    test = (
        phase5f.load_phase5_locked_test_split(
            unlock_test=True,
            protocol_path=protocol_path,
            dataset_path=dataset_path,
        )
    )

    assert test.name == "test"

    assert test.X.shape == (
        6,
        len(
            MODEL_FEATURES
        ),
    )

    assert test.row_count == 6
    assert test.episode_count == 2

    assert (
        test.binary_y.tolist()
        == [
            False,
            True,
            True,
            False,
            True,
            True,
        ]
    )

    assert (
        test.multiclass_y.tolist()
        == [
            "none",
            "flow_delivery",
            "power_coupling",
            "none",
            "pressure_sensor_bias",
            "pumping_effectiveness",
        ]
    )

    np.testing.assert_allclose(
        test.X[
            1,
            -3:,
        ],
        np.asarray(
            [
                0.0,
                -0.1,
                0.0,
            ]
        ),
    )


def test_locked_evaluator_reports_complete_synthetic_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = _synthetic_test_split()

    preflight = {
        "phase":
            "5F",

        "phase5f_unlock_qualified":
            True,

        "selected_detector":
            phase5f.EXPECTED_DETECTOR_ID,

        "selected_diagnoser":
            phase5f.EXPECTED_DIAGNOSER_ID,

        "detector_fit_rows":
            3072,

        "diagnoser_fit_rows":
            1451,

        "detector_threshold":
            0.40,

        "test_fit_rows":
            0,

        "test_consumed_before_evaluation":
            False,

        "frozen_hashes": {
            "phase5_protocol":
                "a",

            "phase5er_protocol":
                "b",

            "redevelopment_benchmark":
                "c",

            "redevelopment_selection":
                "d",

            "redevelopment_final_refit":
                "e",

            "fault_detector":
                "f",

            "fault_diagnoser":
                "g",
        },
    }

    monkeypatch.setattr(
        phase5f,
        "validate_phase5f_preflight",
        lambda **kwargs: preflight,
    )

    monkeypatch.setattr(
        phase5f,
        "load_phase5_locked_test_split",
        lambda **kwargs: synthetic,
    )

    detector = FakeDetector()
    diagnoser = FakeDiagnoser()

    def fake_load_pickle(
        path: str | Path,
    ):
        if "detector" in str(
            path
        ):
            return detector

        return diagnoser

    monkeypatch.setattr(
        phase5f,
        "_load_pickle",
        fake_load_pickle,
    )

    monkeypatch.setattr(
        phase5f,
        "EXPECTED_TEST_ROWS",
        8,
    )

    monkeypatch.setattr(
        phase5f,
        "EXPECTED_TEST_EPISODES",
        2,
    )

    payload = (
        phase5f.run_phase5f_locked_test_evaluation()
    )

    assert payload[
        "phase"
    ] == "5F"

    assert payload[
        "stage"
    ] == "one_time_locked_test_evaluation"

    assert payload[
        "irreversible_test_evaluation"
    ] is True

    assert payload[
        "data_usage"
    ][
        "test_rows"
    ] == 8

    assert payload[
        "data_usage"
    ][
        "test_targets_accessed"
    ] is True

    assert payload[
        "data_usage"
    ][
        "test_predictions_generated"
    ] is True

    assert payload[
        "detector"
    ][
        "candidate_id"
    ] == (
        "ERD4_extra_trees_threshold_040"
    )

    assert (
        "balanced_accuracy"
        in payload[
            "detector"
        ][
            "metrics"
        ]
    )

    assert (
        "summary"
        in payload[
            "detector"
        ][
            "sequence_diagnostics"
        ]
    )

    assert payload[
        "diagnoser"
    ][
        "candidate_id"
    ] == "ERG3_extra_trees"

    assert (
        "macro_f1"
        in payload[
            "diagnoser"
        ][
            "metrics"
        ]
    )

    assert payload[
        "secondary_end_to_end"
    ][
        "selection_metric"
    ] is False

    assert (
        payload[
            "interpretation"
        ][
            "test_is_final_generalisation_evidence"
        ]
        is True
    )

    assert (
        payload[
            "interpretation"
        ][
            "model_changes_permitted_from_test"
        ]
        is False
    )

    assert (
        payload[
            "test_state"
        ][
            "test_consumed"
        ]
        is True
    )


def test_writer_refuses_existing_result_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (
        tmp_path
        / "locked_test_evaluation.json"
    )

    destination.write_text(
        "{}\n",
        encoding="utf-8",
    )

    called = False

    def forbidden_run(
        **kwargs,
    ):
        nonlocal called

        called = True

        raise AssertionError(
            "Evaluation must not run."
        )

    monkeypatch.setattr(
        phase5f,
        "run_phase5f_locked_test_evaluation",
        forbidden_run,
    )

    with pytest.raises(
        FileExistsError,
    ):
        phase5f.write_phase5f_locked_test_evaluation(
            destination
        )

    assert called is False


def test_writer_persists_locked_result_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = (
        tmp_path
        / "locked_test_evaluation.json"
    )

    expected_payload = {
        "phase":
            "5F",

        "stage":
            "one_time_locked_test_evaluation",

        "irreversible_test_evaluation":
            True,

        "test_state": {
            "test_consumed":
                True,
        },
    }

    monkeypatch.setattr(
        phase5f,
        "run_phase5f_locked_test_evaluation",
        lambda **kwargs: expected_payload,
    )

    written = (
        phase5f.write_phase5f_locked_test_evaluation(
            destination
        )
    )

    assert written == destination

    assert json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    ) == expected_payload

    assert destination.read_bytes().endswith(
        b"\n"
    )

    with pytest.raises(
        FileExistsError,
    ):
        phase5f.write_phase5f_locked_test_evaluation(
            destination
        )


def test_real_preflight_accepts_frozen_refit_without_test_access() -> None:
    payload = (
        phase5f.validate_phase5f_preflight()
    )

    assert payload[
        "phase"
    ] == "5F"

    assert payload[
        "phase5f_unlock_qualified"
    ] is True

    assert payload[
        "selected_detector"
    ] == (
        "ERD4_extra_trees_threshold_040"
    )

    assert payload[
        "selected_diagnoser"
    ] == "ERG3_extra_trees"

    assert payload[
        "detector_fit_rows"
    ] == 3072

    assert payload[
        "diagnoser_fit_rows"
    ] == 1451

    assert payload[
        "test_fit_rows"
    ] == 0

    assert (
        payload[
            "test_consumed_before_evaluation"
        ]
        is False
    )
