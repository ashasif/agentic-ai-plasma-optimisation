"""Phase 5F one-time locked TEST evaluation.

This module is the only dedicated Phase 5 path that intentionally interprets
the frozen monitoring TEST predictive features and targets.

Development code remains TRAIN/VALIDATION-only. The locked TEST loader
requires an explicit unlock argument and is called only by the Phase 5F
evaluation path.

The persisted writer refuses to overwrite an existing TEST result before
evaluation begins. TEST evidence is final generalisation evidence only and
must not be used for model selection, threshold changes, retuning, or new
model-family development.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from plasma_ai.monitoring.data import (
    BINARY_TARGET,
    MODEL_FEATURES,
    MULTICLASS_TARGET,
    RAW_FEATURES,
    MonitoringDevelopmentSplit,
    _feature_vector,
    _finalize_split,
    _load_and_validate_feature_manifest,
    _load_and_validate_protocol,
    _parse_boolean,
    _resolve_repository_path,
    _sha256,
)

from plasma_ai.monitoring.redevelopment import (
    REDEVELOPMENT_CLASSES,
    detector_sequence_diagnostics,
    map_redevelopment_active_fault_labels,
    redevelopment_diagnosis_metrics,
    redevelopment_detector_metrics,
)


DEFAULT_PHASE5_PROTOCOL_PATH = Path(
    "configs/phase5/phase5_protocol.json"
)

DEFAULT_PHASE5ER_PROTOCOL_PATH = Path(
    "configs/phase5/phase5er_protocol.json"
)

DEFAULT_BENCHMARK_PATH = Path(
    "results/phase5/redevelopment_benchmark.json"
)

DEFAULT_SELECTION_PATH = Path(
    "results/phase5/redevelopment_selection.json"
)

DEFAULT_FINAL_REFIT_PATH = Path(
    "results/phase5/redevelopment_final_refit.json"
)

DEFAULT_DETECTOR_PATH = Path(
    "artifacts/phase5/fault_detector.pkl"
)

DEFAULT_DIAGNOSER_PATH = Path(
    "artifacts/phase5/fault_diagnoser.pkl"
)

DEFAULT_OUTPUT_PATH = Path(
    "results/phase5/locked_test_evaluation.json"
)


EXPECTED_PHASE5_PROTOCOL_SHA256 = (
    "70de30b7a5f2e36b567f8848ee0e6268"
    "33c25eb8285e7b4df9e63d5db77ddcec"
)

EXPECTED_PHASE5ER_PROTOCOL_SHA256 = (
    "8df07efea044580c7cfbdfc893c04076"
    "ba5fbfbd982b60d2fd5a37f186508bda"
)

EXPECTED_BENCHMARK_SHA256 = (
    "336a6b4af44a5b833c88ebdeeb2b6e7b"
    "10fac50cba9e86ded8548f427b6eee4e"
)

EXPECTED_SELECTION_SHA256 = (
    "74ea1818094f22dda82ab0c5f899e3df"
    "ebd64f3f2ebd90eb6267af9935c26b88"
)

EXPECTED_FINAL_REFIT_SHA256 = (
    "b04d4676b97235c2a23e00920e27935f"
    "96e4e2c2da1751145078f0b89d0a8e7a"
)

EXPECTED_DETECTOR_SHA256 = (
    "4a00e785d7e7f0a8429020399967cdd8"
    "2c18da275d58e5b83bc11707eeac8a6a"
)

EXPECTED_DIAGNOSER_SHA256 = (
    "2e9dea016137594c707ecd3c32116691"
    "cc90d0127133f11c70372c93e6fcca84"
)

EXPECTED_DETECTOR_ID = (
    "ERD4_extra_trees_threshold_040"
)

EXPECTED_DIAGNOSER_ID = (
    "ERG3_extra_trees"
)

EXPECTED_DETECTOR_FIT_ROWS = 3072
EXPECTED_DIAGNOSER_FIT_ROWS = 1451
EXPECTED_DETECTOR_THRESHOLD = 0.40

EXPECTED_TEST_ROWS = 1024
EXPECTED_TEST_EPISODES = 16
EXPECTED_STEPS_PER_EPISODE = 64

END_TO_END_CLASSES = (
    "none",
    "flow_delivery",
    "power_coupling",
    "pressure_path_anomaly",
)


def _resolve(
    value: str | Path,
) -> Path:
    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return _resolve_repository_path(
        path
    )


def _sha256_local(
    path: str | Path,
) -> str:
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def _verify_hash(
    path: str | Path,
    expected_hash: str,
    *,
    label: str,
) -> str:
    resolved = _resolve(
        path
    )

    actual = _sha256_local(
        resolved
    )

    if actual != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch."
        )

    return actual


def _load_json(
    path: str | Path,
) -> dict[str, Any]:
    return json.loads(
        _resolve(
            path
        ).read_text(
            encoding="utf-8"
        )
    )


def _load_pickle(
    path: str | Path,
) -> Any:
    with _resolve(
        path
    ).open(
        "rb"
    ) as handle:
        return pickle.load(
            handle
        )


def _value_counts(
    values: np.ndarray,
) -> dict[str, int]:
    labels, counts = np.unique(
        np.asarray(
            values,
            dtype=str,
        ),
        return_counts=True,
    )

    return {
        str(label): int(count)
        for label, count in zip(
            labels.tolist(),
            counts.tolist(),
            strict=True,
        )
    }


def _classification_metrics(
    *,
    truth: np.ndarray,
    predictions: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    truth_array = np.asarray(
        truth,
        dtype=str,
    )

    prediction_array = np.asarray(
        predictions,
        dtype=str,
    )

    if truth_array.shape != prediction_array.shape:
        raise ValueError(
            "Classification truth/prediction shape mismatch."
        )

    if truth_array.ndim != 1:
        raise ValueError(
            "Classification arrays must be one-dimensional."
        )

    allowed = set(
        labels
    )

    unknown_truth = (
        set(
            truth_array.tolist()
        )
        - allowed
    )

    unknown_prediction = (
        set(
            prediction_array.tolist()
        )
        - allowed
    )

    if unknown_truth:
        raise ValueError(
            "Unexpected truth labels: "
            f"{sorted(unknown_truth)}"
        )

    if unknown_prediction:
        raise ValueError(
            "Unexpected prediction labels: "
            f"{sorted(unknown_prediction)}"
        )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            truth_array,
            prediction_array,
            labels=list(
                labels
            ),
            zero_division=0,
        )
    )

    per_class: dict[
        str,
        dict[str, Any],
    ] = {}

    for index, label in enumerate(
        labels
    ):
        per_class[
            label
        ] = {
            "precision":
                float(
                    precision[
                        index
                    ]
                ),

            "recall":
                float(
                    recall[
                        index
                    ]
                ),

            "f1":
                float(
                    f1[
                        index
                    ]
                ),

            "support":
                int(
                    support[
                        index
                    ]
                ),
        }

    return {
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    truth_array,
                    prediction_array,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    truth_array,
                    prediction_array,
                    labels=list(
                        labels
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),

        "per_class":
            per_class,

        "confusion_matrix": {
            "labels":
                list(
                    labels
                ),

            "matrix":
                confusion_matrix(
                    truth_array,
                    prediction_array,
                    labels=list(
                        labels
                    ),
                ).astype(
                    int
                ).tolist(),
        },
    }


def _prediction_integrity(
    *,
    active_probability: np.ndarray,
    binary_predictions: np.ndarray,
) -> dict[str, Any]:
    probability = np.asarray(
        active_probability,
        dtype=np.float64,
    )

    predictions = np.asarray(
        binary_predictions,
        dtype=np.bool_,
    )

    if probability.ndim != 1:
        raise ValueError(
            "Detector probability output must be one-dimensional."
        )

    if predictions.shape != probability.shape:
        raise ValueError(
            "Detector probability/prediction shape mismatch."
        )

    if not np.all(
        np.isfinite(
            probability
        )
    ):
        raise ValueError(
            "Detector TEST probabilities contain non-finite values."
        )

    if np.any(
        probability < 0.0
    ) or np.any(
        probability > 1.0
    ):
        raise ValueError(
            "Detector TEST probabilities must be within [0, 1]."
        )

    return {
        "row_count":
            int(
                probability.size
            ),

        "probability_finite":
            True,

        "probability_min":
            float(
                np.min(
                    probability
                )
            ),

        "probability_max":
            float(
                np.max(
                    probability
                )
            ),

        "predicted_active_rows":
            int(
                np.sum(
                    predictions
                )
            ),

        "predicted_inactive_rows":
            int(
                np.sum(
                    ~predictions
                )
            ),
    }


def validate_phase5f_preflight(
    *,
    phase5_protocol_path: str | Path = (
        DEFAULT_PHASE5_PROTOCOL_PATH
    ),
    phase5er_protocol_path: str | Path = (
        DEFAULT_PHASE5ER_PROTOCOL_PATH
    ),
    benchmark_path: str | Path = (
        DEFAULT_BENCHMARK_PATH
    ),
    selection_path: str | Path = (
        DEFAULT_SELECTION_PATH
    ),
    final_refit_path: str | Path = (
        DEFAULT_FINAL_REFIT_PATH
    ),
    detector_path: str | Path = (
        DEFAULT_DETECTOR_PATH
    ),
    diagnoser_path: str | Path = (
        DEFAULT_DIAGNOSER_PATH
    ),
) -> dict[str, Any]:
    """Validate the frozen Phase 5F unlock state without reading TEST data."""

    hashes = {
        "phase5_protocol":
            _verify_hash(
                phase5_protocol_path,
                EXPECTED_PHASE5_PROTOCOL_SHA256,
                label="Phase 5 protocol",
            ),

        "phase5er_protocol":
            _verify_hash(
                phase5er_protocol_path,
                EXPECTED_PHASE5ER_PROTOCOL_SHA256,
                label="Phase 5E-R protocol",
            ),

        "redevelopment_benchmark":
            _verify_hash(
                benchmark_path,
                EXPECTED_BENCHMARK_SHA256,
                label="Phase 5E-R benchmark",
            ),

        "redevelopment_selection":
            _verify_hash(
                selection_path,
                EXPECTED_SELECTION_SHA256,
                label="Phase 5E-R selection",
            ),

        "redevelopment_final_refit":
            _verify_hash(
                final_refit_path,
                EXPECTED_FINAL_REFIT_SHA256,
                label="Phase 5E-R final refit",
            ),

        "fault_detector":
            _verify_hash(
                detector_path,
                EXPECTED_DETECTOR_SHA256,
                label="Phase 5 detector",
            ),

        "fault_diagnoser":
            _verify_hash(
                diagnoser_path,
                EXPECTED_DIAGNOSER_SHA256,
                label="Phase 5 diagnoser",
            ),
    }

    phase5_protocol = _load_json(
        phase5_protocol_path
    )

    phase5er_protocol = _load_json(
        phase5er_protocol_path
    )

    selection = _load_json(
        selection_path
    )

    refit = _load_json(
        final_refit_path
    )

    if (
        phase5_protocol.get(
            "protocol_state"
        )
        != "FROZEN"
    ):
        raise RuntimeError(
            "Phase 5 protocol is not frozen."
        )

    test_discipline = phase5_protocol.get(
        "test_discipline",
        {},
    )

    if (
        test_discipline.get(
            "phase5f_single_locked_evaluation"
        )
        is not True
    ):
        raise RuntimeError(
            "Single locked TEST evaluation is not frozen."
        )

    if (
        test_discipline.get(
            "rerun_for_performance_improvement_permitted"
        )
        is not False
    ):
        raise RuntimeError(
            "TEST rerun discipline has drifted."
        )

    if (
        phase5er_protocol.get(
            "protocol_state"
        )
        != "FROZEN"
    ):
        raise RuntimeError(
            "Phase 5E-R protocol is not frozen."
        )

    unlock = phase5er_protocol.get(
        "phase5f_unlock",
        {},
    )

    for key in (
        "requires_passing_detector",
        "requires_passing_diagnoser",
        "requires_selection_record",
        "requires_final_refit_record",
        "requires_clean_repository_checkpoint",
        "test_remains_locked_until_all_requirements_pass",
    ):
        if unlock.get(
            key
        ) is not True:
            raise RuntimeError(
                "Phase 5F unlock contract drift detected: "
                f"{key}"
            )

    if (
        selection[
            "overall_validation_gate"
        ][
            "redevelopment_validation_passed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Redevelopment validation did not pass."
        )

    if (
        selection[
            "overall_validation_gate"
        ][
            "controlled_stop_triggered"
        ]
        is not False
    ):
        raise RuntimeError(
            "Selection artifact records a controlled stop."
        )

    if (
        selection[
            "detector_acceptance"
        ][
            "selected_candidate"
        ]
        != EXPECTED_DETECTOR_ID
    ):
        raise RuntimeError(
            "Unexpected selected detector."
        )

    if (
        selection[
            "diagnoser_acceptance"
        ][
            "selected_candidate"
        ]
        != EXPECTED_DIAGNOSER_ID
    ):
        raise RuntimeError(
            "Unexpected selected diagnoser."
        )

    if (
        selection[
            "final_refit_state"
        ][
            "authorized"
        ]
        is not True
    ):
        raise RuntimeError(
            "Final refit was not authorized."
        )

    if (
        selection[
            "test_state"
        ][
            "test_consumed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Selection artifact unexpectedly records TEST consumption."
        )

    if (
        refit[
            "detector"
        ][
            "candidate_id"
        ]
        != EXPECTED_DETECTOR_ID
    ):
        raise RuntimeError(
            "Final-refit detector identity mismatch."
        )

    if (
        refit[
            "diagnoser"
        ][
            "candidate_id"
        ]
        != EXPECTED_DIAGNOSER_ID
    ):
        raise RuntimeError(
            "Final-refit diagnoser identity mismatch."
        )

    if (
        int(
            refit[
                "detector"
            ][
                "fit_rows"
            ]
        )
        != EXPECTED_DETECTOR_FIT_ROWS
    ):
        raise RuntimeError(
            "Final detector fit-row count mismatch."
        )

    if (
        int(
            refit[
                "diagnoser"
            ][
                "fit_rows"
            ]
        )
        != EXPECTED_DIAGNOSER_FIT_ROWS
    ):
        raise RuntimeError(
            "Final diagnoser fit-row count mismatch."
        )

    if (
        float(
            refit[
                "detector"
            ][
                "threshold"
            ]
        )
        != EXPECTED_DETECTOR_THRESHOLD
    ):
        raise RuntimeError(
            "Final detector threshold mismatch."
        )

    if (
        int(
            refit[
                "fit_data"
            ][
                "test_rows"
            ]
        )
        != 0
    ):
        raise RuntimeError(
            "Final refit unexpectedly used TEST rows."
        )

    if (
        refit[
            "governance"
        ][
            "post_selection_tuning_performed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Post-selection tuning was unexpectedly recorded."
        )

    if (
        refit[
            "test_state"
        ][
            "test_consumed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final refit unexpectedly records TEST consumption."
        )

    if (
        refit[
            "test_state"
        ][
            "test_predictions_generated"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final refit unexpectedly records TEST predictions."
        )

    if (
        refit[
            "test_state"
        ][
            "test_performance_calculated"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final refit unexpectedly records TEST performance."
        )

    detector = _load_pickle(
        detector_path
    )

    diagnoser = _load_pickle(
        diagnoser_path
    )

    if (
        detector.candidate_id
        != EXPECTED_DETECTOR_ID
    ):
        raise RuntimeError(
            "Persisted detector identity mismatch."
        )

    if (
        int(
            detector.fit_rows
        )
        != EXPECTED_DETECTOR_FIT_ROWS
    ):
        raise RuntimeError(
            "Persisted detector fit-row count mismatch."
        )

    if (
        float(
            detector.threshold
        )
        != EXPECTED_DETECTOR_THRESHOLD
    ):
        raise RuntimeError(
            "Persisted detector threshold mismatch."
        )

    if (
        diagnoser.candidate_id
        != EXPECTED_DIAGNOSER_ID
    ):
        raise RuntimeError(
            "Persisted diagnoser identity mismatch."
        )

    if (
        int(
            diagnoser.fit_rows
        )
        != EXPECTED_DIAGNOSER_FIT_ROWS
    ):
        raise RuntimeError(
            "Persisted diagnoser fit-row count mismatch."
        )

    return {
        "phase":
            "5F",

        "phase5f_unlock_qualified":
            True,

        "selected_detector":
            EXPECTED_DETECTOR_ID,

        "selected_diagnoser":
            EXPECTED_DIAGNOSER_ID,

        "detector_fit_rows":
            EXPECTED_DETECTOR_FIT_ROWS,

        "diagnoser_fit_rows":
            EXPECTED_DIAGNOSER_FIT_ROWS,

        "detector_threshold":
            EXPECTED_DETECTOR_THRESHOLD,

        "test_fit_rows":
            0,

        "test_consumed_before_evaluation":
            False,

        "frozen_hashes":
            hashes,
    }


def load_phase5_locked_test_split(
    *,
    unlock_test: bool = False,
    protocol_path: str | Path = (
        DEFAULT_PHASE5_PROTOCOL_PATH
    ),
    dataset_path: str | Path | None = None,
) -> MonitoringDevelopmentSplit:
    """Load the locked Phase 5 TEST split after an explicit Phase 5F unlock.

    Unlike the development loader, this function intentionally interprets
    TEST predictive features and targets. Calling it against the real frozen
    monitoring dataset therefore consumes TEST information.
    """

    if unlock_test is not True:
        raise RuntimeError(
            "Phase 5 TEST access requires explicit "
            "unlock_test=True."
        )

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

    test_discipline = protocol[
        "test_discipline"
    ]

    if (
        test_discipline.get(
            "phase5f_single_locked_evaluation"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen protocol does not authorize Phase 5F."
        )

    if (
        test_discipline.get(
            "run_only_after_validation_acceptance"
        )
        is not True
    ):
        raise RuntimeError(
            "Frozen protocol TEST timing contract drift detected."
        )

    dataset_record = protocol[
        "upstream"
    ][
        "phase3_monitoring_dataset"
    ]

    if dataset_path is None:
        dataset_file = _resolve_repository_path(
            dataset_record[
                "path"
            ]
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

    feature_rows: list[
        tuple[float, ...]
    ] = []

    binary_values: list[
        bool
    ] = []

    multiclass_values: list[
        str
    ] = []

    episode_values: list[
        str
    ] = []

    step_values: list[
        int
    ] = []

    test_rows_seen = 0
    non_test_rows_skipped = 0

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

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
            - set(
                reader.fieldnames
            )
        )

        if missing_columns:
            raise ValueError(
                "Monitoring CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            # Inspect split metadata first. Non-TEST predictive/target
            # payload is not interpreted by the locked TEST loader.
            split_name = row[
                "split"
            ].strip()

            if (
                split_name
                not in allowed_splits
            ):
                raise ValueError(
                    "Unexpected monitoring split: "
                    f"{split_name!r}"
                )

            if split_name != "test":
                non_test_rows_skipped += 1
                continue

            test_rows_seen += 1

            feature_rows.append(
                _feature_vector(
                    row
                )
            )

            binary_values.append(
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

            multiclass_values.append(
                family
            )

            episode_id = row[
                "episode_id"
            ].strip()

            if not episode_id:
                raise ValueError(
                    "Empty episode_id encountered."
                )

            episode_values.append(
                episode_id
            )

            step_values.append(
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

    if (
        test_rows_seen
        != expected_test_rows
    ):
        raise ValueError(
            "Locked TEST row-count mismatch."
        )

    test = _finalize_split(
        name="test",
        feature_rows=feature_rows,
        binary_values=binary_values,
        multiclass_values=multiclass_values,
        episode_values=episode_values,
        step_values=step_values,
        protocol=protocol,
    )

    if (
        test.row_count
        != expected_test_rows
    ):
        raise RuntimeError(
            "Finalized TEST row count mismatch."
        )

    if non_test_rows_skipped <= 0:
        raise RuntimeError(
            "Locked TEST loader expected non-TEST rows "
            "to remain excluded."
        )

    return test


def run_phase5f_locked_test_evaluation(
    *,
    phase5_protocol_path: str | Path = (
        DEFAULT_PHASE5_PROTOCOL_PATH
    ),
    phase5er_protocol_path: str | Path = (
        DEFAULT_PHASE5ER_PROTOCOL_PATH
    ),
    benchmark_path: str | Path = (
        DEFAULT_BENCHMARK_PATH
    ),
    selection_path: str | Path = (
        DEFAULT_SELECTION_PATH
    ),
    final_refit_path: str | Path = (
        DEFAULT_FINAL_REFIT_PATH
    ),
    detector_path: str | Path = (
        DEFAULT_DETECTOR_PATH
    ),
    diagnoser_path: str | Path = (
        DEFAULT_DIAGNOSER_PATH
    ),
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Perform the one-time locked Phase 5 TEST evaluation in memory.

    Calling this function with the real frozen dataset interprets TEST
    features and targets and therefore consumes the predictive holdout.
    """

    preflight = validate_phase5f_preflight(
        phase5_protocol_path=phase5_protocol_path,
        phase5er_protocol_path=phase5er_protocol_path,
        benchmark_path=benchmark_path,
        selection_path=selection_path,
        final_refit_path=final_refit_path,
        detector_path=detector_path,
        diagnoser_path=diagnoser_path,
    )

    detector = _load_pickle(
        detector_path
    )

    diagnoser = _load_pickle(
        diagnoser_path
    )

    test = load_phase5_locked_test_split(
        unlock_test=True,
        protocol_path=phase5_protocol_path,
        dataset_path=dataset_path,
    )

    if test.X.shape != (
        EXPECTED_TEST_ROWS,
        len(
            MODEL_FEATURES
        ),
    ):
        raise RuntimeError(
            "Phase 5F TEST feature-matrix shape mismatch."
        )

    if test.binary_y.shape != (
        EXPECTED_TEST_ROWS,
    ):
        raise RuntimeError(
            "Phase 5F TEST binary-target shape mismatch."
        )

    if test.multiclass_y.shape != (
        EXPECTED_TEST_ROWS,
    ):
        raise RuntimeError(
            "Phase 5F TEST multiclass-target shape mismatch."
        )

    if test.episode_ids.shape != (
        EXPECTED_TEST_ROWS,
    ):
        raise RuntimeError(
            "Phase 5F TEST episode-id shape mismatch."
        )

    if test.step_indices.shape != (
        EXPECTED_TEST_ROWS,
    ):
        raise RuntimeError(
            "Phase 5F TEST step-index shape mismatch."
        )

    if (
        test.episode_count
        != EXPECTED_TEST_EPISODES
    ):
        raise RuntimeError(
            "Phase 5F TEST episode-count mismatch."
        )

    if not np.all(
        np.isfinite(
            test.X
        )
    ):
        raise ValueError(
            "Phase 5F TEST features contain non-finite values."
        )

    active_probability = (
        detector.predict_active_probability(
            test.X
        )
    )

    detector_predictions = (
        detector.predict(
            test.X
        )
    )

    active_probability = np.asarray(
        active_probability,
        dtype=np.float64,
    )

    detector_predictions = np.asarray(
        detector_predictions,
        dtype=np.bool_,
    )

    expected_prediction_shape = (
        EXPECTED_TEST_ROWS,
    )

    if (
        active_probability.shape
        != expected_prediction_shape
    ):
        raise RuntimeError(
            "Detector returned unexpected TEST probability shape."
        )

    if (
        detector_predictions.shape
        != expected_prediction_shape
    ):
        raise RuntimeError(
            "Detector returned unexpected TEST prediction shape."
        )

    detector_integrity = (
        _prediction_integrity(
            active_probability=active_probability,
            binary_predictions=detector_predictions,
        )
    )

    detector_metrics = (
        redevelopment_detector_metrics(
            binary_truth=test.binary_y,
            predictions=detector_predictions,
            active_probability=active_probability,
            original_family_truth=test.multiclass_y,
        )
    )

    sequence_diagnostics = (
        detector_sequence_diagnostics(
            binary_truth=test.binary_y,
            predictions=detector_predictions,
            episode_ids=test.episode_ids,
            step_indices=test.step_indices,
        )
    )

    active_mask = np.asarray(
        test.binary_y,
        dtype=np.bool_,
    )

    active_row_count = int(
        np.sum(
            active_mask
        )
    )

    if active_row_count <= 0:
        raise RuntimeError(
            "Phase 5F TEST contains no active fault rows."
        )

    mapped_active_truth = (
        map_redevelopment_active_fault_labels(
            np.asarray(
                test.multiclass_y[
                    active_mask
                ],
                dtype=str,
            ),
            protocol_path=phase5er_protocol_path,
        )
    )

    diagnosis_predictions = (
        diagnoser.predict(
            test.X[
                active_mask
            ]
        )
    )

    diagnosis_predictions = np.asarray(
        diagnosis_predictions,
        dtype=str,
    )

    if (
        diagnosis_predictions.shape
        != mapped_active_truth.shape
    ):
        raise RuntimeError(
            "Diagnoser returned unexpected active-row "
            "TEST prediction shape."
        )

    allowed_diagnosis_classes = set(
        REDEVELOPMENT_CLASSES
    )

    unknown_diagnosis_predictions = (
        set(
            diagnosis_predictions.tolist()
        )
        - allowed_diagnosis_classes
    )

    if unknown_diagnosis_predictions:
        raise RuntimeError(
            "Diagnoser returned unexpected TEST classes: "
            f"{sorted(unknown_diagnosis_predictions)}"
        )

    diagnosis_metrics = (
        redevelopment_diagnosis_metrics(
            truth=mapped_active_truth,
            predictions=diagnosis_predictions,
        )
    )

    majority_class = str(
        diagnoser.train_majority_class
    )

    if (
        majority_class
        not in allowed_diagnosis_classes
    ):
        raise RuntimeError(
            "Frozen diagnoser majority class is invalid."
        )

    majority_predictions = np.full(
        mapped_active_truth.shape,
        majority_class,
        dtype=object,
    )

    majority_metrics = (
        _classification_metrics(
            truth=mapped_active_truth,
            predictions=majority_predictions,
            labels=tuple(
                REDEVELOPMENT_CLASSES
            ),
        )
    )

    majority_improvement = (
        float(
            diagnosis_metrics[
                "macro_f1"
            ]
        )
        - float(
            majority_metrics[
                "macro_f1"
            ]
        )
    )

    # Secondary end-to-end four-state reporting.
    # This is explicitly non-selective and cannot change the frozen models.
    end_to_end_truth = np.full(
        (
            EXPECTED_TEST_ROWS,
        ),
        "none",
        dtype=object,
    )

    end_to_end_truth[
        active_mask
    ] = np.asarray(
        mapped_active_truth,
        dtype=object,
    )

    end_to_end_predictions = np.full(
        (
            EXPECTED_TEST_ROWS,
        ),
        "none",
        dtype=object,
    )

    predicted_active_mask = np.asarray(
        detector_predictions,
        dtype=np.bool_,
    )

    if np.any(
        predicted_active_mask
    ):
        predicted_active_diagnosis = (
            diagnoser.predict(
                test.X[
                    predicted_active_mask
                ]
            )
        )

        predicted_active_diagnosis = np.asarray(
            predicted_active_diagnosis,
            dtype=str,
        )

        unknown_pipeline_predictions = (
            set(
                predicted_active_diagnosis.tolist()
            )
            - allowed_diagnosis_classes
        )

        if unknown_pipeline_predictions:
            raise RuntimeError(
                "End-to-end diagnoser returned unexpected classes: "
                f"{sorted(unknown_pipeline_predictions)}"
            )

        end_to_end_predictions[
            predicted_active_mask
        ] = np.asarray(
            predicted_active_diagnosis,
            dtype=object,
        )

    end_to_end_metrics = (
        _classification_metrics(
            truth=end_to_end_truth,
            predictions=end_to_end_predictions,
            labels=END_TO_END_CLASSES,
        )
    )

    return {
        "phase":
            "5F",

        "stage":
            "one_time_locked_test_evaluation",

        "evaluation_status":
            "completed",

        "irreversible_test_evaluation":
            True,

        "governance": {
            "phase5f_unlock_qualified":
                True,

            "final_models_frozen_before_test":
                True,

            "test_used_for_model_selection":
                False,

            "test_used_for_threshold_selection":
                False,

            "test_used_for_retuning":
                False,

            "model_changes_permitted_from_test":
                False,

            "rerun_for_performance_improvement_permitted":
                False,
        },

        "input_artifacts": {
            "phase5_protocol": {
                "path":
                    str(
                        phase5_protocol_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "phase5_protocol"
                    ],
            },

            "phase5er_protocol": {
                "path":
                    str(
                        phase5er_protocol_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "phase5er_protocol"
                    ],
            },

            "redevelopment_benchmark": {
                "path":
                    str(
                        benchmark_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "redevelopment_benchmark"
                    ],
            },

            "redevelopment_selection": {
                "path":
                    str(
                        selection_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "redevelopment_selection"
                    ],
            },

            "redevelopment_final_refit": {
                "path":
                    str(
                        final_refit_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "redevelopment_final_refit"
                    ],
            },

            "fault_detector": {
                "path":
                    str(
                        detector_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "fault_detector"
                    ],
            },

            "fault_diagnoser": {
                "path":
                    str(
                        diagnoser_path
                    ),

                "sha256":
                    preflight[
                        "frozen_hashes"
                    ][
                        "fault_diagnoser"
                    ],
            },
        },

        "data_usage": {
            "final_fit_splits": [
                "train",
                "validation",
            ],

            "detector_final_fit_rows":
                EXPECTED_DETECTOR_FIT_ROWS,

            "diagnoser_final_fit_rows":
                EXPECTED_DIAGNOSER_FIT_ROWS,

            "test_rows":
                int(
                    test.row_count
                ),

            "test_episodes":
                int(
                    test.episode_count
                ),

            "test_active_rows":
                active_row_count,

            "test_inactive_rows":
                int(
                    test.row_count
                    - active_row_count
                ),

            "test_features_accessed":
                True,

            "test_targets_accessed":
                True,

            "test_predictions_generated":
                True,

            "test_performance_calculated":
                True,

            "test_evaluation_performed":
                True,

            "test_may_affect_model_selection":
                False,

            "retuning_after_test_permitted":
                False,
        },

        "feature_contract": {
            "features":
                list(
                    MODEL_FEATURES
                ),

            "feature_order_frozen":
                True,

            "feature_count":
                len(
                    MODEL_FEATURES
                ),
        },

        "test_target_summary": {
            "binary_active_rows":
                active_row_count,

            "binary_inactive_rows":
                int(
                    test.row_count
                    - active_row_count
                ),

            "original_family_counts":
                _value_counts(
                    test.multiclass_y
                ),

            "redeveloped_active_class_counts":
                _value_counts(
                    mapped_active_truth
                ),
        },

        "detector": {
            "candidate_id":
                EXPECTED_DETECTOR_ID,

            "fit_rows":
                EXPECTED_DETECTOR_FIT_ROWS,

            "threshold":
                EXPECTED_DETECTOR_THRESHOLD,

            "metrics":
                detector_metrics,

            "prediction_integrity":
                detector_integrity,

            "sequence_diagnostics":
                sequence_diagnostics,
        },

        "diagnoser": {
            "candidate_id":
                EXPECTED_DIAGNOSER_ID,

            "fit_rows":
                EXPECTED_DIAGNOSER_FIT_ROWS,

            "classes":
                list(
                    REDEVELOPMENT_CLASSES
                ),

            "ground_truth_active_rows_only":
                True,

            "metrics":
                diagnosis_metrics,

            "training_majority_class":
                majority_class,

            "majority_baseline": {
                "metrics":
                    majority_metrics,

                "macro_f1_improvement":
                    majority_improvement,
            },
        },

        "secondary_end_to_end": {
            "selection_metric":
                False,

            "classes":
                list(
                    END_TO_END_CLASSES
                ),

            "metrics":
                end_to_end_metrics,

            "truth_counts":
                _value_counts(
                    end_to_end_truth
                ),

            "prediction_counts":
                _value_counts(
                    end_to_end_predictions
                ),
        },

        "interpretation": {
            "performance_threshold_introduced":
                False,

            "test_is_final_generalisation_evidence":
                True,

            "test_is_development_feedback":
                False,

            "model_changes_permitted_from_test":
                False,

            "threshold_changes_permitted_from_test":
                False,

            "retuning_after_test_permitted":
                False,
        },

        "test_state": {
            "test_features_interpreted":
                True,

            "test_targets_interpreted":
                True,

            "test_matrix_created":
                True,

            "test_predictions_generated":
                True,

            "test_performance_calculated":
                True,

            "test_consumed":
                True,
        },

        "preflight":
            preflight,

        "scientific_scope": {
            "synthetic_data_only":
                True,

            "reduced_order_argon_monitoring_environment":
                True,

            "pressure_path_anomaly_is_ambiguity_aware":
                True,

            "pressure_path_anomaly_identifies_specific_mechanism":
                False,

            "experimental_validation":
                False,

            "industrial_validation":
                False,

            "oipt_operating_range_claim":
                False,
        },
    }


def write_phase5f_locked_test_evaluation(
    output_path: str | Path = (
        DEFAULT_OUTPUT_PATH
    ),
    **evaluation_kwargs: Any,
) -> Path:
    """Perform and persist the one-time locked Phase 5 TEST evaluation.

    The destination existence check occurs before any TEST evaluation so an
    already-consumed result cannot be overwritten or regenerated through this
    writer.
    """

    destination = Path(
        output_path
    )

    if not destination.is_absolute():
        destination = _resolve_repository_path(
            destination
        )

    if destination.exists():
        raise FileExistsError(
            "Phase 5F locked TEST evaluation artifact already exists. "
            "Refusing to overwrite consumed TEST evidence."
        )

    payload = (
        run_phase5f_locked_test_evaluation(
            **evaluation_kwargs
        )
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    with destination.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            serialized
        )

    return destination
