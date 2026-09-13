"""Phase 5E-R validation-only diagnostic redevelopment.

This module implements only the machinery frozen in
``configs/phase5/phase5er_protocol.json``.

It deliberately does not load the real monitoring dataset, write result
artifacts, access TEST, or perform Phase 5F evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import math

import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plasma_ai.monitoring.data import MODEL_FEATURES
from plasma_ai.monitoring.supervised import (
    FittedDetectorCandidate,
    fit_detector_candidate,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PHASE5ER_PROTOCOL_PATH = (
    _REPO_ROOT
    / "configs"
    / "phase5"
    / "phase5er_protocol.json"
)

ORIGINAL_ACTIVE_CLASSES = (
    "flow_delivery",
    "power_coupling",
    "pressure_sensor_bias",
    "pumping_effectiveness",
)

REDEVELOPMENT_CLASSES = (
    "flow_delivery",
    "power_coupling",
    "pressure_path_anomaly",
)

REDEVELOPMENT_END_TO_END_CLASSES = (
    "none",
    "flow_delivery",
    "power_coupling",
    "pressure_path_anomaly",
)

REDEVELOPMENT_DETECTOR_CANDIDATE_IDS = (
    "ERD1_hgb_threshold_045",
    "ERD2_hgb_threshold_040",
    "ERD3_extra_trees_threshold_045",
    "ERD4_extra_trees_threshold_040",
)

REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS = (
    "ERG1_logistic_regression",
    "ERG2_hist_gradient_boosting",
    "ERG3_extra_trees",
)

_EXPECTED_CLASS_MAPPING = {
    "flow_delivery":
        "flow_delivery",
    "power_coupling":
        "power_coupling",
    "pressure_sensor_bias":
        "pressure_path_anomaly",
    "pumping_effectiveness":
        "pressure_path_anomaly",
}

_EXPECTED_DETECTOR_BASES = {
    "ERD1_hgb_threshold_045":
        (
            "D2_hist_gradient_boosting",
            0.45,
        ),
    "ERD2_hgb_threshold_040":
        (
            "D2_hist_gradient_boosting",
            0.40,
        ),
    "ERD3_extra_trees_threshold_045":
        (
            "D3_extra_trees",
            0.45,
        ),
    "ERD4_extra_trees_threshold_040":
        (
            "D3_extra_trees",
            0.40,
        ),
}

_EXPECTED_DETECTOR_ACCEPTANCE = {
    "balanced_accuracy_min":
        0.75,
    "macro_f1_min":
        0.75,
    "fault_active_recall_min":
        0.75,
    "specificity_min":
        0.75,
    "per_active_fault_family_recall_min":
        0.60,
}

_EXPECTED_DIAGNOSER_ACCEPTANCE = {
    "macro_f1_min":
        0.60,
    "balanced_accuracy_min":
        0.60,
    "per_class_recall_min":
        0.50,
    "majority_baseline_macro_f1_improvement_min":
        0.10,
}

_EXPECTED_DETECTOR_SELECTION_RULES = (
    "filter_to_candidates_passing_all_acceptance_criteria",
    "highest_balanced_accuracy",
    "highest_macro_f1",
    "lowest_false_positive_rate",
    "highest_decision_threshold",
    "candidate_priority_ERD1_ERD2_ERD3_ERD4",
)

_EXPECTED_DIAGNOSER_SELECTION_RULES = (
    "filter_to_candidates_passing_all_acceptance_criteria",
    "highest_macro_f1",
    "highest_balanced_accuracy",
    "highest_minimum_per_class_recall",
    "candidate_priority_ERG1_ERG2_ERG3",
)


def load_phase5er_contract(
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and strictly validate the frozen Phase 5E-R contract."""

    path = Path(protocol_path)

    protocol = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if protocol.get("phase") != "5E-R":
        raise ValueError(
            "Expected Phase 5E-R protocol."
        )

    if protocol.get("protocol_state") != "FROZEN":
        raise ValueError(
            "Phase 5E-R protocol must be frozen."
        )

    feature_contract = protocol[
        "feature_contract"
    ]

    if tuple(
        feature_contract[
            "model_feature_order"
        ]
    ) != tuple(MODEL_FEATURES):
        raise ValueError(
            "Phase 5E-R feature order differs from MODEL_FEATURES."
        )

    if (
        feature_contract[
            "feature_contract_changed_from_original_phase5"
        ]
        is not False
    ):
        raise ValueError(
            "Phase 5E-R feature contract must remain unchanged."
        )

    if (
        feature_contract[
            "phase4_surrogate_outputs_used"
        ]
        is not False
    ):
        raise ValueError(
            "Phase 4 surrogate outputs are forbidden in Phase 5E-R."
        )

    if (
        feature_contract[
            "new_synthetic_measurements_created"
        ]
        is not False
    ):
        raise ValueError(
            "New synthetic diagnostics are forbidden in Phase 5E-R."
        )

    detector = protocol[
        "binary_detector"
    ]

    if tuple(
        detector["candidates"]
    ) != REDEVELOPMENT_DETECTOR_CANDIDATE_IDS:
        raise ValueError(
            "Unexpected Phase 5E-R detector candidate order."
        )

    for candidate_id, (
        expected_base,
        expected_threshold,
    ) in _EXPECTED_DETECTOR_BASES.items():
        record = detector[
            "candidates"
        ][
            candidate_id
        ]

        if record["base_candidate"] != expected_base:
            raise ValueError(
                f"{candidate_id}: unexpected base detector."
            )

        if not math.isclose(
            float(record["threshold"]),
            expected_threshold,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"{candidate_id}: unexpected threshold."
            )

    if detector["acceptance"] != _EXPECTED_DETECTOR_ACCEPTANCE:
        raise ValueError(
            "Detector acceptance criteria changed."
        )

    if (
        detector[
            "acceptance_changed_from_original_phase5"
        ]
        is not False
    ):
        raise ValueError(
            "Detector acceptance must remain unchanged."
        )

    if tuple(
        detector["selection_rules"]
    ) != _EXPECTED_DETECTOR_SELECTION_RULES:
        raise ValueError(
            "Unexpected detector selection rules."
        )

    diagnoser = protocol[
        "active_fault_diagnosis"
    ]

    if tuple(
        diagnoser[
            "redevelopment_classes"
        ]
    ) != REDEVELOPMENT_CLASSES:
        raise ValueError(
            "Unexpected Phase 5E-R diagnosis classes."
        )

    if diagnoser[
        "class_mapping"
    ] != _EXPECTED_CLASS_MAPPING:
        raise ValueError(
            "Unexpected Phase 5E-R class mapping."
        )

    if tuple(
        diagnoser["candidates"]
    ) != REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS:
        raise ValueError(
            "Unexpected Phase 5E-R diagnoser candidate order."
        )

    if (
        diagnoser["acceptance"]
        != _EXPECTED_DIAGNOSER_ACCEPTANCE
    ):
        raise ValueError(
            "Diagnoser acceptance criteria changed."
        )

    if tuple(
        diagnoser["selection_rules"]
    ) != _EXPECTED_DIAGNOSER_SELECTION_RULES:
        raise ValueError(
            "Unexpected diagnoser selection rules."
        )

    if (
        protocol["failure_action"]
        != "controlled_stop_before_test"
    ):
        raise ValueError(
            "Unexpected Phase 5E-R failure action."
        )

    final_refit = protocol[
        "final_refit"
    ]

    if final_refit[
        "test_rows_for_fit"
    ] != 0:
        raise ValueError(
            "TEST rows must remain excluded from refit."
        )

    if (
        protocol["governance"][
            "test_access_during_redevelopment"
        ]
        is not False
    ):
        raise ValueError(
            "TEST access must remain forbidden during redevelopment."
        )

    return protocol


def _validate_feature_matrix(
    X: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        X,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Feature matrix must contain at least one row."
        )

    if matrix.shape[1] != len(
        MODEL_FEATURES
    ):
        raise ValueError(
            "Feature matrix has incorrect column count."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            "Feature matrix must contain only finite values."
        )

    return matrix


def _validate_binary_target(
    y: np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    values = np.asarray(
        y
    )

    if values.ndim != 1:
        raise ValueError(
            "Binary target must be one-dimensional."
        )

    if values.shape[0] != expected_rows:
        raise ValueError(
            "Binary target row count does not match features."
        )

    unique = set(
        values.tolist()
    )

    if not unique.issubset(
        {
            False,
            True,
            0,
            1,
        }
    ):
        raise ValueError(
            "Binary target must contain only False/True values."
        )

    return values.astype(
        bool,
        copy=False,
    )


def _validate_original_family_target(
    labels: np.ndarray,
    *,
    binary_y: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    values = np.asarray(
        labels,
        dtype=str,
    )

    if values.ndim != 1:
        raise ValueError(
            "Fault-family target must be one-dimensional."
        )

    if values.shape[0] != expected_rows:
        raise ValueError(
            "Fault-family target row count does not match features."
        )

    allowed = {
        "none",
        *ORIGINAL_ACTIVE_CLASSES,
    }

    unknown = set(
        values.tolist()
    ) - allowed

    if unknown:
        raise ValueError(
            "Unexpected original fault-family labels: "
            f"{sorted(unknown)!r}"
        )

    inactive_invalid = (
        (~binary_y)
        & (
            values
            != "none"
        )
    )

    if np.any(
        inactive_invalid
    ):
        raise ValueError(
            "Inactive rows must use fault-family label 'none'."
        )

    active_invalid = (
        binary_y
        & (
            values
            == "none"
        )
    )

    if np.any(
        active_invalid
    ):
        raise ValueError(
            "Active rows must use an active fault-family label."
        )

    return values


def map_redevelopment_active_fault_labels(
    labels: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> np.ndarray:
    """Map original active fault families to the frozen 3-class target."""

    protocol = load_phase5er_contract(
        protocol_path
    )

    mapping = protocol[
        "active_fault_diagnosis"
    ][
        "class_mapping"
    ]

    values = np.asarray(
        labels,
        dtype=str,
    )

    if values.ndim != 1:
        raise ValueError(
            "Active fault labels must be one-dimensional."
        )

    unknown = set(
        values.tolist()
    ) - set(
        ORIGINAL_ACTIVE_CLASSES
    )

    if unknown:
        raise ValueError(
            "Cannot map non-active or unknown fault-family labels: "
            f"{sorted(unknown)!r}"
        )

    return np.asarray(
        [
            mapping[value]
            for value in values
        ],
        dtype=str,
    )


def fit_redevelopment_detector_candidate(
    candidate_id: str,
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> FittedDetectorCandidate:
    """Fit one frozen Phase 5E-R detector candidate on supplied rows."""

    protocol = load_phase5er_contract(
        protocol_path
    )

    if (
        candidate_id
        not in REDEVELOPMENT_DETECTOR_CANDIDATE_IDS
    ):
        raise ValueError(
            f"Unknown Phase 5E-R detector candidate: {candidate_id!r}"
        )

    X = _validate_feature_matrix(
        train_X
    )

    y = _validate_binary_target(
        train_binary_y,
        expected_rows=X.shape[0],
    )

    if np.unique(
        y
    ).size != 2:
        raise ValueError(
            "Detector training target must contain both classes."
        )

    config = protocol[
        "binary_detector"
    ][
        "candidates"
    ][
        candidate_id
    ]

    base_candidate = config[
        "base_candidate"
    ]

    threshold = float(
        config["threshold"]
    )

    fitted_base = fit_detector_candidate(
        base_candidate,
        X,
        y,
    )

    return replace(
        fitted_base,
        candidate_id=candidate_id,
        threshold=threshold,
    )


@dataclass(frozen=True)
class FittedRedevelopmentDiagnoserCandidate:
    """Fitted three-class ambiguity-aware Phase 5E-R diagnoser."""

    candidate_id: str
    model: Any
    fit_rows: int
    train_majority_class: str

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        matrix = _validate_feature_matrix(
            X
        )

        predictions = np.asarray(
            self.model.predict(
                matrix
            ),
            dtype=str,
        )

        unknown = set(
            predictions.tolist()
        ) - set(
            REDEVELOPMENT_CLASSES
        )

        if unknown:
            raise RuntimeError(
                "Diagnoser produced unexpected classes: "
                f"{sorted(unknown)!r}"
            )

        return predictions


def _build_redevelopment_diagnoser(
    candidate_id: str,
    *,
    protocol: Mapping[str, Any],
) -> Any:
    record = protocol[
        "active_fault_diagnosis"
    ][
        "candidates"
    ][
        candidate_id
    ]

    parameters = dict(
        record["parameters"]
    )

    classifier = record[
        "classifier"
    ]

    if classifier == "LogisticRegression":
        if record.get(
            "scaler"
        ) != "StandardScaler":
            raise ValueError(
                "Redevelopment logistic candidate requires StandardScaler."
            )

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        **parameters
                    ),
                ),
            ]
        )

    if classifier == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            **parameters
        )

    if classifier == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            **parameters
        )

    raise ValueError(
        f"Unsupported redevelopment classifier: {classifier!r}"
    )


def _majority_class(
    labels: np.ndarray,
) -> str:
    counts = {
        label:
            int(
                np.sum(
                    labels
                    == label
                )
            )
        for label in REDEVELOPMENT_CLASSES
    }

    maximum = max(
        counts.values()
    )

    for label in REDEVELOPMENT_CLASSES:
        if counts[label] == maximum:
            return label

    raise RuntimeError(
        "Unable to determine redevelopment majority class."
    )


def fit_redevelopment_diagnoser_candidate(
    candidate_id: str,
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    train_multiclass_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> FittedRedevelopmentDiagnoserCandidate:
    """Fit one frozen three-class diagnoser on active supplied rows."""

    protocol = load_phase5er_contract(
        protocol_path
    )

    if (
        candidate_id
        not in REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS
    ):
        raise ValueError(
            f"Unknown Phase 5E-R diagnoser candidate: {candidate_id!r}"
        )

    X = _validate_feature_matrix(
        train_X
    )

    binary_y = _validate_binary_target(
        train_binary_y,
        expected_rows=X.shape[0],
    )

    original_labels = _validate_original_family_target(
        train_multiclass_y,
        binary_y=binary_y,
        expected_rows=X.shape[0],
    )

    active_X = X[
        binary_y
    ]

    active_original_labels = original_labels[
        binary_y
    ]

    if active_X.shape[0] == 0:
        raise ValueError(
            "Diagnoser training requires active rows."
        )

    mapped_labels = map_redevelopment_active_fault_labels(
        active_original_labels,
        protocol_path=protocol_path,
    )

    observed = set(
        mapped_labels.tolist()
    )

    if observed != set(
        REDEVELOPMENT_CLASSES
    ):
        raise ValueError(
            "Diagnoser training data must represent all "
            "three redevelopment classes."
        )

    model = _build_redevelopment_diagnoser(
        candidate_id,
        protocol=protocol,
    )

    model.fit(
        active_X,
        mapped_labels,
    )

    model_classes = tuple(
        np.asarray(
            model.classes_,
            dtype=str,
        ).tolist()
    )

    if set(
        model_classes
    ) != set(
        REDEVELOPMENT_CLASSES
    ):
        raise RuntimeError(
            "Fitted diagnoser does not contain expected classes."
        )

    majority_class = _majority_class(
        mapped_labels
    )

    return FittedRedevelopmentDiagnoserCandidate(
        candidate_id=candidate_id,
        model=model,
        fit_rows=int(
            active_X.shape[0]
        ),
        train_majority_class=majority_class,
    )


def redevelopment_detector_metrics(
    *,
    binary_truth: np.ndarray,
    predictions: np.ndarray,
    active_probability: np.ndarray,
    original_family_truth: np.ndarray,
) -> dict[str, Any]:
    """Calculate frozen detector validation metrics."""

    truth = np.asarray(
        binary_truth,
        dtype=bool,
    )

    predicted = np.asarray(
        predictions,
        dtype=bool,
    )

    probability = np.asarray(
        active_probability,
        dtype=float,
    )

    families = np.asarray(
        original_family_truth,
        dtype=str,
    )

    if not (
        truth.ndim
        == predicted.ndim
        == probability.ndim
        == families.ndim
        == 1
    ):
        raise ValueError(
            "Detector metric inputs must be one-dimensional."
        )

    if not (
        truth.shape
        == predicted.shape
        == probability.shape
        == families.shape
    ):
        raise ValueError(
            "Detector metric input shapes do not match."
        )

    if truth.size == 0:
        raise ValueError(
            "Detector metrics require at least one row."
        )

    if np.unique(
        truth
    ).size != 2:
        raise ValueError(
            "Detector metrics require both binary classes."
        )

    if not np.all(
        np.isfinite(
            probability
        )
    ):
        raise ValueError(
            "Detector probabilities must be finite."
        )

    if np.any(
        (
            probability
            < 0.0
        )
        | (
            probability
            > 1.0
        )
    ):
        raise ValueError(
            "Detector probabilities must lie in [0, 1]."
        )

    matrix = confusion_matrix(
        truth,
        predicted,
        labels=[
            False,
            True,
        ],
    )

    tn, fp, fn, tp = (
        int(value)
        for value in matrix.ravel()
    )

    specificity = (
        tn
        / (
            tn
            + fp
        )
    )

    false_positive_rate = (
        fp
        / (
            tn
            + fp
        )
    )

    per_family: dict[str, dict[str, Any]] = {}

    for family in ORIGINAL_ACTIVE_CLASSES:
        mask = (
            truth
            & (
                families
                == family
            )
        )

        support = int(
            np.sum(
                mask
            )
        )

        if support == 0:
            raise ValueError(
                f"No active validation support for {family!r}."
            )

        detected = int(
            np.sum(
                predicted[
                    mask
                ]
            )
        )

        per_family[
            family
        ] = {
            "support":
                support,
            "detected":
                detected,
            "recall":
                float(
                    detected
                    / support
                ),
        }

    return {
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    truth,
                    predicted,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    truth,
                    predicted,
                    labels=[
                        False,
                        True,
                    ],
                    average="macro",
                    zero_division=0,
                )
            ),

        "positive_precision":
            float(
                precision_score(
                    truth,
                    predicted,
                    pos_label=True,
                    zero_division=0,
                )
            ),

        "active_recall":
            float(
                recall_score(
                    truth,
                    predicted,
                    pos_label=True,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "false_positive_rate":
            float(
                false_positive_rate
            ),

        "auroc":
            float(
                roc_auc_score(
                    truth,
                    probability,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    truth,
                    probability,
                )
            ),

        "confusion_matrix": {
            "labels": [
                False,
                True,
            ],
            "matrix":
                matrix.tolist(),
        },

        "per_active_fault_family_recall":
            per_family,
    }


def evaluate_redevelopment_detector_candidate(
    fitted: FittedDetectorCandidate,
    validation_X: np.ndarray,
    validation_binary_y: np.ndarray,
    validation_multiclass_y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted redevelopment detector on supplied rows."""

    if (
        fitted.candidate_id
        not in REDEVELOPMENT_DETECTOR_CANDIDATE_IDS
    ):
        raise ValueError(
            "Expected fitted Phase 5E-R detector candidate."
        )

    X = _validate_feature_matrix(
        validation_X
    )

    binary_y = _validate_binary_target(
        validation_binary_y,
        expected_rows=X.shape[0],
    )

    families = _validate_original_family_target(
        validation_multiclass_y,
        binary_y=binary_y,
        expected_rows=X.shape[0],
    )

    probability = fitted.predict_active_probability(
        X
    )

    predictions = fitted.predict(
        X
    )

    metrics = redevelopment_detector_metrics(
        binary_truth=binary_y,
        predictions=predictions,
        active_probability=probability,
        original_family_truth=families,
    )

    return {
        "candidate_id":
            fitted.candidate_id,
        "fit_rows":
            int(
                fitted.fit_rows
            ),
        "validation_rows":
            int(
                X.shape[0]
            ),
        "threshold":
            float(
                fitted.threshold
            ),
        "metrics":
            metrics,
    }


def redevelopment_diagnosis_metrics(
    *,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Calculate three-class ambiguity-aware diagnosis metrics."""

    truth_values = np.asarray(
        truth,
        dtype=str,
    )

    predicted_values = np.asarray(
        predictions,
        dtype=str,
    )

    if truth_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError(
            "Diagnosis metric inputs must be one-dimensional."
        )

    if truth_values.shape != predicted_values.shape:
        raise ValueError(
            "Diagnosis metric input shapes do not match."
        )

    if truth_values.size == 0:
        raise ValueError(
            "Diagnosis metrics require at least one row."
        )

    allowed = set(
        REDEVELOPMENT_CLASSES
    )

    unknown_truth = set(
        truth_values.tolist()
    ) - allowed

    unknown_predictions = set(
        predicted_values.tolist()
    ) - allowed

    if unknown_truth:
        raise ValueError(
            "Unexpected truth classes: "
            f"{sorted(unknown_truth)!r}"
        )

    if unknown_predictions:
        raise ValueError(
            "Unexpected predicted classes: "
            f"{sorted(unknown_predictions)!r}"
        )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            truth_values,
            predicted_values,
            labels=list(
                REDEVELOPMENT_CLASSES
            ),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        truth_values,
        predicted_values,
        labels=list(
            REDEVELOPMENT_CLASSES
        ),
    )

    per_class: dict[str, dict[str, Any]] = {}

    for index, label in enumerate(
        REDEVELOPMENT_CLASSES
    ):
        per_class[
            label
        ] = {
            "precision":
                float(
                    precision[index]
                ),
            "recall":
                float(
                    recall[index]
                ),
            "f1":
                float(
                    f1[index]
                ),
            "support":
                int(
                    support[index]
                ),
        }

    return {
        "macro_f1":
            float(
                f1_score(
                    truth_values,
                    predicted_values,
                    labels=list(
                        REDEVELOPMENT_CLASSES
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    truth_values,
                    predicted_values,
                )
            ),

        "minimum_per_class_recall":
            float(
                min(
                    item[
                        "recall"
                    ]
                    for item in per_class.values()
                )
            ),

        "per_class":
            per_class,

        "confusion_matrix": {
            "labels":
                list(
                    REDEVELOPMENT_CLASSES
                ),
            "matrix":
                matrix.tolist(),
        },
    }


def _majority_baseline_metrics(
    truth: np.ndarray,
    *,
    majority_class: str,
) -> dict[str, Any]:
    if majority_class not in REDEVELOPMENT_CLASSES:
        raise ValueError(
            "Unexpected redevelopment majority class."
        )

    truth_values = np.asarray(
        truth,
        dtype=str,
    )

    predictions = np.asarray(
        [
            majority_class
        ]
        * truth_values.size,
        dtype=str,
    )

    metrics = redevelopment_diagnosis_metrics(
        truth=truth_values,
        predictions=predictions,
    )

    return {
        "majority_class":
            majority_class,
        "macro_f1":
            metrics[
                "macro_f1"
            ],
        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],
    }


def evaluate_redevelopment_diagnoser_candidate(
    fitted: FittedRedevelopmentDiagnoserCandidate,
    validation_X: np.ndarray,
    validation_binary_y: np.ndarray,
    validation_multiclass_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Evaluate three-class diagnosis on ground-truth active rows only."""

    if (
        fitted.candidate_id
        not in REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS
    ):
        raise ValueError(
            "Expected fitted Phase 5E-R diagnoser candidate."
        )

    X = _validate_feature_matrix(
        validation_X
    )

    binary_y = _validate_binary_target(
        validation_binary_y,
        expected_rows=X.shape[0],
    )

    original_labels = _validate_original_family_target(
        validation_multiclass_y,
        binary_y=binary_y,
        expected_rows=X.shape[0],
    )

    active_X = X[
        binary_y
    ]

    active_original = original_labels[
        binary_y
    ]

    if active_X.shape[0] == 0:
        raise ValueError(
            "Diagnoser evaluation requires active validation rows."
        )

    mapped_truth = map_redevelopment_active_fault_labels(
        active_original,
        protocol_path=protocol_path,
    )

    predictions = fitted.predict(
        active_X
    )

    metrics = redevelopment_diagnosis_metrics(
        truth=mapped_truth,
        predictions=predictions,
    )

    baseline = _majority_baseline_metrics(
        mapped_truth,
        majority_class=fitted.train_majority_class,
    )

    improvement = (
        metrics["macro_f1"]
        - baseline["macro_f1"]
    )

    return {
        "candidate_id":
            fitted.candidate_id,
        "fit_rows":
            int(
                fitted.fit_rows
            ),
        "validation_active_rows":
            int(
                active_X.shape[0]
            ),
        "metrics":
            metrics,
        "majority_baseline":
            baseline,
        "macro_f1_improvement_vs_majority":
            float(
                improvement
            ),
    }


def redevelopment_detector_acceptance(
    evaluation: Mapping[str, Any],
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Apply the frozen Phase 5E-R detector validation gate."""

    protocol = load_phase5er_contract(
        protocol_path
    )

    thresholds = protocol[
        "binary_detector"
    ][
        "acceptance"
    ]

    metrics = evaluation[
        "metrics"
    ]

    minimum_family_recall = min(
        record["recall"]
        for record in metrics[
            "per_active_fault_family_recall"
        ].values()
    )

    checks = {
        "balanced_accuracy":
            bool(
                metrics[
                    "balanced_accuracy"
                ]
                >= thresholds[
                    "balanced_accuracy_min"
                ]
            ),

        "macro_f1":
            bool(
                metrics[
                    "macro_f1"
                ]
                >= thresholds[
                    "macro_f1_min"
                ]
            ),

        "fault_active_recall":
            bool(
                metrics[
                    "active_recall"
                ]
                >= thresholds[
                    "fault_active_recall_min"
                ]
            ),

        "specificity":
            bool(
                metrics[
                    "specificity"
                ]
                >= thresholds[
                    "specificity_min"
                ]
            ),

        "per_active_fault_family_recall":
            bool(
                minimum_family_recall
                >= thresholds[
                    "per_active_fault_family_recall_min"
                ]
            ),
    }

    return {
        "candidate_id":
            evaluation[
                "candidate_id"
            ],
        "checks":
            checks,
        "passed":
            bool(
                all(
                    checks.values()
                )
            ),
        "minimum_per_active_fault_family_recall":
            float(
                minimum_family_recall
            ),
    }


def redevelopment_diagnoser_acceptance(
    evaluation: Mapping[str, Any],
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Apply the frozen Phase 5E-R diagnoser validation gate."""

    protocol = load_phase5er_contract(
        protocol_path
    )

    thresholds = protocol[
        "active_fault_diagnosis"
    ][
        "acceptance"
    ]

    metrics = evaluation[
        "metrics"
    ]

    improvement = evaluation[
        "macro_f1_improvement_vs_majority"
    ]

    checks = {
        "macro_f1":
            bool(
                metrics[
                    "macro_f1"
                ]
                >= thresholds[
                    "macro_f1_min"
                ]
            ),

        "balanced_accuracy":
            bool(
                metrics[
                    "balanced_accuracy"
                ]
                >= thresholds[
                    "balanced_accuracy_min"
                ]
            ),

        "per_class_recall":
            bool(
                metrics[
                    "minimum_per_class_recall"
                ]
                >= thresholds[
                    "per_class_recall_min"
                ]
            ),

        "majority_baseline_improvement":
            bool(
                improvement
                >= thresholds[
                    "majority_baseline_macro_f1_improvement_min"
                ]
            ),
    }

    return {
        "candidate_id":
            evaluation[
                "candidate_id"
            ],
        "checks":
            checks,
        "passed":
            bool(
                all(
                    checks.values()
                )
            ),
    }


def select_redevelopment_detector(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> str | None:
    """Select among passing detector candidates using frozen rules."""

    load_phase5er_contract(
        protocol_path
    )

    passing = []

    priority = {
        candidate_id:
            index
        for index, candidate_id in enumerate(
            REDEVELOPMENT_DETECTOR_CANDIDATE_IDS
        )
    }

    for candidate_id in REDEVELOPMENT_DETECTOR_CANDIDATE_IDS:
        if candidate_id not in evaluations:
            continue

        evaluation = evaluations[
            candidate_id
        ]

        acceptance = redevelopment_detector_acceptance(
            evaluation,
            protocol_path=protocol_path,
        )

        if acceptance[
            "passed"
        ]:
            passing.append(
                candidate_id
            )

    if not passing:
        return None

    return min(
        passing,
        key=lambda candidate_id: (
            -float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "balanced_accuracy"
                ]
            ),
            -float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "macro_f1"
                ]
            ),
            float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "false_positive_rate"
                ]
            ),
            -float(
                evaluations[
                    candidate_id
                ][
                    "threshold"
                ]
            ),
            priority[
                candidate_id
            ],
        ),
    )


def select_redevelopment_diagnoser(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    protocol_path: str | Path = DEFAULT_PHASE5ER_PROTOCOL_PATH,
) -> str | None:
    """Select among passing diagnosers using frozen rules."""

    load_phase5er_contract(
        protocol_path
    )

    passing = []

    priority = {
        candidate_id:
            index
        for index, candidate_id in enumerate(
            REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS
        )
    }

    for candidate_id in REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS:
        if candidate_id not in evaluations:
            continue

        evaluation = evaluations[
            candidate_id
        ]

        acceptance = redevelopment_diagnoser_acceptance(
            evaluation,
            protocol_path=protocol_path,
        )

        if acceptance[
            "passed"
        ]:
            passing.append(
                candidate_id
            )

    if not passing:
        return None

    return min(
        passing,
        key=lambda candidate_id: (
            -float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "macro_f1"
                ]
            ),
            -float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "balanced_accuracy"
                ]
            ),
            -float(
                evaluations[
                    candidate_id
                ][
                    "metrics"
                ][
                    "minimum_per_class_recall"
                ]
            ),
            priority[
                candidate_id
            ],
        ),
    )


def detector_sequence_diagnostics(
    *,
    binary_truth: np.ndarray,
    predictions: np.ndarray,
    episode_ids: np.ndarray,
    step_indices: np.ndarray,
) -> dict[str, Any]:
    """Ordered-step detector diagnostics; step_index is not physical time."""

    truth_raw = np.asarray(
        binary_truth
    )

    predictions_raw = np.asarray(
        predictions
    )

    episodes = np.asarray(
        episode_ids,
        dtype=str,
    )

    steps_raw = np.asarray(
        step_indices
    )

    if not (
        truth_raw.ndim
        == predictions_raw.ndim
        == episodes.ndim
        == steps_raw.ndim
        == 1
    ):
        raise ValueError(
            "Sequence diagnostic inputs must be one-dimensional."
        )

    if not (
        truth_raw.shape
        == predictions_raw.shape
        == episodes.shape
        == steps_raw.shape
    ):
        raise ValueError(
            "Sequence diagnostic input shapes do not match."
        )

    if truth_raw.size == 0:
        raise ValueError(
            "Sequence diagnostics require at least one row."
        )

    if not set(
        truth_raw.tolist()
    ).issubset(
        {
            False,
            True,
            0,
            1,
        }
    ):
        raise ValueError(
            "Sequence truth must be binary."
        )

    if not set(
        predictions_raw.tolist()
    ).issubset(
        {
            False,
            True,
            0,
            1,
        }
    ):
        raise ValueError(
            "Sequence predictions must be binary."
        )

    truth = truth_raw.astype(
        bool,
        copy=False,
    )

    alarms = predictions_raw.astype(
        bool,
        copy=False,
    )

    if not np.all(
        np.isfinite(
            steps_raw.astype(
                float
            )
        )
    ):
        raise ValueError(
            "step_index values must be finite."
        )

    steps = steps_raw.astype(
        np.int64
    )

    if not np.array_equal(
        steps.astype(
            steps_raw.dtype,
            copy=False,
        ),
        steps_raw,
    ):
        # Works for ordinary integer arrays and rejects fractional steps.
        if not np.all(
            np.asarray(
                steps_raw,
                dtype=float,
            )
            == steps
        ):
            raise ValueError(
                "step_index values must be integer-valued."
            )

    records = []

    for episode_id in sorted(
        set(
            episodes.tolist()
        )
    ):
        mask = (
            episodes
            == episode_id
        )

        episode_steps = steps[
            mask
        ]

        if np.unique(
            episode_steps
        ).size != episode_steps.size:
            raise ValueError(
                f"Duplicate step_index within episode {episode_id!r}."
            )

        order = np.argsort(
            episode_steps
        )

        episode_steps = episode_steps[
            order
        ]

        episode_truth = truth[
            mask
        ][
            order
        ]

        episode_alarms = alarms[
            mask
        ][
            order
        ]

        fault_episode = bool(
            np.any(
                episode_truth
            )
        )

        if fault_episode:
            first_active_position = int(
                np.flatnonzero(
                    episode_truth
                )[0]
            )

            first_active_step = int(
                episode_steps[
                    first_active_position
                ]
            )

            pre_active_false_alarm = bool(
                np.any(
                    episode_alarms
                    & (
                        episode_steps
                        < first_active_step
                    )
                )
            )

            after_active = np.flatnonzero(
                episode_alarms
                & (
                    episode_steps
                    >= first_active_step
                )
            )

            if after_active.size:
                first_alarm_step = int(
                    episode_steps[
                        int(
                            after_active[0]
                        )
                    ]
                )

                delay = int(
                    first_alarm_step
                    - first_active_step
                )

                missed = False

            else:
                first_alarm_step = None
                delay = None
                missed = True

            normal_false_alarm = False

        else:
            first_active_step = None
            first_alarm_step = None
            delay = None
            missed = False
            pre_active_false_alarm = False
            normal_false_alarm = bool(
                np.any(
                    episode_alarms
                )
            )

        records.append(
            {
                "episode_id":
                    episode_id,
                "fault_episode":
                    fault_episode,
                "first_active_step":
                    first_active_step,
                "first_alarm_at_or_after_active_step":
                    first_alarm_step,
                "detection_delay_steps":
                    delay,
                "missed_fault_episode":
                    missed,
                "pre_active_false_alarm":
                    pre_active_false_alarm,
                "normal_episode_false_alarm":
                    normal_false_alarm,
            }
        )

    fault_records = [
        record
        for record in records
        if record[
            "fault_episode"
        ]
    ]

    normal_records = [
        record
        for record in records
        if not record[
            "fault_episode"
        ]
    ]

    detected_records = [
        record
        for record in fault_records
        if not record[
            "missed_fault_episode"
        ]
    ]

    delays = [
        record[
            "detection_delay_steps"
        ]
        for record in detected_records
    ]

    summary = {
        "ordered_unit":
            "step_index; not calibrated physical time",

        "fault_episode_count":
            len(
                fault_records
            ),

        "detected_fault_episode_count":
            len(
                detected_records
            ),

        "missed_fault_episode_count":
            (
                len(
                    fault_records
                )
                - len(
                    detected_records
                )
            ),

        "fault_episode_detection_rate":
            (
                float(
                    len(
                        detected_records
                    )
                    / len(
                        fault_records
                    )
                )
                if fault_records
                else 0.0
            ),

        "mean_detection_delay_steps":
            (
                float(
                    np.mean(
                        delays
                    )
                )
                if delays
                else None
            ),

        "median_detection_delay_steps":
            (
                float(
                    np.median(
                        delays
                    )
                )
                if delays
                else None
            ),

        "max_detection_delay_steps":
            (
                int(
                    np.max(
                        delays
                    )
                )
                if delays
                else None
            ),

        "pre_active_false_alarm_episode_count":
            int(
                sum(
                    record[
                        "pre_active_false_alarm"
                    ]
                    for record in fault_records
                )
            ),

        "normal_episode_count":
            len(
                normal_records
            ),

        "normal_episode_false_alarm_count":
            int(
                sum(
                    record[
                        "normal_episode_false_alarm"
                    ]
                    for record in normal_records
                )
            ),
    }

    return {
        "summary":
            summary,
        "episode_records":
            records,
    }
