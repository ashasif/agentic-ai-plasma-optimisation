"""Frozen Phase 5D supervised fault-detection and diagnosis machinery.

This module contains candidate builders, leakage-safe fitting helpers and
validation metric functions for the frozen Phase 5 supervised benchmark.

It deliberately contains no real-dataset loader, no TEST access, no model
selection and no result-artifact writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

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

from plasma_ai.monitoring.data import (
    DEFAULT_PROTOCOL_PATH,
    MODEL_FEATURES,
)


ACTIVE_CLASSES = (
    "flow_delivery",
    "power_coupling",
    "pressure_sensor_bias",
    "pumping_effectiveness",
)

SECONDARY_END_TO_END_CLASSES = (
    "none",
    *ACTIVE_CLASSES,
)

DETECTOR_CANDIDATE_IDS = (
    "D1_logistic_regression",
    "D2_hist_gradient_boosting",
    "D3_extra_trees",
)

DIAGNOSER_CANDIDATE_IDS = (
    "G1_logistic_regression",
    "G2_hist_gradient_boosting",
    "G3_extra_trees",
)


_EXPECTED_DETECTOR_CANDIDATES = {
    "D1_logistic_regression": {
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "parameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
        "threshold": 0.5,
    },
    "D2_hist_gradient_boosting": {
        "classifier": "HistGradientBoostingClassifier",
        "parameters": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.1,
            "early_stopping": False,
            "random_state": 20260924,
        },
        "threshold": 0.5,
    },
    "D3_extra_trees": {
        "classifier": "ExtraTreesClassifier",
        "parameters": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": 20260924,
        },
        "threshold": 0.5,
    },
}


_EXPECTED_DIAGNOSER_CANDIDATES = {
    "G1_logistic_regression": {
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "parameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
        },
    },
    "G2_hist_gradient_boosting": {
        "classifier": "HistGradientBoostingClassifier",
        "parameters": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.1,
            "early_stopping": False,
            "random_state": 20260924,
        },
    },
    "G3_extra_trees": {
        "classifier": "ExtraTreesClassifier",
        "parameters": {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": 20260924,
        },
    },
}


_EXPECTED_VALIDATION_ACCEPTANCE = {
    "binary_detector": {
        "balanced_accuracy_min": 0.75,
        "macro_f1_min": 0.75,
        "fault_active_recall_min": 0.75,
        "specificity_min": 0.75,
        "per_active_fault_family_recall_min": 0.6,
    },
    "active_fault_diagnoser": {
        "macro_f1_min": 0.5,
        "balanced_accuracy_min": 0.5,
        "per_class_recall_min": 0.25,
        "majority_baseline_macro_f1_improvement_min": 0.1,
    },
    "failure_action": "controlled_stop_before_test",
}


_EXPECTED_SELECTION_RULES = {
    "binary_detector": [
        "highest_balanced_accuracy",
        "highest_macro_f1",
        "lowest_false_positive_rate",
        "candidate_priority_D1_D2_D3",
    ],
    "active_fault_diagnoser": [
        "highest_macro_f1",
        "highest_balanced_accuracy",
        "highest_minimum_per_class_recall",
        "candidate_priority_G1_G2_G3",
    ],
}


def load_phase5_supervised_contract(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and verify the exact frozen Phase 5D supervised contract."""

    path = Path(protocol_path)

    protocol = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if protocol.get("phase") != "5":
        raise ValueError(
            "Expected Phase 5 protocol."
        )

    if protocol.get("protocol_state") != "FROZEN":
        raise ValueError(
            "Phase 5 protocol must be FROZEN."
        )

    if (
        protocol.get("subphases", {}).get("5D")
        != "supervised detection and diagnosis benchmark"
    ):
        raise ValueError(
            "Frozen Phase 5D subphase contract drift detected."
        )

    if (
        tuple(protocol.get("model_feature_order", ()))
        != MODEL_FEATURES
    ):
        raise ValueError(
            "Frozen Phase 5 model-feature contract drift detected."
        )

    targets = protocol.get(
        "targets",
        {},
    )

    if tuple(
        targets.get(
            "active_classes",
            (),
        )
    ) != ACTIVE_CLASSES:
        raise ValueError(
            "Frozen active fault-class contract drift detected."
        )

    if tuple(
        targets.get(
            "secondary_end_to_end_classes",
            (),
        )
    ) != SECONDARY_END_TO_END_CLASSES:
        raise ValueError(
            "Frozen end-to-end class contract drift detected."
        )

    if (
        protocol.get("detector_candidates")
        != _EXPECTED_DETECTOR_CANDIDATES
    ):
        raise ValueError(
            "Frozen detector-candidate contract drift detected."
        )

    if (
        protocol.get("diagnoser_candidates")
        != _EXPECTED_DIAGNOSER_CANDIDATES
    ):
        raise ValueError(
            "Frozen diagnoser-candidate contract drift detected."
        )

    if (
        protocol.get("validation_acceptance")
        != _EXPECTED_VALIDATION_ACCEPTANCE
    ):
        raise ValueError(
            "Frozen validation-acceptance contract drift detected."
        )

    if (
        protocol.get("selection_rules")
        != _EXPECTED_SELECTION_RULES
    ):
        raise ValueError(
            "Frozen selection-rule contract drift detected."
        )

    final_refit = protocol.get(
        "final_refit",
        {},
    )

    if final_refit.get(
        "test_rows_for_fit"
    ) != 0:
        raise ValueError(
            "Frozen Phase 5 final-refit TEST lock drift detected."
        )

    return {
        "detector_candidates":
            protocol["detector_candidates"],
        "diagnoser_candidates":
            protocol["diagnoser_candidates"],
        "validation_acceptance":
            protocol["validation_acceptance"],
        "selection_rules":
            protocol["selection_rules"],
        "final_refit":
            protocol["final_refit"],
        "targets":
            protocol["targets"],
    }


def _validate_feature_matrix(
    X: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        X,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Phase 5 supervised feature matrix must be two-dimensional."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Phase 5 supervised feature matrix must contain rows."
        )

    if matrix.shape[1] != len(
        MODEL_FEATURES
    ):
        raise ValueError(
            "Phase 5 supervised feature matrix has wrong feature width."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Phase 5 supervised feature matrix contains non-finite values."
        )

    return matrix


def _validate_binary_target(
    y: np.ndarray,
    *,
    rows: int,
) -> np.ndarray:
    target = np.asarray(y)

    if target.ndim != 1:
        raise ValueError(
            "Phase 5 binary target must be one-dimensional."
        )

    if target.shape[0] != rows:
        raise ValueError(
            "Phase 5 binary target row-count mismatch."
        )

    if target.dtype != np.bool_:
        if not np.all(
            np.isin(
                target,
                [False, True],
            )
        ):
            raise ValueError(
                "Phase 5 binary target must contain only boolean values."
            )

        target = target.astype(
            np.bool_,
            copy=False,
        )

    return target


def _validate_family_target(
    families: np.ndarray,
    *,
    binary_y: np.ndarray,
    rows: int,
) -> np.ndarray:
    target = np.asarray(
        families,
        dtype=str,
    )

    if target.ndim != 1:
        raise ValueError(
            "Phase 5 active_fault_family target must be one-dimensional."
        )

    if target.shape[0] != rows:
        raise ValueError(
            "Phase 5 active_fault_family row-count mismatch."
        )

    binary = _validate_binary_target(
        binary_y,
        rows=rows,
    )

    allowed = set(
        SECONDARY_END_TO_END_CLASSES
    )

    unexpected = sorted(
        set(
            target.tolist()
        )
        - allowed
    )

    if unexpected:
        raise ValueError(
            "Unexpected Phase 5 fault-family label(s): "
            f"{unexpected}"
        )

    inactive_invalid = (
        (~binary)
        & (target != "none")
    )

    if np.any(
        inactive_invalid
    ):
        raise ValueError(
            "Inactive rows must have active_fault_family == 'none'."
        )

    active_invalid = (
        binary
        & (~np.isin(
            target,
            ACTIVE_CLASSES,
        ))
    )

    if np.any(
        active_invalid
    ):
        raise ValueError(
            "Active rows must use one of the frozen active fault classes."
        )

    return target


def _build_classifier(
    *,
    classifier_name: str,
    parameters: dict[str, Any],
    scaler: str | None,
):
    if classifier_name == "LogisticRegression":
        classifier = LogisticRegression(
            **parameters
        )

        if scaler != "StandardScaler":
            raise ValueError(
                "Frozen logistic candidate requires StandardScaler."
            )

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

    if scaler is not None:
        raise ValueError(
            "Unexpected scaler for non-logistic frozen candidate."
        )

    if classifier_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            **parameters
        )

    if classifier_name == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            **parameters
        )

    raise ValueError(
        f"Unknown frozen classifier: {classifier_name!r}"
    )


def build_detector_candidate(
    candidate_id: str,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
):
    """Build one exact frozen detector candidate."""

    contract = load_phase5_supervised_contract(
        protocol_path
    )

    configs = contract[
        "detector_candidates"
    ]

    if candidate_id not in configs:
        raise ValueError(
            f"Unknown frozen detector candidate: {candidate_id!r}"
        )

    config = configs[
        candidate_id
    ]

    return _build_classifier(
        classifier_name=config[
            "classifier"
        ],
        parameters=dict(
            config["parameters"]
        ),
        scaler=config.get(
            "scaler"
        ),
    )


def build_diagnoser_candidate(
    candidate_id: str,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
):
    """Build one exact frozen active-fault diagnoser candidate."""

    contract = load_phase5_supervised_contract(
        protocol_path
    )

    configs = contract[
        "diagnoser_candidates"
    ]

    if candidate_id not in configs:
        raise ValueError(
            f"Unknown frozen diagnoser candidate: {candidate_id!r}"
        )

    config = configs[
        candidate_id
    ]

    return _build_classifier(
        classifier_name=config[
            "classifier"
        ],
        parameters=dict(
            config["parameters"]
        ),
        scaler=config.get(
            "scaler"
        ),
    )


def _active_probability(
    model,
    X: np.ndarray,
) -> np.ndarray:
    matrix = _validate_feature_matrix(
        X
    )

    probabilities = np.asarray(
        model.predict_proba(
            matrix
        ),
        dtype=np.float64,
    )

    classes = np.asarray(
        model.classes_
    )

    matches = np.flatnonzero(
        classes == True
    )

    if matches.size != 1:
        raise ValueError(
            "Detector model must expose exactly one active=True class."
        )

    scores = probabilities[
        :,
        int(matches[0]),
    ]

    if scores.shape != (
        matrix.shape[0],
    ):
        raise ValueError(
            "Detector active-probability shape mismatch."
        )

    if not np.all(
        np.isfinite(scores)
    ):
        raise ValueError(
            "Detector active probabilities contain non-finite values."
        )

    if np.any(
        (scores < 0.0)
        | (scores > 1.0)
    ):
        raise ValueError(
            "Detector active probabilities must lie in [0, 1]."
        )

    return scores


@dataclass(frozen=True)
class FittedDetectorCandidate:
    """One fitted frozen binary detector candidate."""

    candidate_id: str
    model: Any
    threshold: float
    fit_rows: int

    def predict_active_probability(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return _active_probability(
            self.model,
            X,
        )

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


@dataclass(frozen=True)
class FittedDiagnoserCandidate:
    """One fitted frozen active-fault diagnosis candidate."""

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

        if predictions.shape != (
            matrix.shape[0],
        ):
            raise ValueError(
                "Diagnoser prediction shape mismatch."
            )

        unexpected = sorted(
            set(
                predictions.tolist()
            )
            - set(
                ACTIVE_CLASSES
            )
        )

        if unexpected:
            raise ValueError(
                "Diagnoser emitted unexpected class(es): "
                f"{unexpected}"
            )

        return predictions


def fit_detector_candidate(
    candidate_id: str,
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> FittedDetectorCandidate:
    """Fit one detector on all supplied TRAIN rows."""

    matrix = _validate_feature_matrix(
        train_X
    )

    target = _validate_binary_target(
        train_binary_y,
        rows=matrix.shape[0],
    )

    unique = np.unique(
        target
    )

    if unique.size != 2:
        raise ValueError(
            "Detector training requires both inactive and active classes."
        )

    contract = load_phase5_supervised_contract(
        protocol_path
    )

    config = contract[
        "detector_candidates"
    ][
        candidate_id
    ] if candidate_id in contract[
        "detector_candidates"
    ] else None

    if config is None:
        raise ValueError(
            f"Unknown frozen detector candidate: {candidate_id!r}"
        )

    model = build_detector_candidate(
        candidate_id,
        protocol_path=protocol_path,
    )

    model.fit(
        matrix,
        target,
    )

    return FittedDetectorCandidate(
        candidate_id=candidate_id,
        model=model,
        threshold=float(
            config["threshold"]
        ),
        fit_rows=int(
            matrix.shape[0]
        ),
    )


def _majority_active_class(
    active_families: np.ndarray,
) -> str:
    values = np.asarray(
        active_families,
        dtype=str,
    )

    counts = {
        name: int(
            np.sum(
                values == name
            )
        )
        for name in ACTIVE_CLASSES
    }

    maximum = max(
        counts.values()
    )

    # Frozen class order provides deterministic tie resolution.
    for name in ACTIVE_CLASSES:
        if counts[name] == maximum:
            return name

    raise RuntimeError(
        "Unable to determine TRAIN active-fault majority class."
    )


def fit_diagnoser_candidate(
    candidate_id: str,
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    train_multiclass_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> FittedDiagnoserCandidate:
    """Fit one diagnoser using only ground-truth active TRAIN rows."""

    matrix = _validate_feature_matrix(
        train_X
    )

    binary = _validate_binary_target(
        train_binary_y,
        rows=matrix.shape[0],
    )

    families = _validate_family_target(
        train_multiclass_y,
        binary_y=binary,
        rows=matrix.shape[0],
    )

    active_matrix = matrix[
        binary
    ]

    active_families = families[
        binary
    ]

    if active_matrix.shape[0] == 0:
        raise ValueError(
            "Diagnoser training has no active fault rows."
        )

    represented = set(
        active_families.tolist()
    )

    if represented != set(
        ACTIVE_CLASSES
    ):
        raise ValueError(
            "Diagnoser training must represent all four frozen "
            "active fault classes."
        )

    model = build_diagnoser_candidate(
        candidate_id,
        protocol_path=protocol_path,
    )

    model.fit(
        active_matrix,
        active_families,
    )

    return FittedDiagnoserCandidate(
        candidate_id=candidate_id,
        model=model,
        fit_rows=int(
            active_matrix.shape[0]
        ),
        train_majority_class=
            _majority_active_class(
                active_families
            ),
    )


def detector_validation_metrics(
    *,
    binary_truth: np.ndarray,
    predictions: np.ndarray,
    active_probability: np.ndarray,
    active_fault_family: np.ndarray,
) -> dict[str, Any]:
    """Calculate frozen detector validation metrics."""

    truth = np.asarray(
        binary_truth,
        dtype=np.bool_,
    )

    predicted = np.asarray(
        predictions,
        dtype=np.bool_,
    )

    scores = np.asarray(
        active_probability,
        dtype=np.float64,
    )

    families = np.asarray(
        active_fault_family,
        dtype=str,
    )

    if not (
        truth.ndim
        == predicted.ndim
        == scores.ndim
        == families.ndim
        == 1
    ):
        raise ValueError(
            "Detector metric inputs must be one-dimensional."
        )

    if not (
        truth.shape
        == predicted.shape
        == scores.shape
        == families.shape
    ):
        raise ValueError(
            "Detector metric input shapes do not match."
        )

    if truth.size == 0:
        raise ValueError(
            "Detector metrics require rows."
        )

    if not np.all(
        np.isfinite(scores)
    ):
        raise ValueError(
            "Detector scores contain non-finite values."
        )

    matrix = confusion_matrix(
        truth,
        predicted,
        labels=[
            False,
            True,
        ],
    )

    tn = int(
        matrix[0, 0]
    )
    fp = int(
        matrix[0, 1]
    )
    fn = int(
        matrix[1, 0]
    )
    tp = int(
        matrix[1, 1]
    )

    negative = (
        tn + fp
    )

    specificity = (
        tn / negative
        if negative
        else 0.0
    )

    false_positive_rate = (
        fp / negative
        if negative
        else 0.0
    )

    per_family = {}

    for family in ACTIVE_CLASSES:
        mask = (
            families == family
        )

        support = int(
            np.sum(mask)
        )

        detected = int(
            np.sum(
                predicted[mask]
            )
        )

        recall = (
            detected / support
            if support
            else 0.0
        )

        per_family[
            family
        ] = {
            "support":
                support,
            "detected":
                detected,
            "recall":
                float(recall),
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
                    average="macro",
                    zero_division=0,
                )
            ),
        "positive_precision":
            float(
                precision_score(
                    truth,
                    predicted,
                    zero_division=0,
                )
            ),
        "active_recall":
            float(
                recall_score(
                    truth,
                    predicted,
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
                    scores,
                )
            ),
        "average_precision":
            float(
                average_precision_score(
                    truth,
                    scores,
                )
            ),
        "confusion_matrix": {
            "labels": [
                "inactive",
                "active",
            ],
            "matrix":
                matrix.tolist(),
            "true_negative":
                tn,
            "false_positive":
                fp,
            "false_negative":
                fn,
            "true_positive":
                tp,
        },
        "per_active_fault_family_recall":
            per_family,
    }


def evaluate_detector_candidate(
    fitted: FittedDetectorCandidate,
    validation_X: np.ndarray,
    validation_binary_y: np.ndarray,
    validation_multiclass_y: np.ndarray,
) -> dict[str, Any]:
    """Score one fitted detector on supplied validation rows."""

    matrix = _validate_feature_matrix(
        validation_X
    )

    binary = _validate_binary_target(
        validation_binary_y,
        rows=matrix.shape[0],
    )

    families = _validate_family_target(
        validation_multiclass_y,
        binary_y=binary,
        rows=matrix.shape[0],
    )

    probabilities = (
        fitted.predict_active_probability(
            matrix
        )
    )

    predictions = (
        probabilities
        >= fitted.threshold
    )

    metrics = detector_validation_metrics(
        binary_truth=binary,
        predictions=predictions,
        active_probability=probabilities,
        active_fault_family=families,
    )

    return {
        "candidate_id":
            fitted.candidate_id,
        "fit_rows":
            int(
                fitted.fit_rows
            ),
        "threshold":
            float(
                fitted.threshold
            ),
        "metrics":
            metrics,
    }


def diagnosis_metrics(
    *,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Calculate active-fault four-class diagnosis metrics."""

    truth_values = np.asarray(
        truth,
        dtype=str,
    )

    predicted_values = np.asarray(
        predictions,
        dtype=str,
    )

    if (
        truth_values.ndim != 1
        or predicted_values.ndim != 1
    ):
        raise ValueError(
            "Diagnosis metric inputs must be one-dimensional."
        )

    if (
        truth_values.shape
        != predicted_values.shape
    ):
        raise ValueError(
            "Diagnosis metric input shapes do not match."
        )

    if truth_values.size == 0:
        raise ValueError(
            "Diagnosis metrics require active rows."
        )

    allowed = set(
        ACTIVE_CLASSES
    )

    if (
        set(
            truth_values.tolist()
        )
        - allowed
    ):
        raise ValueError(
            "Diagnosis truth contains non-active class labels."
        )

    if (
        set(
            predicted_values.tolist()
        )
        - allowed
    ):
        raise ValueError(
            "Diagnosis predictions contain non-active class labels."
        )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            truth_values,
            predicted_values,
            labels=list(
                ACTIVE_CLASSES
            ),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        truth_values,
        predicted_values,
        labels=list(
            ACTIVE_CLASSES
        ),
    )

    per_class = {}

    for index, name in enumerate(
        ACTIVE_CLASSES
    ):
        per_class[name] = {
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
                        ACTIVE_CLASSES
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
                np.min(
                    recall
                )
            ),
        "per_class":
            per_class,
        "confusion_matrix": {
            "labels":
                list(
                    ACTIVE_CLASSES
                ),
            "matrix":
                matrix.tolist(),
        },
    }


def _majority_baseline_metrics(
    *,
    validation_truth: np.ndarray,
    majority_class: str,
) -> dict[str, Any]:
    if majority_class not in ACTIVE_CLASSES:
        raise ValueError(
            "Majority baseline class must be a frozen active class."
        )

    truth = np.asarray(
        validation_truth,
        dtype=str,
    )

    predictions = np.asarray(
        [majority_class] * truth.size,
        dtype=str,
    )

    metrics = diagnosis_metrics(
        truth=truth,
        predictions=predictions,
    )

    return {
        "majority_class":
            majority_class,
        "macro_f1":
            float(
                metrics[
                    "macro_f1"
                ]
            ),
        "balanced_accuracy":
            float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
    }


def evaluate_diagnoser_candidate(
    fitted: FittedDiagnoserCandidate,
    validation_X: np.ndarray,
    validation_binary_y: np.ndarray,
    validation_multiclass_y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate diagnosis only on ground-truth active validation rows."""

    matrix = _validate_feature_matrix(
        validation_X
    )

    binary = _validate_binary_target(
        validation_binary_y,
        rows=matrix.shape[0],
    )

    families = _validate_family_target(
        validation_multiclass_y,
        binary_y=binary,
        rows=matrix.shape[0],
    )

    active_matrix = matrix[
        binary
    ]

    active_truth = families[
        binary
    ]

    if active_matrix.shape[0] == 0:
        raise ValueError(
            "Diagnoser validation has no active rows."
        )

    predictions = fitted.predict(
        active_matrix
    )

    metrics = diagnosis_metrics(
        truth=active_truth,
        predictions=predictions,
    )

    majority = _majority_baseline_metrics(
        validation_truth=active_truth,
        majority_class=
            fitted.train_majority_class,
    )

    improvement = (
        metrics["macro_f1"]
        - majority["macro_f1"]
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
                active_matrix.shape[0]
            ),
        "metrics":
            metrics,
        "majority_baseline":
            majority,
        "macro_f1_improvement_vs_majority":
            float(
                improvement
            ),
    }
