"""Synthetic tests for the frozen Phase 5D supervised benchmark.

No real monitoring dataset is loaded by this module and no Phase 5 TEST
payload is accessed.
"""

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plasma_ai.monitoring.data import MODEL_FEATURES
from plasma_ai.monitoring.supervised import (
    ACTIVE_CLASSES,
    DETECTOR_CANDIDATE_IDS,
    DIAGNOSER_CANDIDATE_IDS,
    build_detector_candidate,
    build_diagnoser_candidate,
    detector_validation_metrics,
    diagnosis_metrics,
    evaluate_detector_candidate,
    evaluate_diagnoser_candidate,
    fit_detector_candidate,
    fit_diagnoser_candidate,
    load_phase5_supervised_contract,
)


def _synthetic_monitoring_data():
    rng = np.random.default_rng(
        20260924
    )

    healthy_count = 160

    active_counts = {
        "flow_delivery": 60,
        "power_coupling": 40,
        "pressure_sensor_bias": 35,
        "pumping_effectiveness": 25,
    }

    total_active = sum(
        active_counts.values()
    )

    rows = (
        healthy_count
        + total_active
    )

    X = rng.normal(
        0.0,
        0.2,
        size=(
            rows,
            len(MODEL_FEATURES),
        ),
    )

    # Generic positive process-variable region.
    X[:, 0] += 50.0
    X[:, 1] += 30.0
    X[:, 2] += 20.0
    X[:, 3] += 50.0
    X[:, 4] += 20.0
    X[:, 5] += 30.0

    # Residual channels are small in healthy state.
    X[:, 6:] = rng.normal(
        0.0,
        0.01,
        size=(
            rows,
            3,
        ),
    )

    binary = np.zeros(
        rows,
        dtype=np.bool_,
    )

    families = np.full(
        rows,
        "none",
        dtype="<U32",
    )

    cursor = healthy_count

    for class_index, (
        family,
        count,
    ) in enumerate(
        active_counts.items()
    ):
        start = cursor
        stop = cursor + count

        binary[
            start:stop
        ] = True

        families[
            start:stop
        ] = family

        # Give each synthetic fault class a deterministic signal.
        if family == "flow_delivery":
            X[start:stop, 7] -= 0.08
            X[start:stop, 4] -= 1.0

        elif family == "power_coupling":
            X[start:stop, 6] -= 0.08
            X[start:stop, 3] -= 1.0

        elif family == "pressure_sensor_bias":
            X[start:stop, 8] += 0.08
            X[start:stop, 5] += 1.0

        elif family == "pumping_effectiveness":
            X[start:stop, 8] -= 0.08
            X[start:stop, 1] += 1.0

        cursor = stop

    assert cursor == rows

    return X, binary, families


def test_frozen_supervised_contract_is_exact():
    contract = load_phase5_supervised_contract()

    assert tuple(
        contract[
            "detector_candidates"
        ]
    ) == DETECTOR_CANDIDATE_IDS

    assert tuple(
        contract[
            "diagnoser_candidates"
        ]
    ) == DIAGNOSER_CANDIDATE_IDS

    assert tuple(
        contract[
            "targets"
        ][
            "active_classes"
        ]
    ) == ACTIVE_CLASSES

    assert (
        contract[
            "validation_acceptance"
        ][
            "failure_action"
        ]
        == "controlled_stop_before_test"
    )

    assert (
        contract[
            "final_refit"
        ][
            "test_rows_for_fit"
        ]
        == 0
    )


def test_logistic_detector_builder_is_exact_pipeline():
    model = build_detector_candidate(
        "D1_logistic_regression"
    )

    assert isinstance(
        model,
        Pipeline,
    )

    assert isinstance(
        model.named_steps[
            "scaler"
        ],
        StandardScaler,
    )

    classifier = model.named_steps[
        "classifier"
    ]

    assert isinstance(
        classifier,
        LogisticRegression,
    )

    params = classifier.get_params()

    assert params["C"] == 1.0
    assert params["solver"] == "lbfgs"
    assert params["max_iter"] == 2000
    assert params["class_weight"] == "balanced"


def test_hist_gradient_detector_builder_is_exact():
    model = build_detector_candidate(
        "D2_hist_gradient_boosting"
    )

    assert isinstance(
        model,
        HistGradientBoostingClassifier,
    )

    params = model.get_params()

    assert params["learning_rate"] == 0.05
    assert params["max_iter"] == 300
    assert params["max_leaf_nodes"] == 15
    assert params["l2_regularization"] == 0.1
    assert params["early_stopping"] is False
    assert params["random_state"] == 20260924


def test_extra_trees_detector_builder_is_exact():
    model = build_detector_candidate(
        "D3_extra_trees"
    )

    assert isinstance(
        model,
        ExtraTreesClassifier,
    )

    params = model.get_params()

    assert params["n_estimators"] == 500
    assert params["max_depth"] is None
    assert params["min_samples_leaf"] == 1
    assert params["max_features"] == 1.0
    assert params["class_weight"] == "balanced"
    assert params["n_jobs"] == 1
    assert params["random_state"] == 20260924


@pytest.mark.parametrize(
    "candidate_id",
    DIAGNOSER_CANDIDATE_IDS,
)
def test_all_diagnoser_builders_have_expected_classifier(
    candidate_id,
):
    model = build_diagnoser_candidate(
        candidate_id
    )

    if candidate_id.startswith(
        "G1_"
    ):
        assert isinstance(
            model,
            Pipeline,
        )

        assert isinstance(
            model.named_steps[
                "scaler"
            ],
            StandardScaler,
        )

        assert isinstance(
            model.named_steps[
                "classifier"
            ],
            LogisticRegression,
        )

    elif candidate_id.startswith(
        "G2_"
    ):
        assert isinstance(
            model,
            HistGradientBoostingClassifier,
        )

    elif candidate_id.startswith(
        "G3_"
    ):
        assert isinstance(
            model,
            ExtraTreesClassifier,
        )

    else:
        raise AssertionError(
            candidate_id
        )


@pytest.mark.parametrize(
    "candidate_id",
    DETECTOR_CANDIDATE_IDS,
)
def test_all_detector_candidates_fit_synthetic_data(
    candidate_id,
):
    X, binary, _ = (
        _synthetic_monitoring_data()
    )

    fitted = fit_detector_candidate(
        candidate_id,
        X,
        binary,
    )

    assert fitted.fit_rows == X.shape[0]
    assert fitted.threshold == 0.5

    probabilities = (
        fitted.predict_active_probability(
            X
        )
    )

    predictions = fitted.predict(
        X
    )

    assert probabilities.shape == (
        X.shape[0],
    )

    assert predictions.shape == (
        X.shape[0],
    )

    assert predictions.dtype == np.bool_

    assert np.all(
        np.isfinite(
            probabilities
        )
    )

    assert np.all(
        (probabilities >= 0.0)
        & (probabilities <= 1.0)
    )


@pytest.mark.parametrize(
    "candidate_id",
    DIAGNOSER_CANDIDATE_IDS,
)
def test_all_diagnoser_candidates_fit_only_active_synthetic_rows(
    candidate_id,
):
    X, binary, families = (
        _synthetic_monitoring_data()
    )

    fitted = fit_diagnoser_candidate(
        candidate_id,
        X,
        binary,
        families,
    )

    assert fitted.fit_rows == int(
        np.sum(
            binary
        )
    )

    assert (
        fitted.train_majority_class
        == "flow_delivery"
    )

    predictions = fitted.predict(
        X[
            binary
        ]
    )

    assert predictions.shape == (
        int(
            np.sum(binary)
        ),
    )

    assert set(
        predictions.tolist()
    ).issubset(
        set(
            ACTIVE_CLASSES
        )
    )


def test_detector_metrics_match_known_confusion_values():
    truth = np.asarray(
        [
            False,
            False,
            True,
            True,
        ],
        dtype=np.bool_,
    )

    predicted = np.asarray(
        [
            False,
            True,
            True,
            False,
        ],
        dtype=np.bool_,
    )

    scores = np.asarray(
        [
            0.1,
            0.8,
            0.9,
            0.2,
        ],
        dtype=np.float64,
    )

    families = np.asarray(
        [
            "none",
            "none",
            "flow_delivery",
            "power_coupling",
        ],
        dtype=str,
    )

    metrics = detector_validation_metrics(
        binary_truth=truth,
        predictions=predicted,
        active_probability=scores,
        active_fault_family=families,
    )

    confusion = metrics[
        "confusion_matrix"
    ]

    assert confusion[
        "true_negative"
    ] == 1

    assert confusion[
        "false_positive"
    ] == 1

    assert confusion[
        "false_negative"
    ] == 1

    assert confusion[
        "true_positive"
    ] == 1

    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "specificity"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "active_recall"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "per_active_fault_family_recall"
    ][
        "flow_delivery"
    ][
        "recall"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "per_active_fault_family_recall"
    ][
        "power_coupling"
    ][
        "recall"
    ] == pytest.approx(
        0.0
    )


def test_diagnosis_metrics_match_perfect_predictions():
    truth = np.asarray(
        ACTIVE_CLASSES,
        dtype=str,
    )

    metrics = diagnosis_metrics(
        truth=truth,
        predictions=truth.copy(),
    )

    assert metrics[
        "macro_f1"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "minimum_per_class_recall"
    ] == pytest.approx(
        1.0
    )

    for name in ACTIVE_CLASSES:
        assert metrics[
            "per_class"
        ][
            name
        ][
            "recall"
        ] == pytest.approx(
            1.0
        )


def test_detector_evaluation_returns_frozen_metric_structure():
    X, binary, families = (
        _synthetic_monitoring_data()
    )

    fitted = fit_detector_candidate(
        "D1_logistic_regression",
        X,
        binary,
    )

    result = evaluate_detector_candidate(
        fitted,
        X,
        binary,
        families,
    )

    assert result[
        "candidate_id"
    ] == "D1_logistic_regression"

    assert result[
        "fit_rows"
    ] == X.shape[0]

    assert result[
        "threshold"
    ] == 0.5

    metrics = result[
        "metrics"
    ]

    for key in (
        "balanced_accuracy",
        "macro_f1",
        "positive_precision",
        "active_recall",
        "specificity",
        "false_positive_rate",
        "auroc",
        "average_precision",
        "confusion_matrix",
        "per_active_fault_family_recall",
    ):
        assert key in metrics


def test_diagnoser_evaluation_uses_active_rows_and_majority_baseline():
    X, binary, families = (
        _synthetic_monitoring_data()
    )

    fitted = fit_diagnoser_candidate(
        "G1_logistic_regression",
        X,
        binary,
        families,
    )

    result = evaluate_diagnoser_candidate(
        fitted,
        X,
        binary,
        families,
    )

    assert result[
        "candidate_id"
    ] == "G1_logistic_regression"

    assert result[
        "fit_rows"
    ] == int(
        np.sum(binary)
    )

    assert result[
        "validation_active_rows"
    ] == int(
        np.sum(binary)
    )

    assert result[
        "majority_baseline"
    ][
        "majority_class"
    ] == "flow_delivery"

    assert np.isfinite(
        result[
            "macro_f1_improvement_vs_majority"
        ]
    )


def test_diagnoser_rejects_inactive_row_with_active_family():
    X, binary, families = (
        _synthetic_monitoring_data()
    )

    families = families.copy()

    inactive_index = int(
        np.flatnonzero(
            ~binary
        )[0]
    )

    families[
        inactive_index
    ] = "flow_delivery"

    with pytest.raises(
        ValueError,
        match="Inactive rows",
    ):
        fit_diagnoser_candidate(
            "G1_logistic_regression",
            X,
            binary,
            families,
        )


def test_unknown_candidate_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown frozen detector candidate",
    ):
        build_detector_candidate(
            "D999_unknown"
        )

    with pytest.raises(
        ValueError,
        match="Unknown frozen diagnoser candidate",
    ):
        build_diagnoser_candidate(
            "G999_unknown"
        )


def test_supervised_fit_rejects_nonfinite_features():
    X, binary, _ = (
        _synthetic_monitoring_data()
    )

    X = X.copy()
    X[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        fit_detector_candidate(
            "D1_logistic_regression",
            X,
            binary,
        )
