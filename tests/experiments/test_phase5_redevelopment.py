"""Synthetic-only qualification for Phase 5E-R redevelopment."""

from __future__ import annotations

from dataclasses import is_dataclass

import json

import numpy as np
import pytest

from plasma_ai.monitoring.data import MODEL_FEATURES
from plasma_ai.monitoring.redevelopment import (
    DEFAULT_PHASE5ER_PROTOCOL_PATH,
    FittedRedevelopmentDiagnoserCandidate,
    ORIGINAL_ACTIVE_CLASSES,
    REDEVELOPMENT_CLASSES,
    REDEVELOPMENT_DETECTOR_CANDIDATE_IDS,
    REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS,
    detector_sequence_diagnostics,
    evaluate_redevelopment_detector_candidate,
    evaluate_redevelopment_diagnoser_candidate,
    fit_redevelopment_detector_candidate,
    fit_redevelopment_diagnoser_candidate,
    load_phase5er_contract,
    map_redevelopment_active_fault_labels,
    redevelopment_detector_acceptance,
    redevelopment_diagnoser_acceptance,
    select_redevelopment_detector,
    select_redevelopment_diagnoser,
)


def _binary_fixture(
    *,
    seed: int = 101,
    rows: int = 240,
):
    rng = np.random.default_rng(
        seed
    )

    X = rng.normal(
        size=(
            rows,
            len(
                MODEL_FEATURES
            ),
        )
    )

    score = (
        X[:, 0]
        + 0.45 * X[:, 1]
        - 0.30 * X[:, 2]
    )

    y = (
        score
        > np.median(
            score
        )
    )

    active_labels = np.asarray(
        [
            ORIGINAL_ACTIVE_CLASSES[
                index
                % len(
                    ORIGINAL_ACTIVE_CLASSES
                )
            ]
            for index in range(
                int(
                    np.sum(
                        y
                    )
                )
            )
        ],
        dtype=str,
    )

    family = np.full(
        rows,
        "none",
        dtype="<U32",
    )

    family[
        y
    ] = active_labels

    return (
        X,
        y,
        family,
    )


def _diagnosis_fixture(
    *,
    seed: int = 202,
):
    rng = np.random.default_rng(
        seed
    )

    source_labels = (
        "flow_delivery",
        "power_coupling",
        "pressure_sensor_bias",
        "pumping_effectiveness",
    )

    blocks = []
    labels = []

    for index, label in enumerate(
        source_labels
    ):
        block = rng.normal(
            loc=float(
                index
            )
            * 2.0,
            scale=0.25,
            size=(
                45,
                len(
                    MODEL_FEATURES
                ),
            ),
        )

        blocks.append(
            block
        )

        labels.extend(
            [label]
            * block.shape[0]
        )

    X = np.vstack(
        blocks
    )

    binary_y = np.ones(
        X.shape[0],
        dtype=bool,
    )

    multiclass_y = np.asarray(
        labels,
        dtype=str,
    )

    return (
        X,
        binary_y,
        multiclass_y,
    )


def _passing_detector_evaluation(
    candidate_id: str,
    *,
    balanced_accuracy: float = 0.90,
    macro_f1: float = 0.89,
    false_positive_rate: float = 0.05,
    threshold: float = 0.45,
):
    return {
        "candidate_id":
            candidate_id,
        "threshold":
            threshold,
        "metrics": {
            "balanced_accuracy":
                balanced_accuracy,
            "macro_f1":
                macro_f1,
            "active_recall":
                0.85,
            "specificity":
                0.92,
            "false_positive_rate":
                false_positive_rate,
            "per_active_fault_family_recall": {
                family: {
                    "support":
                        10,
                    "detected":
                        8,
                    "recall":
                        0.80,
                }
                for family in ORIGINAL_ACTIVE_CLASSES
            },
        },
    }


def _passing_diagnoser_evaluation(
    candidate_id: str,
    *,
    macro_f1: float = 0.80,
    balanced_accuracy: float = 0.81,
    minimum_recall: float = 0.70,
):
    return {
        "candidate_id":
            candidate_id,
        "metrics": {
            "macro_f1":
                macro_f1,
            "balanced_accuracy":
                balanced_accuracy,
            "minimum_per_class_recall":
                minimum_recall,
        },
        "macro_f1_improvement_vs_majority":
            0.25,
    }


def test_frozen_protocol_contract_loads():
    protocol = load_phase5er_contract()

    assert (
        protocol["phase"]
        == "5E-R"
    )

    assert (
        protocol["protocol_state"]
        == "FROZEN"
    )

    assert tuple(
        protocol[
            "feature_contract"
        ][
            "model_feature_order"
        ]
    ) == tuple(
        MODEL_FEATURES
    )


def test_protocol_has_no_test_development_access():
    protocol = load_phase5er_contract()

    assert (
        protocol["governance"][
            "test_access_during_redevelopment"
        ]
        is False
    )

    assert (
        protocol["final_refit"][
            "test_rows_for_fit"
        ]
        == 0
    )


def test_class_mapping_exact():
    source = np.asarray(
        [
            "flow_delivery",
            "power_coupling",
            "pressure_sensor_bias",
            "pumping_effectiveness",
        ],
        dtype=str,
    )

    mapped = (
        map_redevelopment_active_fault_labels(
            source
        )
    )

    assert mapped.tolist() == [
        "flow_delivery",
        "power_coupling",
        "pressure_path_anomaly",
        "pressure_path_anomaly",
    ]


@pytest.mark.parametrize(
    "bad_label",
    [
        "none",
        "unknown",
        "pressure_path_anomaly",
    ],
)
def test_mapping_rejects_non_original_active_labels(
    bad_label,
):
    with pytest.raises(
        ValueError
    ):
        map_redevelopment_active_fault_labels(
            np.asarray(
                [
                    bad_label
                ],
                dtype=str,
            )
        )


@pytest.mark.parametrize(
    "candidate_id,expected_threshold",
    [
        (
            "ERD1_hgb_threshold_045",
            0.45,
        ),
        (
            "ERD2_hgb_threshold_040",
            0.40,
        ),
        (
            "ERD3_extra_trees_threshold_045",
            0.45,
        ),
        (
            "ERD4_extra_trees_threshold_040",
            0.40,
        ),
    ],
)
def test_redevelopment_detector_fit_uses_frozen_threshold(
    candidate_id,
    expected_threshold,
):
    X, y, _ = _binary_fixture()

    fitted = (
        fit_redevelopment_detector_candidate(
            candidate_id,
            X,
            y,
        )
    )

    assert fitted.candidate_id == candidate_id
    assert fitted.threshold == expected_threshold
    assert fitted.fit_rows == X.shape[0]

    probability = (
        fitted.predict_active_probability(
            X[:15]
        )
    )

    assert probability.shape == (
        15,
    )

    assert np.all(
        probability
        >= 0.0
    )

    assert np.all(
        probability
        <= 1.0
    )


def test_hgb_threshold_candidates_have_identical_probabilities():
    X, y, _ = _binary_fixture(
        seed=303
    )

    fitted_045 = (
        fit_redevelopment_detector_candidate(
            "ERD1_hgb_threshold_045",
            X,
            y,
        )
    )

    fitted_040 = (
        fit_redevelopment_detector_candidate(
            "ERD2_hgb_threshold_040",
            X,
            y,
        )
    )

    p045 = (
        fitted_045.predict_active_probability(
            X
        )
    )

    p040 = (
        fitted_040.predict_active_probability(
            X
        )
    )

    np.testing.assert_allclose(
        p045,
        p040,
        rtol=0.0,
        atol=0.0,
    )

    pred045 = fitted_045.predict(
        X
    )

    pred040 = fitted_040.predict(
        X
    )

    assert np.all(
        (~pred045)
        | pred040
    )


def test_extra_trees_threshold_candidates_have_identical_probabilities():
    X, y, _ = _binary_fixture(
        seed=404
    )

    fitted_045 = (
        fit_redevelopment_detector_candidate(
            "ERD3_extra_trees_threshold_045",
            X,
            y,
        )
    )

    fitted_040 = (
        fit_redevelopment_detector_candidate(
            "ERD4_extra_trees_threshold_040",
            X,
            y,
        )
    )

    p045 = (
        fitted_045.predict_active_probability(
            X
        )
    )

    p040 = (
        fitted_040.predict_active_probability(
            X
        )
    )

    np.testing.assert_allclose(
        p045,
        p040,
        rtol=0.0,
        atol=0.0,
    )

    assert np.all(
        (~fitted_045.predict(X))
        | fitted_040.predict(X)
    )


def test_detector_evaluation_reports_required_fields():
    X, y, family = _binary_fixture(
        seed=505
    )

    fitted = (
        fit_redevelopment_detector_candidate(
            "ERD1_hgb_threshold_045",
            X,
            y,
        )
    )

    evaluation = (
        evaluate_redevelopment_detector_candidate(
            fitted,
            X,
            y,
            family,
        )
    )

    assert evaluation[
        "candidate_id"
    ] == "ERD1_hgb_threshold_045"

    assert evaluation[
        "threshold"
    ] == 0.45

    metrics = evaluation[
        "metrics"
    ]

    for name in (
        "balanced_accuracy",
        "macro_f1",
        "positive_precision",
        "active_recall",
        "specificity",
        "false_positive_rate",
        "auroc",
        "average_precision",
    ):
        assert 0.0 <= metrics[
            name
        ] <= 1.0

    assert set(
        metrics[
            "per_active_fault_family_recall"
        ]
    ) == set(
        ORIGINAL_ACTIVE_CLASSES
    )


@pytest.mark.parametrize(
    "candidate_id",
    REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS,
)
def test_redevelopment_diagnoser_fits_three_classes(
    candidate_id,
):
    X, binary_y, multiclass_y = (
        _diagnosis_fixture()
    )

    fitted = (
        fit_redevelopment_diagnoser_candidate(
            candidate_id,
            X,
            binary_y,
            multiclass_y,
        )
    )

    assert is_dataclass(
        fitted
    )

    assert isinstance(
        fitted,
        FittedRedevelopmentDiagnoserCandidate,
    )

    assert fitted.candidate_id == candidate_id

    assert fitted.fit_rows == X.shape[0]

    assert (
        fitted.train_majority_class
        in REDEVELOPMENT_CLASSES
    )

    assert set(
        np.asarray(
            fitted.model.classes_,
            dtype=str,
        ).tolist()
    ) == set(
        REDEVELOPMENT_CLASSES
    )

    predictions = fitted.predict(
        X[:20]
    )

    assert set(
        predictions.tolist()
    ).issubset(
        set(
            REDEVELOPMENT_CLASSES
        )
    )


def test_diagnoser_evaluation_maps_ground_truth_before_scoring():
    X, binary_y, multiclass_y = (
        _diagnosis_fixture(
            seed=606
        )
    )

    fitted = (
        fit_redevelopment_diagnoser_candidate(
            "ERG2_hist_gradient_boosting",
            X,
            binary_y,
            multiclass_y,
        )
    )

    evaluation = (
        evaluate_redevelopment_diagnoser_candidate(
            fitted,
            X,
            binary_y,
            multiclass_y,
        )
    )

    assert evaluation[
        "validation_active_rows"
    ] == X.shape[0]

    assert set(
        evaluation[
            "metrics"
        ][
            "per_class"
        ]
    ) == set(
        REDEVELOPMENT_CLASSES
    )

    assert (
        evaluation[
            "majority_baseline"
        ][
            "majority_class"
        ]
        in REDEVELOPMENT_CLASSES
    )


def test_detector_acceptance_pass_and_fail():
    passing = _passing_detector_evaluation(
        "ERD1_hgb_threshold_045"
    )

    result = (
        redevelopment_detector_acceptance(
            passing
        )
    )

    assert result[
        "passed"
    ] is True

    failing = _passing_detector_evaluation(
        "ERD1_hgb_threshold_045"
    )

    failing[
        "metrics"
    ][
        "active_recall"
    ] = 0.749

    result = (
        redevelopment_detector_acceptance(
            failing
        )
    )

    assert result[
        "passed"
    ] is False

    assert result[
        "checks"
    ][
        "fault_active_recall"
    ] is False


def test_diagnoser_acceptance_pass_and_fail():
    passing = _passing_diagnoser_evaluation(
        "ERG1_logistic_regression"
    )

    result = (
        redevelopment_diagnoser_acceptance(
            passing
        )
    )

    assert result[
        "passed"
    ] is True

    failing = _passing_diagnoser_evaluation(
        "ERG1_logistic_regression"
    )

    failing[
        "metrics"
    ][
        "minimum_per_class_recall"
    ] = 0.49

    result = (
        redevelopment_diagnoser_acceptance(
            failing
        )
    )

    assert result[
        "passed"
    ] is False

    assert result[
        "checks"
    ][
        "per_class_recall"
    ] is False


def test_detector_selection_uses_frozen_order():
    evaluations = {
        "ERD1_hgb_threshold_045":
            _passing_detector_evaluation(
                "ERD1_hgb_threshold_045",
                balanced_accuracy=0.90,
                macro_f1=0.88,
                false_positive_rate=0.05,
                threshold=0.45,
            ),

        "ERD2_hgb_threshold_040":
            _passing_detector_evaluation(
                "ERD2_hgb_threshold_040",
                balanced_accuracy=0.90,
                macro_f1=0.89,
                false_positive_rate=0.06,
                threshold=0.40,
            ),

        "ERD3_extra_trees_threshold_045":
            _passing_detector_evaluation(
                "ERD3_extra_trees_threshold_045",
                balanced_accuracy=0.89,
                macro_f1=0.90,
                false_positive_rate=0.03,
                threshold=0.45,
            ),
    }

    assert (
        select_redevelopment_detector(
            evaluations
        )
        == "ERD2_hgb_threshold_040"
    )


def test_detector_selection_returns_none_when_nothing_passes():
    evaluation = _passing_detector_evaluation(
        "ERD1_hgb_threshold_045"
    )

    evaluation[
        "metrics"
    ][
        "active_recall"
    ] = 0.20

    assert (
        select_redevelopment_detector(
            {
                "ERD1_hgb_threshold_045":
                    evaluation
            }
        )
        is None
    )


def test_diagnoser_selection_uses_frozen_order():
    evaluations = {
        "ERG1_logistic_regression":
            _passing_diagnoser_evaluation(
                "ERG1_logistic_regression",
                macro_f1=0.80,
                balanced_accuracy=0.79,
                minimum_recall=0.70,
            ),

        "ERG2_hist_gradient_boosting":
            _passing_diagnoser_evaluation(
                "ERG2_hist_gradient_boosting",
                macro_f1=0.82,
                balanced_accuracy=0.78,
                minimum_recall=0.68,
            ),

        "ERG3_extra_trees":
            _passing_diagnoser_evaluation(
                "ERG3_extra_trees",
                macro_f1=0.81,
                balanced_accuracy=0.84,
                minimum_recall=0.75,
            ),
    }

    assert (
        select_redevelopment_diagnoser(
            evaluations
        )
        == "ERG2_hist_gradient_boosting"
    )


def test_diagnoser_selection_returns_none_when_nothing_passes():
    evaluation = _passing_diagnoser_evaluation(
        "ERG1_logistic_regression"
    )

    evaluation[
        "metrics"
    ][
        "minimum_per_class_recall"
    ] = 0.0

    assert (
        select_redevelopment_diagnoser(
            {
                "ERG1_logistic_regression":
                    evaluation
            }
        )
        is None
    )


def test_sequence_diagnostics_covers_delay_miss_and_false_alarm():
    binary_truth = np.asarray(
        [
            False,
            False,
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    predictions = np.asarray(
        [
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        dtype=bool,
    )

    episode_ids = np.asarray(
        [
            "fault_a",
            "fault_a",
            "fault_a",
            "fault_a",
            "fault_b",
            "fault_b",
            "fault_b",
            "fault_b",
            "normal",
            "normal",
        ],
        dtype=str,
    )

    step_indices = np.asarray(
        [
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
        ],
        dtype=int,
    )

    result = detector_sequence_diagnostics(
        binary_truth=binary_truth,
        predictions=predictions,
        episode_ids=episode_ids,
        step_indices=step_indices,
    )

    summary = result[
        "summary"
    ]

    assert summary[
        "fault_episode_count"
    ] == 2

    assert summary[
        "detected_fault_episode_count"
    ] == 1

    assert summary[
        "missed_fault_episode_count"
    ] == 1

    assert summary[
        "pre_active_false_alarm_episode_count"
    ] == 1

    assert summary[
        "normal_episode_false_alarm_count"
    ] == 1

    assert summary[
        "mean_detection_delay_steps"
    ] == 1.0


def test_sequence_diagnostics_sorts_steps_within_episode():
    result = detector_sequence_diagnostics(
        binary_truth=np.asarray(
            [
                True,
                False,
                True,
            ]
        ),
        predictions=np.asarray(
            [
                True,
                False,
                False,
            ]
        ),
        episode_ids=np.asarray(
            [
                "a",
                "a",
                "a",
            ]
        ),
        step_indices=np.asarray(
            [
                2,
                0,
                1,
            ]
        ),
    )

    record = result[
        "episode_records"
    ][0]

    assert record[
        "first_active_step"
    ] == 1

    assert record[
        "first_alarm_at_or_after_active_step"
    ] == 2

    assert record[
        "detection_delay_steps"
    ] == 1


def test_sequence_diagnostics_rejects_duplicate_steps():
    with pytest.raises(
        ValueError
    ):
        detector_sequence_diagnostics(
            binary_truth=np.asarray(
                [
                    False,
                    True,
                ]
            ),
            predictions=np.asarray(
                [
                    False,
                    True,
                ]
            ),
            episode_ids=np.asarray(
                [
                    "a",
                    "a",
                ]
            ),
            step_indices=np.asarray(
                [
                    0,
                    0,
                ]
            ),
        )


def test_no_result_artifacts_are_written_by_module_import(
    tmp_path,
):
    # Import-time behaviour is already exercised above.
    # The implementation exposes no result writer and requires no output path.
    import plasma_ai.monitoring.redevelopment as redevelopment

    assert not hasattr(
        redevelopment,
        "write_result"
    )

    assert not hasattr(
        redevelopment,
        "write_benchmark"
    )


def test_protocol_path_exists():
    assert (
        DEFAULT_PHASE5ER_PROTOCOL_PATH.exists()
    )


def test_candidate_id_constants_exact():
    assert (
        REDEVELOPMENT_DETECTOR_CANDIDATE_IDS
        == (
            "ERD1_hgb_threshold_045",
            "ERD2_hgb_threshold_040",
            "ERD3_extra_trees_threshold_045",
            "ERD4_extra_trees_threshold_040",
        )
    )

    assert (
        REDEVELOPMENT_DIAGNOSER_CANDIDATE_IDS
        == (
            "ERG1_logistic_regression",
            "ERG2_hist_gradient_boosting",
            "ERG3_extra_trees",
        )
    )
