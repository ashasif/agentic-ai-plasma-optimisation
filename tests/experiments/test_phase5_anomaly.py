"""Synthetic tests for frozen Phase 5C anomaly baselines.

These tests do not load the real Phase 3 monitoring dataset and do not access
the Phase 5 TEST split.
"""

import numpy as np
import pytest

from plasma_ai.monitoring.anomaly import (
    RESIDUAL_FEATURE_INDICES,
    RESIDUAL_FEATURES,
    fit_isolation_forest_baseline,
    fit_phase5_anomaly_baselines,
    fit_robust_residual_baseline,
    load_phase5_anomaly_contract,
)
from plasma_ai.monitoring.data import MODEL_FEATURES


def _synthetic_training_data():
    rng = np.random.default_rng(
        20260924
    )

    healthy_rows = 120
    fault_rows = 20

    healthy = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            healthy_rows,
            len(MODEL_FEATURES),
        ),
    )

    # Make the raw-feature positions physically positive enough
    # for a generic 9-column monitoring fixture.
    healthy[:, 0] += 50.0
    healthy[:, 1] += 30.0
    healthy[:, 2] += 20.0
    healthy[:, 3] += 50.0
    healthy[:, 4] += 20.0
    healthy[:, 5] += 30.0

    # Frozen residual features use much smaller scales.
    healthy[:, 6:] = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(
            healthy_rows,
            3,
        ),
    )

    faults = healthy[:fault_rows].copy()

    faults[:, 6] += 0.25
    faults[:, 7] -= 0.20
    faults[:, 8] += 0.30

    X = np.vstack(
        (
            healthy,
            faults,
        )
    )

    y = np.concatenate(
        (
            np.zeros(
                healthy_rows,
                dtype=np.bool_,
            ),
            np.ones(
                fault_rows,
                dtype=np.bool_,
            ),
        )
    )

    return (
        X,
        y,
        healthy_rows,
    )


def test_frozen_anomaly_contract_is_exact():
    contract = load_phase5_anomaly_contract()

    robust = contract[
        "robust_residual"
    ]

    isolation = contract[
        "isolation_forest"
    ]

    assert tuple(
        robust["residual_features"]
    ) == RESIDUAL_FEATURES

    assert (
        robust["threshold_quantile"]
        == 0.99
    )

    assert (
        isolation["n_estimators"]
        == 500
    )

    assert (
        isolation["contamination"]
        == 0.01
    )

    assert (
        isolation["random_state"]
        == 20260924
    )

    assert isolation["n_jobs"] == 1


def test_residual_indices_match_frozen_feature_order():
    assert RESIDUAL_FEATURE_INDICES == (
        6,
        7,
        8,
    )

    assert tuple(
        MODEL_FEATURES[index]
        for index in RESIDUAL_FEATURE_INDICES
    ) == RESIDUAL_FEATURES


def test_robust_baseline_fits_only_healthy_rows():
    X, y, healthy_rows = (
        _synthetic_training_data()
    )

    baseline = (
        fit_robust_residual_baseline(
            X,
            y,
        )
    )

    assert (
        baseline.healthy_fit_rows
        == healthy_rows
    )


def test_robust_baseline_uses_median_and_scaled_mad():
    X, y, _ = (
        _synthetic_training_data()
    )

    baseline = (
        fit_robust_residual_baseline(
            X,
            y,
        )
    )

    healthy = X[
        ~y
    ][
        :,
        RESIDUAL_FEATURE_INDICES,
    ]

    expected_median = np.median(
        healthy,
        axis=0,
    )

    expected_mad = np.median(
        np.abs(
            healthy
            - expected_median
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        baseline.medians,
        expected_median,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        baseline.scales,
        1.4826 * expected_mad,
        rtol=1e-15,
        atol=1e-15,
    )

    assert baseline.scale_sources == (
        "mad",
        "mad",
        "mad",
    )


def test_robust_baseline_threshold_is_train_healthy_quantile():
    X, y, _ = (
        _synthetic_training_data()
    )

    baseline = (
        fit_robust_residual_baseline(
            X,
            y,
        )
    )

    healthy_scores = baseline.score(
        X[
            ~y
        ]
    )

    expected = np.quantile(
        healthy_scores,
        0.99,
        method="linear",
    )

    assert baseline.threshold == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-15,
    )


def test_robust_row_score_is_max_absolute_z():
    X, y, _ = (
        _synthetic_training_data()
    )

    baseline = (
        fit_robust_residual_baseline(
            X,
            y,
        )
    )

    row = X[
        [0]
    ]

    residuals = row[
        :,
        RESIDUAL_FEATURE_INDICES,
    ]

    expected = np.max(
        np.abs(
            (
                residuals
                - baseline.medians
            )
            / baseline.scales
        ),
        axis=1,
    )

    np.testing.assert_allclose(
        baseline.score(row),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_mad_uses_standard_deviation_fallback():
    X, y, _ = (
        _synthetic_training_data()
    )

    healthy_indices = np.flatnonzero(
        ~y
    )

    # More than half the healthy rows are exactly zero, so MAD is
    # zero, while a minority retain nonzero values so std is positive.
    X[
        healthy_indices,
        6,
    ] = 0.0

    X[
        healthy_indices[-20:],
        6,
    ] = np.linspace(
        0.01,
        0.20,
        20,
    )

    baseline = (
        fit_robust_residual_baseline(
            X,
            y,
        )
    )

    assert (
        baseline.scale_sources[0]
        == "standard_deviation_fallback"
    )

    expected_std = np.std(
        X[
            ~y,
            6,
        ],
        ddof=0,
    )

    assert baseline.scales[0] == pytest.approx(
        expected_std
    )


def test_zero_mad_and_zero_std_is_rejected():
    X, y, _ = (
        _synthetic_training_data()
    )

    X[
        ~y,
        6,
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="zero MAD and zero standard deviation",
    ):
        fit_robust_residual_baseline(
            X,
            y,
        )


def test_isolation_forest_uses_exact_frozen_configuration():
    X, y, healthy_rows = (
        _synthetic_training_data()
    )

    baseline = (
        fit_isolation_forest_baseline(
            X,
            y,
        )
    )

    assert (
        baseline.healthy_fit_rows
        == healthy_rows
    )

    params = baseline.model.get_params()

    assert params["n_estimators"] == 500
    assert params["contamination"] == 0.01
    assert params["random_state"] == 20260924
    assert params["n_jobs"] == 1


def test_isolation_forest_scores_and_predictions_are_finite():
    X, y, _ = (
        _synthetic_training_data()
    )

    baseline = (
        fit_isolation_forest_baseline(
            X,
            y,
        )
    )

    scores = baseline.score(X)
    predictions = baseline.predict(X)

    assert scores.shape == (
        X.shape[0],
    )

    assert predictions.shape == (
        X.shape[0],
    )

    assert predictions.dtype == np.bool_

    assert np.all(
        np.isfinite(scores)
    )


def test_combined_baseline_fit_uses_only_healthy_rows():
    X, y, healthy_rows = (
        _synthetic_training_data()
    )

    baselines = (
        fit_phase5_anomaly_baselines(
            X,
            y,
        )
    )

    assert (
        baselines.robust_residual.healthy_fit_rows
        == healthy_rows
    )

    assert (
        baselines.isolation_forest.healthy_fit_rows
        == healthy_rows
    )


def test_anomaly_fit_rejects_no_healthy_training_rows():
    X, _, _ = (
        _synthetic_training_data()
    )

    all_fault = np.ones(
        X.shape[0],
        dtype=np.bool_,
    )

    with pytest.raises(
        ValueError,
        match="No healthy TRAIN rows",
    ):
        fit_robust_residual_baseline(
            X,
            all_fault,
        )

    with pytest.raises(
        ValueError,
        match="No healthy TRAIN rows",
    ):
        fit_isolation_forest_baseline(
            X,
            all_fault,
        )


def test_anomaly_scoring_rejects_wrong_feature_width():
    X, y, _ = (
        _synthetic_training_data()
    )

    baselines = (
        fit_phase5_anomaly_baselines(
            X,
            y,
        )
    )

    invalid = np.ones(
        (5, 8),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="wrong feature width",
    ):
        baselines.robust_residual.score(
            invalid
        )

    with pytest.raises(
        ValueError,
        match="wrong feature width",
    ):
        baselines.isolation_forest.score(
            invalid
        )


def test_anomaly_fit_rejects_nonfinite_input():
    X, y, _ = (
        _synthetic_training_data()
    )

    X[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        fit_robust_residual_baseline(
            X,
            y,
        )
