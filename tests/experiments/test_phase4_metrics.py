import numpy as np
import pytest

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
    regression_metrics,
)


def test_density_metrics_match_known_values():
    truth = np.array(
        [100.0, 200.0, 400.0, 800.0],
        dtype=np.float64,
    )

    prediction = np.array(
        [110.0, 180.0, 440.0, 720.0],
        dtype=np.float64,
    )

    metrics = regression_metrics(
        DENSITY_TARGET,
        truth,
        prediction,
    )

    relative_errors = np.array(
        [0.10, 0.10, 0.10, 0.10]
    )

    assert set(metrics) == {
        "mae",
        "rmse",
        "r2",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "p95_absolute_relative_error",
        "max_absolute_relative_error",
    }

    assert metrics["mae"] == pytest.approx(
        37.5
    )

    assert metrics[
        "mean_absolute_relative_error"
    ] == pytest.approx(
        np.mean(relative_errors)
    )

    assert metrics[
        "median_absolute_relative_error"
    ] == pytest.approx(
        0.10
    )

    assert metrics[
        "p95_absolute_relative_error"
    ] == pytest.approx(
        0.10
    )

    assert metrics[
        "max_absolute_relative_error"
    ] == pytest.approx(
        0.10
    )

    assert np.isfinite(
        metrics["rmse"]
    )

    assert np.isfinite(
        metrics["r2"]
    )


def test_temperature_metrics_use_absolute_eV_errors():
    truth = np.array(
        [2.0, 3.0, 4.0, 5.0],
        dtype=np.float64,
    )

    prediction = np.array(
        [2.1, 2.8, 4.3, 4.6],
        dtype=np.float64,
    )

    metrics = regression_metrics(
        TEMPERATURE_TARGET,
        truth,
        prediction,
    )

    absolute_errors = np.array(
        [0.1, 0.2, 0.3, 0.4]
    )

    assert set(metrics) == {
        "mae_eV",
        "rmse_eV",
        "r2",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "p95_absolute_error_eV",
        "max_absolute_error_eV",
    }

    assert metrics["mae_eV"] == pytest.approx(
        0.25
    )

    assert metrics[
        "max_absolute_error_eV"
    ] == pytest.approx(
        0.4
    )

    assert metrics[
        "p95_absolute_error_eV"
    ] == pytest.approx(
        np.percentile(
            absolute_errors,
            95.0,
        )
    )

    assert np.isfinite(
        metrics["rmse_eV"]
    )

    assert np.isfinite(
        metrics["r2"]
    )


def test_perfect_predictions_have_zero_error_and_unit_r2():
    truth = np.array(
        [1.0, 2.0, 4.0, 8.0],
        dtype=np.float64,
    )

    metrics = regression_metrics(
        DENSITY_TARGET,
        truth,
        truth.copy(),
    )

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == pytest.approx(
        1.0
    )
    assert metrics[
        "max_absolute_relative_error"
    ] == 0.0


def test_relative_metrics_reject_nonpositive_truth():
    truth = np.array(
        [1.0, 0.0, 2.0],
        dtype=np.float64,
    )

    prediction = np.array(
        [1.0, 1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        regression_metrics(
            DENSITY_TARGET,
            truth,
            prediction,
        )


def test_metric_inputs_must_have_matching_shapes():
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        regression_metrics(
            DENSITY_TARGET,
            np.array(
                [1.0, 2.0]
            ),
            np.array(
                [1.0]
            ),
        )


def test_metric_inputs_must_be_one_dimensional():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        regression_metrics(
            DENSITY_TARGET,
            np.array(
                [[1.0], [2.0]]
            ),
            np.array(
                [[1.0], [2.0]]
            ),
        )


def test_metric_inputs_must_be_finite():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        regression_metrics(
            DENSITY_TARGET,
            np.array(
                [1.0, np.nan]
            ),
            np.array(
                [1.0, 2.0]
            ),
        )


def test_unknown_target_is_rejected():
    with pytest.raises(
        KeyError,
        match="Unknown Phase 4 target",
    ):
        regression_metrics(
            "unknown_target",
            np.array(
                [1.0, 2.0]
            ),
            np.array(
                [1.1, 1.9]
            ),
        )
