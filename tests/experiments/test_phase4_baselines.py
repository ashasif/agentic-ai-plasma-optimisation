import numpy as np
import pytest

from plasma_ai.surrogate.baselines import (
    fit_reference_baseline,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)


def _design():
    train_X = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )

    validation_X = np.array(
        [
            [1.5, 1.5],
            [2.5, 1.5],
            [1.5, 2.5],
        ],
        dtype=np.float64,
    )

    return train_X, validation_X


def test_linear_identity_baseline_recovers_linear_temperature_mapping():
    train_X, validation_X = _design()

    train_y = (
        2.0
        + 0.5 * train_X[:, 0]
        - 0.2 * train_X[:, 1]
    )

    validation_y = (
        2.0
        + 0.5 * validation_X[:, 0]
        - 0.2 * validation_X[:, 1]
    )

    result = fit_reference_baseline(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        target_name=TEMPERATURE_TARGET,
        transform_name="identity",
        model_name="linear_regression",
    )

    assert result.model_name == (
        "linear_regression"
    )

    assert result.transform_name == "identity"

    assert result.metrics[
        "rmse_eV"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result.metrics[
        "r2"
    ] == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_linear_log10_baseline_recovers_log_linear_density_mapping():
    train_X, validation_X = _design()

    train_log_y = (
        15.0
        + 0.20 * train_X[:, 0]
        + 0.10 * train_X[:, 1]
    )

    validation_log_y = (
        15.0
        + 0.20 * validation_X[:, 0]
        + 0.10 * validation_X[:, 1]
    )

    train_y = np.power(
        10.0,
        train_log_y,
    )

    validation_y = np.power(
        10.0,
        validation_log_y,
    )

    result = fit_reference_baseline(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        target_name=DENSITY_TARGET,
        transform_name="log10",
        model_name="linear_regression",
    )

    assert result.metrics[
        "p95_absolute_relative_error"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result.metrics[
        "r2"
    ] == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_dummy_mean_baseline_returns_finite_metrics():
    train_X, validation_X = _design()

    train_y = np.array(
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        dtype=np.float64,
    )

    validation_y = np.array(
        [2.5, 4.5, 6.5],
        dtype=np.float64,
    )

    result = fit_reference_baseline(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        target_name=TEMPERATURE_TARGET,
        transform_name="identity",
        model_name="dummy_mean",
    )

    assert all(
        np.isfinite(value)
        for value in result.metrics.values()
    )


def test_temperature_rejects_log10_transform():
    train_X, validation_X = _design()

    train_y = np.full(
        train_X.shape[0],
        3.0,
    )

    validation_y = np.full(
        validation_X.shape[0],
        3.0,
    )

    with pytest.raises(
        ValueError,
        match="not permitted",
    ):
        fit_reference_baseline(
            train_X=train_X,
            train_y=train_y,
            validation_X=validation_X,
            validation_y=validation_y,
            target_name=TEMPERATURE_TARGET,
            transform_name="log10",
            model_name="linear_regression",
        )


def test_unknown_reference_model_is_rejected():
    train_X, validation_X = _design()

    train_y = np.arange(
        1,
        train_X.shape[0] + 1,
        dtype=np.float64,
    )

    validation_y = np.arange(
        1,
        validation_X.shape[0] + 1,
        dtype=np.float64,
    )

    with pytest.raises(
        KeyError,
        match="Unknown Phase 4 reference model",
    ):
        fit_reference_baseline(
            train_X=train_X,
            train_y=train_y,
            validation_X=validation_X,
            validation_y=validation_y,
            target_name=DENSITY_TARGET,
            transform_name="identity",
            model_name="unknown",
        )


def test_training_row_mismatch_is_rejected():
    train_X, validation_X = _design()

    with pytest.raises(
        ValueError,
        match="row counts must match",
    ):
        fit_reference_baseline(
            train_X=train_X,
            train_y=np.array(
                [1.0, 2.0]
            ),
            validation_X=validation_X,
            validation_y=np.array(
                [1.0, 2.0, 3.0]
            ),
            target_name=DENSITY_TARGET,
            transform_name="identity",
            model_name="linear_regression",
        )


def test_feature_count_mismatch_is_rejected():
    train_X, validation_X = _design()

    train_y = np.arange(
        1,
        train_X.shape[0] + 1,
        dtype=np.float64,
    )

    validation_y = np.arange(
        1,
        validation_X.shape[0] + 1,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="same feature count",
    ):
        fit_reference_baseline(
            train_X=train_X,
            train_y=train_y,
            validation_X=validation_X[:, :1],
            validation_y=validation_y,
            target_name=DENSITY_TARGET,
            transform_name="identity",
            model_name="linear_regression",
        )
