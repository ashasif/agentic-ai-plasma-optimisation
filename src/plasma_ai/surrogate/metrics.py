"""Frozen Phase 4 surrogate-regression metrics.

Metric definitions follow the Phase 4A modelling protocol.

All metrics operate on predictions in physical target units.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import r2_score


DENSITY_TARGET = (
    "true_electron_density_m3"
)

TEMPERATURE_TARGET = (
    "true_electron_temperature_eV"
)


def _validated_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(
        y_true,
        dtype=np.float64,
    )

    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if truth.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if prediction.ndim != 1:
        raise ValueError(
            "y_pred must be one-dimensional."
        )

    if truth.shape != prediction.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if truth.size == 0:
        raise ValueError(
            "Metric inputs must not be empty."
        )

    if not np.isfinite(truth).all():
        raise ValueError(
            "y_true must contain only finite values."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "y_pred must contain only finite values."
        )

    return truth, prediction


def _absolute_errors(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    return np.abs(
        prediction - truth
    )


def _absolute_relative_errors(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    if np.any(truth <= 0.0):
        raise ValueError(
            "Relative-error metrics require "
            "strictly positive true target values."
        )

    return (
        np.abs(prediction - truth)
        / np.abs(truth)
    )


def _mae(
    absolute_errors: np.ndarray,
) -> float:
    return float(
        np.mean(absolute_errors)
    )


def _rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction - truth
                )
            )
        )
    )


def regression_metrics(
    target_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the frozen metric set for one Phase 4 target."""
    truth, prediction = _validated_vectors(
        y_true,
        y_pred,
    )

    absolute_errors = _absolute_errors(
        truth,
        prediction,
    )

    relative_errors = _absolute_relative_errors(
        truth,
        prediction,
    )

    r2 = float(
        r2_score(
            truth,
            prediction,
        )
    )

    if not math.isfinite(r2):
        raise ValueError(
            "R-squared is not finite for these inputs."
        )

    if target_name == DENSITY_TARGET:
        return {
            "mae": _mae(
                absolute_errors
            ),
            "rmse": _rmse(
                truth,
                prediction,
            ),
            "r2": r2,
            "mean_absolute_relative_error": float(
                np.mean(relative_errors)
            ),
            "median_absolute_relative_error": float(
                np.median(relative_errors)
            ),
            "p95_absolute_relative_error": float(
                np.percentile(
                    relative_errors,
                    95.0,
                )
            ),
            "max_absolute_relative_error": float(
                np.max(relative_errors)
            ),
        }

    if target_name == TEMPERATURE_TARGET:
        return {
            "mae_eV": _mae(
                absolute_errors
            ),
            "rmse_eV": _rmse(
                truth,
                prediction,
            ),
            "r2": r2,
            "mean_absolute_relative_error": float(
                np.mean(relative_errors)
            ),
            "median_absolute_relative_error": float(
                np.median(relative_errors)
            ),
            "p95_absolute_error_eV": float(
                np.percentile(
                    absolute_errors,
                    95.0,
                )
            ),
            "max_absolute_error_eV": float(
                np.max(absolute_errors)
            ),
        }

    raise KeyError(
        f"Unknown Phase 4 target {target_name!r}."
    )
