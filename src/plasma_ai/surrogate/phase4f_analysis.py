"""Pure Phase 4F locked-evaluation analysis helpers.

These functions implement the reporting rules frozen before TEST access:

- physical-domain normalisation;
- boundary/interior and central-domain masks;
- physical-range-third operating-space masks;
- target-specific regional metric summaries;
- deterministic worst-case observation ranking;
- prediction-integrity diagnostics.

This module does not load the Phase 4 dataset and cannot access TEST targets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
    regression_metrics,
)


POWER_MIN_W = 15.0
POWER_MAX_W = 90.0

PRESSURE_MIN_MTORR = 10.0
PRESSURE_MAX_MTORR = 60.0

BOUNDARY_LOWER_MAX = 0.10
BOUNDARY_UPPER_MIN = 0.90

CENTRAL_LOWER = 0.20
CENTRAL_UPPER = 0.80

LOW_UPPER = 1.0 / 3.0
HIGH_LOWER = 2.0 / 3.0

WORST_CASE_COUNT = 10


def _validated_features(
    X: ArrayLike,
) -> NDArray[np.float64]:
    matrix = np.asarray(
        X,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Phase 4F features must be two-dimensional."
        )

    if matrix.shape[1] != 2:
        raise ValueError(
            "Phase 4F requires exactly two frozen features."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Phase 4F features must not be empty."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Phase 4F features must contain only finite values."
        )

    return matrix


def normalised_coordinates(
    X: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return frozen physical-domain normalized power and pressure."""

    matrix = _validated_features(
        X
    )

    power_norm = (
        matrix[:, 0] - POWER_MIN_W
    ) / (
        POWER_MAX_W - POWER_MIN_W
    )

    pressure_norm = (
        matrix[:, 1] - PRESSURE_MIN_MTORR
    ) / (
        PRESSURE_MAX_MTORR - PRESSURE_MIN_MTORR
    )

    tolerance = 1.0e-12

    if (
        np.any(power_norm < -tolerance)
        or np.any(power_norm > 1.0 + tolerance)
        or np.any(pressure_norm < -tolerance)
        or np.any(pressure_norm > 1.0 + tolerance)
    ):
        raise ValueError(
            "Phase 4F features lie outside the frozen physical domain."
        )

    return (
        np.clip(
            power_norm,
            0.0,
            1.0,
        ),
        np.clip(
            pressure_norm,
            0.0,
            1.0,
        ),
    )


def build_error_space_masks(
    X: ArrayLike,
) -> dict[str, NDArray[np.bool_]]:
    """Build every frozen Phase 4F reporting-region mask."""

    (
        power_norm,
        pressure_norm,
    ) = normalised_coordinates(
        X
    )

    near_boundary = (
        (power_norm <= BOUNDARY_LOWER_MAX)
        | (power_norm >= BOUNDARY_UPPER_MIN)
        | (pressure_norm <= BOUNDARY_LOWER_MAX)
        | (pressure_norm >= BOUNDARY_UPPER_MIN)
    )

    interior = ~near_boundary

    central_domain = (
        (power_norm >= CENTRAL_LOWER)
        & (power_norm <= CENTRAL_UPPER)
        & (pressure_norm >= CENTRAL_LOWER)
        & (pressure_norm <= CENTRAL_UPPER)
    )

    low_power = (
        (power_norm >= 0.0)
        & (power_norm < LOW_UPPER)
    )

    middle_power = (
        (power_norm >= LOW_UPPER)
        & (power_norm < HIGH_LOWER)
    )

    high_power = (
        (power_norm >= HIGH_LOWER)
        & (power_norm <= 1.0)
    )

    low_pressure = (
        (pressure_norm >= 0.0)
        & (pressure_norm < LOW_UPPER)
    )

    middle_pressure = (
        (pressure_norm >= LOW_UPPER)
        & (pressure_norm < HIGH_LOWER)
    )

    high_pressure = (
        (pressure_norm >= HIGH_LOWER)
        & (pressure_norm <= 1.0)
    )

    return {
        "near_boundary": near_boundary,
        "interior": interior,
        "central_domain": central_domain,
        "power_low": low_power,
        "power_middle": middle_power,
        "power_high": high_power,
        "pressure_low": low_pressure,
        "pressure_middle": middle_pressure,
        "pressure_high": high_pressure,
        "corner_low_power_low_pressure": (
            low_power
            & low_pressure
        ),
        "corner_low_power_high_pressure": (
            low_power
            & high_pressure
        ),
        "corner_high_power_low_pressure": (
            high_power
            & low_pressure
        ),
        "corner_high_power_high_pressure": (
            high_power
            & high_pressure
        ),
    }


def regional_metric_summary(
    target_name: str,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mask: ArrayLike,
) -> dict:
    """Return row count and frozen metrics for one reporting region."""

    truth = np.asarray(
        y_true,
        dtype=np.float64,
    )

    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    region_mask = np.asarray(
        mask,
        dtype=bool,
    )

    if (
        truth.ndim != 1
        or prediction.ndim != 1
        or region_mask.ndim != 1
    ):
        raise ValueError(
            "Regional metric inputs must be one-dimensional."
        )

    if not (
        truth.shape
        == prediction.shape
        == region_mask.shape
    ):
        raise ValueError(
            "Regional metric inputs must have identical shapes."
        )

    count = int(
        np.count_nonzero(
            region_mask
        )
    )

    if count < 2:
        raise ValueError(
            "A Phase 4F reporting region must contain "
            "at least two observations."
        )

    return {
        "rows": count,
        "metrics": regression_metrics(
            target_name,
            truth[region_mask],
            prediction[region_mask],
        ),
    }


def error_space_summary(
    target_name: str,
    X: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, dict]:
    """Return every frozen Phase 4F regional error summary."""

    masks = build_error_space_masks(
        X
    )

    return {
        name: regional_metric_summary(
            target_name,
            y_true,
            y_pred,
            mask,
        )
        for name, mask in masks.items()
    }


def prediction_integrity(
    y_pred: ArrayLike,
) -> dict:
    """Return frozen prediction-integrity diagnostics."""

    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if prediction.ndim != 1:
        raise ValueError(
            "Prediction integrity requires a one-dimensional vector."
        )

    if prediction.size == 0:
        raise ValueError(
            "Prediction vector must not be empty."
        )

    finite_mask = np.isfinite(
        prediction
    )

    all_finite = bool(
        finite_mask.all()
    )

    if all_finite:
        minimum = float(
            np.min(
                prediction
            )
        )
        maximum = float(
            np.max(
                prediction
            )
        )
        all_positive = bool(
            np.all(
                prediction > 0.0
            )
        )
    else:
        finite_values = prediction[
            finite_mask
        ]

        minimum = (
            float(
                np.min(
                    finite_values
                )
            )
            if finite_values.size
            else None
        )

        maximum = (
            float(
                np.max(
                    finite_values
                )
            )
            if finite_values.size
            else None
        )

        all_positive = False

    return {
        "prediction_count": int(
            prediction.size
        ),
        "all_predictions_finite": all_finite,
        "prediction_minimum": minimum,
        "prediction_maximum": maximum,
        "all_predictions_strictly_positive": (
            all_positive
        ),
    }


def worst_case_observations(
    target_name: str,
    X: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    count: int = WORST_CASE_COUNT,
) -> list[dict]:
    """Return deterministically ranked worst TEST observations."""

    matrix = _validated_features(
        X
    )

    truth = np.asarray(
        y_true,
        dtype=np.float64,
    )

    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError(
            "Worst-case targets and predictions must be one-dimensional."
        )

    if not (
        truth.shape
        == prediction.shape
        == (matrix.shape[0],)
    ):
        raise ValueError(
            "Worst-case features, targets, and predictions "
            "must contain the same number of rows."
        )

    if not np.isfinite(truth).all():
        raise ValueError(
            "Worst-case true values must be finite."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "Worst-case predictions must be finite."
        )

    if np.any(
        truth <= 0.0
    ):
        raise ValueError(
            "Worst-case relative errors require positive truths."
        )

    if count <= 0:
        raise ValueError(
            "Worst-case count must be positive."
        )

    count = min(
        int(count),
        truth.size,
    )

    absolute_error = np.abs(
        prediction - truth
    )

    absolute_relative_error = (
        absolute_error
        / np.abs(
            truth
        )
    )

    if target_name == DENSITY_TARGET:
        ranking_error = (
            absolute_relative_error
        )
    elif target_name == TEMPERATURE_TARGET:
        ranking_error = (
            absolute_error
        )
    else:
        raise KeyError(
            f"Unknown Phase 4 target {target_name!r}."
        )

    # Verified frozen TEST file order is split_index 1..N.
    split_indices = np.arange(
        1,
        truth.size + 1,
        dtype=np.int64,
    )

    order = np.lexsort(
        (
            split_indices,
            -ranking_error,
        )
    )

    selected = order[
        :count
    ]

    records = []

    for index in selected:
        records.append({
            "test_split_index": int(
                split_indices[index]
            ),
            "nominal_absorbed_power_W": float(
                matrix[index, 0]
            ),
            "target_pressure_mTorr": float(
                matrix[index, 1]
            ),
            "true_value": float(
                truth[index]
            ),
            "predicted_value": float(
                prediction[index]
            ),
            "absolute_error": float(
                absolute_error[index]
            ),
            "absolute_relative_error": float(
                absolute_relative_error[index]
            ),
        })

    return records
