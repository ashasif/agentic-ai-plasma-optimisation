"""Reference surrogate baselines for Phase 4B.

The Phase 4A protocol freezes two reference model families:

- training-mean DummyRegressor
- ordinary LinearRegression

This module contains target-specific fitting and validation logic only.
It does not load the Phase 3 dataset and therefore cannot access the
locked TEST targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from plasma_ai.surrogate.metrics import (
    regression_metrics,
)
from plasma_ai.surrogate.transforms import (
    TargetTransform,
    allowed_transforms_for_target,
)


REFERENCE_MODELS = (
    "dummy_mean",
    "linear_regression",
)


@dataclass(frozen=True)
class BaselineResult:
    """Validation result for one reference surrogate."""

    target_name: str
    transform_name: str
    model_name: str
    metrics: Mapping[str, float]


def _validated_matrix(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional matrix."
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"{name} must contain at least one row."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _validated_target(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _build_reference_model(
    model_name: str,
):
    if model_name == "dummy_mean":
        return DummyRegressor(
            strategy="mean",
        )

    if model_name == "linear_regression":
        return LinearRegression()

    raise KeyError(
        f"Unknown Phase 4 reference model "
        f"{model_name!r}."
    )


def fit_reference_baseline(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    target_name: str,
    transform_name: str,
    model_name: str,
) -> BaselineResult:
    """Fit one reference model and score VALIDATION only.

    Target transformation is fitted deterministically from the
    predeclared Phase 4A policy. Validation predictions are always
    inverse transformed before physical-scale metrics are calculated.
    """
    X_train = _validated_matrix(
        train_X,
        name="train_X",
    )

    X_validation = _validated_matrix(
        validation_X,
        name="validation_X",
    )

    y_train = _validated_target(
        train_y,
        name="train_y",
    )

    y_validation = _validated_target(
        validation_y,
        name="validation_y",
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "train_X and train_y row counts must match."
        )

    if (
        X_validation.shape[0]
        != y_validation.shape[0]
    ):
        raise ValueError(
            "validation_X and validation_y "
            "row counts must match."
        )

    if (
        X_train.shape[1]
        != X_validation.shape[1]
    ):
        raise ValueError(
            "TRAIN and VALIDATION must have "
            "the same feature count."
        )

    allowed = allowed_transforms_for_target(
        target_name
    )

    if transform_name not in allowed:
        raise ValueError(
            f"Transform {transform_name!r} is not "
            f"permitted for target {target_name!r}. "
            f"Allowed transforms: {allowed}"
        )

    transform = TargetTransform(
        transform_name
    )

    transformed_train_y = transform.forward(
        y_train
    )

    model = _build_reference_model(
        model_name
    )

    model.fit(
        X_train,
        transformed_train_y,
    )

    transformed_prediction = model.predict(
        X_validation
    )

    physical_prediction = transform.inverse(
        transformed_prediction
    )

    metrics = regression_metrics(
        target_name,
        y_validation,
        physical_prediction,
    )

    return BaselineResult(
        target_name=target_name,
        transform_name=transform_name,
        model_name=model_name,
        metrics=metrics,
    )
