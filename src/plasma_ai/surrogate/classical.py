"""Classical surrogate candidates for Phase 4C.

Phase 4C benchmarks the frozen classical model families beyond the
Phase 4B reference baselines:

- polynomial response-surface regression;
- ExtraTreesRegressor;
- HistGradientBoostingRegressor.

The nonlinear benchmark configurations in this module are predeclared
representative points from the frozen Phase 4A search spaces. They are
not validation-selected hyperparameters. Exhaustive controlled
validation comparison belongs to Phase 4D.

This module contains model construction only. It does not load data and
therefore cannot access locked TEST targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from plasma_ai.surrogate.metrics import regression_metrics
from plasma_ai.surrogate.transforms import (
    TargetTransform,
    allowed_transforms_for_target,
)


POLYNOMIAL_MODEL = "polynomial_regression"
EXTRA_TREES_MODEL = "extra_trees"
HIST_GRADIENT_BOOSTING_MODEL = "hist_gradient_boosting"

PHASE4_RANDOM_STATE = 20260913

POLYNOMIAL_DEGREES = (
    2,
    3,
)

EXTRA_TREES_FROZEN_SEARCH_SPACE = {
    "n_estimators": (200, 500),
    "max_depth": (None, 8, 16),
    "min_samples_leaf": (1, 2, 4),
    "max_features": (1.0,),
}

HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE = {
    "learning_rate": (0.05, 0.10),
    "max_iter": (200, 400),
    "max_leaf_nodes": (15, 31),
    "l2_regularization": (0.0, 0.1),
}

EXTRA_TREES_BENCHMARK_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
}

HIST_GRADIENT_BOOSTING_BENCHMARK_PARAMETERS = {
    "learning_rate": 0.10,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.0,
}


@dataclass(frozen=True)
class ClassicalCandidateSpec:
    """Fully identified classical surrogate candidate."""

    candidate_id: str
    model_name: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ClassicalValidationResult:
    """Physical-scale VALIDATION result for one candidate."""

    target_name: str
    transform_name: str
    candidate_id: str
    model_name: str
    parameters: Mapping[str, object]
    metrics: Mapping[str, float]
    prediction_sanity: Mapping[str, object]


PHASE4C_BENCHMARK_SPECS = (
    ClassicalCandidateSpec(
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 2,
        },
    ),
    ClassicalCandidateSpec(
        candidate_id="polynomial_degree_3",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 3,
        },
    ),
    ClassicalCandidateSpec(
        candidate_id="extra_trees_reference",
        model_name=EXTRA_TREES_MODEL,
        parameters=EXTRA_TREES_BENCHMARK_PARAMETERS,
    ),
    ClassicalCandidateSpec(
        candidate_id="hist_gradient_boosting_reference",
        model_name=HIST_GRADIENT_BOOSTING_MODEL,
        parameters=HIST_GRADIENT_BOOSTING_BENCHMARK_PARAMETERS,
    ),
)


def _require_exact_parameters(
    parameters: Mapping[str, object],
    *,
    expected: set[str],
    model_name: str,
) -> dict[str, object]:
    observed = set(parameters)

    if observed != expected:
        raise ValueError(
            f"{model_name!r} parameters must be exactly "
            f"{sorted(expected)}; observed {sorted(observed)}."
        )

    return dict(parameters)


def _require_frozen_parameter_values(
    parameters: Mapping[str, object],
    *,
    search_space: Mapping[str, tuple[object, ...]],
    model_name: str,
) -> None:
    for name, value in parameters.items():
        allowed = search_space[name]

        if value not in allowed:
            raise ValueError(
                f"{model_name!r} parameter {name!r} must be one "
                f"of the frozen Phase 4 values {allowed}; "
                f"observed {value!r}."
            )


def build_classical_candidate(
    spec: ClassicalCandidateSpec,
):
    """Build one unfitted Phase 4 classical surrogate candidate."""
    if spec.model_name == POLYNOMIAL_MODEL:
        parameters = _require_exact_parameters(
            spec.parameters,
            expected={"degree"},
            model_name=spec.model_name,
        )

        degree = parameters["degree"]

        if degree not in POLYNOMIAL_DEGREES:
            raise ValueError(
                "Polynomial degree must be one of the frozen "
                f"Phase 4 values {POLYNOMIAL_DEGREES}; "
                f"observed {degree!r}."
            )

        return Pipeline([
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=int(degree),
                    include_bias=False,
                    interaction_only=False,
                ),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ])

    if spec.model_name == EXTRA_TREES_MODEL:
        parameters = _require_exact_parameters(
            spec.parameters,
            expected={
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            },
            model_name=spec.model_name,
        )

        _require_frozen_parameter_values(
            parameters,
            search_space=EXTRA_TREES_FROZEN_SEARCH_SPACE,
            model_name=spec.model_name,
        )

        return ExtraTreesRegressor(
            **parameters,
            random_state=PHASE4_RANDOM_STATE,
            n_jobs=1,
        )

    if spec.model_name == HIST_GRADIENT_BOOSTING_MODEL:
        parameters = _require_exact_parameters(
            spec.parameters,
            expected={
                "learning_rate",
                "max_iter",
                "max_leaf_nodes",
                "l2_regularization",
            },
            model_name=spec.model_name,
        )

        _require_frozen_parameter_values(
            parameters,
            search_space=HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE,
            model_name=spec.model_name,
        )

        return HistGradientBoostingRegressor(
            **parameters,
            random_state=PHASE4_RANDOM_STATE,
        )

    raise KeyError(
        f"Unknown Phase 4 classical model "
        f"{spec.model_name!r}."
    )


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


def fit_classical_candidate(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    target_name: str,
    transform_name: str,
    spec: ClassicalCandidateSpec,
) -> ClassicalValidationResult:
    """Fit on TRAIN and evaluate on VALIDATION in physical units.

    No dataset loading occurs here. Target transformations are applied
    only to the fitting target. Predictions are inverse transformed
    before the frozen Phase 4 metrics and numerical sanity metadata are
    calculated.
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

    model = build_classical_candidate(
        spec
    )

    model.fit(
        X_train,
        transformed_train_y,
    )

    transformed_prediction = np.asarray(
        model.predict(X_validation),
        dtype=np.float64,
    )

    if transformed_prediction.ndim != 1:
        raise ValueError(
            "Classical surrogate predictions must "
            "be one-dimensional."
        )

    if (
        transformed_prediction.shape
        != y_validation.shape
    ):
        raise ValueError(
            "Classical surrogate prediction shape "
            "does not match VALIDATION target shape."
        )

    if not np.isfinite(
        transformed_prediction
    ).all():
        raise ValueError(
            "Classical surrogate produced "
            "non-finite transformed predictions."
        )

    physical_prediction = transform.inverse(
        transformed_prediction
    )

    metrics = regression_metrics(
        target_name,
        y_validation,
        physical_prediction,
    )

    prediction_sanity = {
        "all_finite": bool(
            np.isfinite(
                physical_prediction
            ).all()
        ),
        "minimum": float(
            np.min(physical_prediction)
        ),
        "maximum": float(
            np.max(physical_prediction)
        ),
        "negative_count": int(
            np.count_nonzero(
                physical_prediction < 0.0
            )
        ),
        "non_positive_count": int(
            np.count_nonzero(
                physical_prediction <= 0.0
            )
        ),
    }

    return ClassicalValidationResult(
        target_name=target_name,
        transform_name=transform_name,
        candidate_id=spec.candidate_id,
        model_name=spec.model_name,
        parameters=dict(spec.parameters),
        metrics=metrics,
        prediction_sanity=prediction_sanity,
    )
