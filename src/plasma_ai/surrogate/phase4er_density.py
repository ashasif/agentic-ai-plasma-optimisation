"""Phase 4E-R monotonic electron-density surrogate candidates.

This module implements the separately predeclared Phase 4E-R density
redevelopment search.

It contains no dataset loading and therefore cannot access TEST targets.

The completed Phase 4D classical model-construction path is intentionally
left unchanged.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from plasma_ai.surrogate.classical import (
    HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE,
    HIST_GRADIENT_BOOSTING_MODEL,
    PHASE4_RANDOM_STATE,
    ClassicalCandidateSpec,
    ClassicalValidationResult,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    regression_metrics,
)
from plasma_ai.surrogate.transforms import TargetTransform


PHASE4ER_DENSITY_TRANSFORM = "log10"

PHASE4ER_MONOTONIC_CST = (
    1,
    1,
)

PHASE4ER_EARLY_STOPPING = False

PHASE4ER_EXPECTED_CANDIDATE_COUNT = 16


def enumerate_phase4er_density_specs(
) -> tuple[ClassicalCandidateSpec, ...]:
    """Return the exact frozen 16-candidate Phase 4E-R density grid."""

    specs: list[ClassicalCandidateSpec] = []

    for (
        learning_rate,
        max_iter,
        max_leaf_nodes,
        l2_regularization,
    ) in product(
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
            "learning_rate"
        ],
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
            "max_iter"
        ],
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
            "max_leaf_nodes"
        ],
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
            "l2_regularization"
        ],
    ):
        specs.append(
            ClassicalCandidateSpec(
                candidate_id=(
                    "phase4er_hist_gradient_boosting"
                    f"_lr{learning_rate:g}"
                    f"_iter{max_iter}"
                    f"_leaves{max_leaf_nodes}"
                    f"_l2{l2_regularization:g}"
                ),
                model_name=HIST_GRADIENT_BOOSTING_MODEL,
                parameters={
                    "learning_rate": learning_rate,
                    "max_iter": max_iter,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                },
            )
        )

    if len(specs) != PHASE4ER_EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            "Frozen Phase 4E-R density grid must contain "
            f"exactly {PHASE4ER_EXPECTED_CANDIDATE_COUNT} "
            f"candidates; observed {len(specs)}."
        )

    candidate_ids = {
        spec.candidate_id
        for spec in specs
    }

    if len(candidate_ids) != len(specs):
        raise RuntimeError(
            "Frozen Phase 4E-R density candidate IDs "
            "must be unique."
        )

    return tuple(specs)


def build_phase4er_density_candidate(
    spec: ClassicalCandidateSpec,
) -> HistGradientBoostingRegressor:
    """Build one unfitted frozen Phase 4E-R density candidate."""

    if spec.model_name != HIST_GRADIENT_BOOSTING_MODEL:
        raise ValueError(
            "Phase 4E-R density candidates must use "
            "HistGradientBoostingRegressor."
        )

    expected_parameters = {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "l2_regularization",
    }

    observed_parameters = set(
        spec.parameters
    )

    if observed_parameters != expected_parameters:
        raise ValueError(
            "Phase 4E-R density candidate parameters must "
            f"be exactly {sorted(expected_parameters)}; "
            f"observed {sorted(observed_parameters)}."
        )

    parameters = dict(
        spec.parameters
    )

    for name, value in parameters.items():
        allowed = (
            HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
                name
            ]
        )

        if value not in allowed:
            raise ValueError(
                f"Phase 4E-R parameter {name!r} must "
                f"be one of {allowed}; observed {value!r}."
            )

    return HistGradientBoostingRegressor(
        **parameters,
        monotonic_cst=list(
            PHASE4ER_MONOTONIC_CST
        ),
        early_stopping=PHASE4ER_EARLY_STOPPING,
        random_state=PHASE4_RANDOM_STATE,
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

    if array.shape[1] != 2:
        raise ValueError(
            f"{name} must contain exactly the two "
            "frozen Phase 4 input features."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _validated_density_target(
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

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    if np.any(
        array <= 0.0
    ):
        raise ValueError(
            f"{name} must contain strictly positive "
            "electron-density values."
        )

    return array


def fit_phase4er_density_candidate(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    spec: ClassicalCandidateSpec,
) -> ClassicalValidationResult:
    """Fit on TRAIN and evaluate on VALIDATION in physical units."""

    X_train = _validated_matrix(
        train_X,
        name="train_X",
    )

    X_validation = _validated_matrix(
        validation_X,
        name="validation_X",
    )

    y_train = _validated_density_target(
        train_y,
        name="train_y",
    )

    y_validation = _validated_density_target(
        validation_y,
        name="validation_y",
    )

    if (
        X_train.shape[0]
        != y_train.shape[0]
    ):
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

    transform = TargetTransform(
        PHASE4ER_DENSITY_TRANSFORM
    )

    transformed_train_y = transform.forward(
        y_train
    )

    model = build_phase4er_density_candidate(
        spec
    )

    model.fit(
        X_train,
        transformed_train_y,
    )

    transformed_prediction = np.asarray(
        model.predict(
            X_validation
        ),
        dtype=np.float64,
    )

    if transformed_prediction.ndim != 1:
        raise ValueError(
            "Phase 4E-R predictions must "
            "be one-dimensional."
        )

    if (
        transformed_prediction.shape
        != y_validation.shape
    ):
        raise ValueError(
            "Phase 4E-R prediction shape does not "
            "match VALIDATION target shape."
        )

    if not np.isfinite(
        transformed_prediction
    ).all():
        raise ValueError(
            "Phase 4E-R candidate produced "
            "non-finite transformed predictions."
        )

    physical_prediction = transform.inverse(
        transformed_prediction
    )

    metrics = regression_metrics(
        DENSITY_TARGET,
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
            np.min(
                physical_prediction
            )
        ),
        "maximum": float(
            np.max(
                physical_prediction
            )
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
        target_name=DENSITY_TARGET,
        transform_name=PHASE4ER_DENSITY_TRANSFORM,
        candidate_id=spec.candidate_id,
        model_name=spec.model_name,
        parameters=dict(
            spec.parameters
        ),
        metrics=metrics,
        prediction_sanity=prediction_sanity,
    )
