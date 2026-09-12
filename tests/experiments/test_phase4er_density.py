"""Tests for the predeclared Phase 4E-R constrained density surrogate."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from plasma_ai.surrogate.classical import (
    EXTRA_TREES_MODEL,
    HIST_GRADIENT_BOOSTING_MODEL,
    PHASE4_RANDOM_STATE,
    ClassicalCandidateSpec,
    ClassicalValidationResult,
)
from plasma_ai.surrogate.metrics import DENSITY_TARGET
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    PHASE4ER_EARLY_STOPPING,
    PHASE4ER_EXPECTED_CANDIDATE_COUNT,
    PHASE4ER_MONOTONIC_CST,
    build_phase4er_density_candidate,
    enumerate_phase4er_density_specs,
    fit_phase4er_density_candidate,
)


def _toy_density_data():
    power = np.linspace(
        15.0,
        90.0,
        10,
    )
    pressure = np.linspace(
        10.0,
        60.0,
        8,
    )

    train_X = np.asarray(
        [
            [pwr, prs]
            for pwr in power
            for prs in pressure
        ],
        dtype=float,
    )

    train_y = (
        1.0e16
        + 1.6e15 * train_X[:, 0]
        + 4.0e14 * train_X[:, 1]
        + 2.0e12 * train_X[:, 0] * train_X[:, 1]
    )

    validation_X = np.asarray(
        [
            [18.0, 12.0],
            [25.0, 20.0],
            [35.0, 30.0],
            [45.0, 40.0],
            [60.0, 50.0],
            [75.0, 55.0],
            [85.0, 58.0],
            [89.0, 59.0],
        ],
        dtype=float,
    )

    validation_y = (
        1.0e16
        + 1.6e15 * validation_X[:, 0]
        + 4.0e14 * validation_X[:, 1]
        + 2.0e12
        * validation_X[:, 0]
        * validation_X[:, 1]
    )

    return (
        train_X,
        train_y,
        validation_X,
        validation_y,
    )


def test_phase4er_density_grid_is_exact_and_unique():
    specs = enumerate_phase4er_density_specs()

    assert PHASE4ER_EXPECTED_CANDIDATE_COUNT == 16
    assert len(specs) == PHASE4ER_EXPECTED_CANDIDATE_COUNT

    assert len(
        {
            spec.candidate_id
            for spec in specs
        }
    ) == 16

    assert all(
        spec.model_name
        == HIST_GRADIENT_BOOSTING_MODEL
        for spec in specs
    )

    assert specs[0].candidate_id == (
        "phase4er_hist_gradient_boosting"
        "_lr0.05_iter200_leaves15_l20"
    )

    assert specs[-1].candidate_id == (
        "phase4er_hist_gradient_boosting"
        "_lr0.1_iter400_leaves31_l20.1"
    )


def test_phase4er_density_contract_constants():
    assert PHASE4ER_DENSITY_TRANSFORM == "log10"
    assert PHASE4ER_MONOTONIC_CST == (1, 1)
    assert PHASE4ER_EARLY_STOPPING is False


def test_phase4er_density_builder_freezes_constraints():
    spec = enumerate_phase4er_density_specs()[0]

    model = build_phase4er_density_candidate(
        spec
    )

    assert isinstance(
        model,
        HistGradientBoostingRegressor,
    )

    assert model.monotonic_cst == [1, 1]
    assert model.early_stopping is False
    assert model.random_state == PHASE4_RANDOM_STATE


def test_phase4er_density_builder_rejects_wrong_family():
    spec = ClassicalCandidateSpec(
        candidate_id="invalid_family",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="HistGradientBoostingRegressor",
    ):
        build_phase4er_density_candidate(
            spec
        )


def test_phase4er_density_builder_rejects_nonfrozen_value():
    spec = ClassicalCandidateSpec(
        candidate_id="invalid_learning_rate",
        model_name=HIST_GRADIENT_BOOSTING_MODEL,
        parameters={
            "learning_rate": 0.2,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        build_phase4er_density_candidate(
            spec
        )


def test_phase4er_density_fit_returns_physical_validation_result():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_density_data()

    spec = enumerate_phase4er_density_specs()[0]

    result = fit_phase4er_density_candidate(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        spec=spec,
    )

    assert isinstance(
        result,
        ClassicalValidationResult,
    )

    assert result.target_name == DENSITY_TARGET
    assert result.transform_name == "log10"
    assert result.candidate_id == spec.candidate_id

    assert result.prediction_sanity[
        "all_finite"
    ] is True

    assert result.prediction_sanity[
        "non_positive_count"
    ] == 0

    assert result.metrics[
        "p95_absolute_relative_error"
    ] >= 0.0

    assert result.metrics[
        "median_absolute_relative_error"
    ] >= 0.0

    assert result.metrics["rmse"] >= 0.0


def test_phase4er_density_fit_rejects_wrong_feature_width():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_density_data()

    spec = enumerate_phase4er_density_specs()[0]

    with pytest.raises(
        ValueError,
        match="exactly the two frozen",
    ):
        fit_phase4er_density_candidate(
            train_X=np.column_stack(
                (
                    train_X,
                    np.ones(
                        train_X.shape[0]
                    ),
                )
            ),
            train_y=train_y,
            validation_X=np.column_stack(
                (
                    validation_X,
                    np.ones(
                        validation_X.shape[0]
                    ),
                )
            ),
            validation_y=validation_y,
            spec=spec,
        )


def test_phase4er_density_fit_rejects_nonpositive_target():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_density_data()

    invalid_train_y = train_y.copy()
    invalid_train_y[0] = 0.0

    spec = enumerate_phase4er_density_specs()[0]

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        fit_phase4er_density_candidate(
            train_X=train_X,
            train_y=invalid_train_y,
            validation_X=validation_X,
            validation_y=validation_y,
            spec=spec,
        )
