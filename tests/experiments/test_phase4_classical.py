"""Tests for Phase 4C classical surrogate candidate construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from plasma_ai.surrogate.classical import (
    ClassicalCandidateSpec,
    EXTRA_TREES_BENCHMARK_PARAMETERS,
    EXTRA_TREES_FROZEN_SEARCH_SPACE,
    EXTRA_TREES_MODEL,
    HIST_GRADIENT_BOOSTING_BENCHMARK_PARAMETERS,
    HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE,
    HIST_GRADIENT_BOOSTING_MODEL,
    PHASE4_RANDOM_STATE,
    PHASE4C_BENCHMARK_SPECS,
    POLYNOMIAL_MODEL,
    build_classical_candidate,
    fit_classical_candidate,
)


PROTOCOL_PATH = Path(
    "configs/phase4/surrogate_protocol.json"
)


def test_classical_search_spaces_match_frozen_protocol():
    protocol = json.loads(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    candidates = protocol["candidate_models"]

    extra_trees = candidates["extra_trees"]
    hist_gradient_boosting = candidates[
        "hist_gradient_boosting"
    ]

    expected_extra_trees = {
        name: tuple(values)
        for name, values in extra_trees[
            "search_space"
        ].items()
    }

    expected_hist_gradient_boosting = {
        name: tuple(values)
        for name, values in hist_gradient_boosting[
            "search_space"
        ].items()
    }

    assert (
        EXTRA_TREES_FROZEN_SEARCH_SPACE
        == expected_extra_trees
    )

    assert (
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE
        == expected_hist_gradient_boosting
    )

    assert (
        PHASE4_RANDOM_STATE
        == extra_trees["random_state"]
        == hist_gradient_boosting["random_state"]
    )


def test_phase4c_benchmark_specs_are_explicit():
    assert [
        spec.candidate_id
        for spec in PHASE4C_BENCHMARK_SPECS
    ] == [
        "polynomial_degree_2",
        "polynomial_degree_3",
        "extra_trees_reference",
        "hist_gradient_boosting_reference",
    ]

    assert [
        spec.model_name
        for spec in PHASE4C_BENCHMARK_SPECS
    ] == [
        POLYNOMIAL_MODEL,
        POLYNOMIAL_MODEL,
        EXTRA_TREES_MODEL,
        HIST_GRADIENT_BOOSTING_MODEL,
    ]


@pytest.mark.parametrize(
    "degree",
    [2, 3],
)
def test_polynomial_candidate_builder(degree):
    spec = ClassicalCandidateSpec(
        candidate_id=f"polynomial_degree_{degree}",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": degree,
        },
    )

    model = build_classical_candidate(spec)

    assert isinstance(model, Pipeline)

    polynomial = model.named_steps[
        "polynomial_features"
    ]

    regression = model.named_steps[
        "linear_regression"
    ]

    assert isinstance(
        polynomial,
        PolynomialFeatures,
    )
    assert polynomial.degree == degree
    assert polynomial.include_bias is False
    assert polynomial.interaction_only is False

    assert isinstance(
        regression,
        LinearRegression,
    )


def test_extra_trees_reference_builder():
    spec = next(
        spec
        for spec in PHASE4C_BENCHMARK_SPECS
        if spec.model_name == EXTRA_TREES_MODEL
    )

    model = build_classical_candidate(spec)

    assert isinstance(
        model,
        ExtraTreesRegressor,
    )

    assert model.n_estimators == 200
    assert model.max_depth is None
    assert model.min_samples_leaf == 1
    assert model.max_features == 1.0
    assert model.random_state == PHASE4_RANDOM_STATE
    assert model.n_jobs == 1

    assert dict(spec.parameters) == (
        EXTRA_TREES_BENCHMARK_PARAMETERS
    )


def test_hist_gradient_boosting_reference_builder():
    spec = next(
        spec
        for spec in PHASE4C_BENCHMARK_SPECS
        if (
            spec.model_name
            == HIST_GRADIENT_BOOSTING_MODEL
        )
    )

    model = build_classical_candidate(spec)

    assert isinstance(
        model,
        HistGradientBoostingRegressor,
    )

    assert model.learning_rate == 0.10
    assert model.max_iter == 200
    assert model.max_leaf_nodes == 31
    assert model.l2_regularization == 0.0
    assert model.random_state == PHASE4_RANDOM_STATE

    assert dict(spec.parameters) == (
        HIST_GRADIENT_BOOSTING_BENCHMARK_PARAMETERS
    )


def test_rejects_non_frozen_polynomial_degree():
    spec = ClassicalCandidateSpec(
        candidate_id="invalid_polynomial",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 4,
        },
    )

    with pytest.raises(
        ValueError,
        match="Polynomial degree",
    ):
        build_classical_candidate(spec)


def test_rejects_non_frozen_extra_trees_value():
    spec = ClassicalCandidateSpec(
        candidate_id="invalid_extra_trees",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="frozen Phase 4 values",
    ):
        build_classical_candidate(spec)


def test_rejects_non_frozen_hgb_value():
    spec = ClassicalCandidateSpec(
        candidate_id="invalid_hgb",
        model_name=HIST_GRADIENT_BOOSTING_MODEL,
        parameters={
            "learning_rate": 0.20,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="frozen Phase 4 values",
    ):
        build_classical_candidate(spec)


def test_rejects_unknown_classical_model():
    spec = ClassicalCandidateSpec(
        candidate_id="unknown",
        model_name="unsupported_model",
        parameters={},
    )

    with pytest.raises(
        KeyError,
        match="Unknown Phase 4 classical model",
    ):
        build_classical_candidate(spec)



def _toy_regression_data():
    train_X = np.array([
        [15.0, 10.0],
        [25.0, 20.0],
        [35.0, 15.0],
        [45.0, 30.0],
        [60.0, 25.0],
        [75.0, 45.0],
        [90.0, 60.0],
        [55.0, 50.0],
    ])

    validation_X = np.array([
        [20.0, 15.0],
        [40.0, 20.0],
        [65.0, 35.0],
        [80.0, 55.0],
    ])

    def density(X):
        power = X[:, 0]
        pressure = X[:, 1]
        return (
            8.0e16
            + 5.0e14 * power
            + 2.0e14 * pressure
            + 1.0e12 * power * pressure
        )

    return (
        train_X,
        density(train_X),
        validation_X,
        density(validation_X),
    )


def test_classical_fit_evaluates_physical_scale():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_regression_data()

    spec = ClassicalCandidateSpec(
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 2,
        },
    )

    result = fit_classical_candidate(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        target_name="true_electron_density_m3",
        transform_name="identity",
        spec=spec,
    )

    assert result.candidate_id == (
        "polynomial_degree_2"
    )
    assert result.model_name == POLYNOMIAL_MODEL
    assert result.transform_name == "identity"

    assert set(result.metrics) == {
        "mae",
        "rmse",
        "r2",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "p95_absolute_relative_error",
        "max_absolute_relative_error",
    }

    assert result.prediction_sanity[
        "all_finite"
    ] is True

    assert result.prediction_sanity[
        "negative_count"
    ] == 0

    assert result.prediction_sanity[
        "non_positive_count"
    ] == 0


def test_density_log10_predictions_return_to_physical_scale():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_regression_data()

    spec = ClassicalCandidateSpec(
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 2,
        },
    )

    result = fit_classical_candidate(
        train_X=train_X,
        train_y=train_y,
        validation_X=validation_X,
        validation_y=validation_y,
        target_name="true_electron_density_m3",
        transform_name="log10",
        spec=spec,
    )

    assert result.prediction_sanity[
        "minimum"
    ] > 0.0

    assert np.isfinite(
        list(result.metrics.values())
    ).all()


def test_classical_fit_rejects_disallowed_transform():
    (
        train_X,
        _,
        validation_X,
        _,
    ) = _toy_regression_data()

    train_y = np.array([
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
    ])

    validation_y = np.array([
        2.15,
        2.25,
        2.45,
        2.65,
    ])

    spec = ClassicalCandidateSpec(
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 2,
        },
    )

    with pytest.raises(
        ValueError,
        match="not permitted",
    ):
        fit_classical_candidate(
            train_X=train_X,
            train_y=train_y,
            validation_X=validation_X,
            validation_y=validation_y,
            target_name=(
                "true_electron_temperature_eV"
            ),
            transform_name="log10",
            spec=spec,
        )


def test_classical_fit_rejects_nonfinite_input():
    (
        train_X,
        train_y,
        validation_X,
        validation_y,
    ) = _toy_regression_data()

    train_X = train_X.copy()
    train_X[0, 0] = np.nan

    spec = ClassicalCandidateSpec(
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={
            "degree": 2,
        },
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        fit_classical_candidate(
            train_X=train_X,
            train_y=train_y,
            validation_X=validation_X,
            validation_y=validation_y,
            target_name="true_electron_density_m3",
            transform_name="identity",
            spec=spec,
        )
