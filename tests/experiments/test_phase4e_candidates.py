"""Tests for Phase 4E selected-candidate reconstruction and fitting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plasma_ai.surrogate.classical import (
    EXTRA_TREES_MODEL,
)
from plasma_ai.surrogate.data import (
    SurrogateDataset,
    SurrogateSplit,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4e_candidates import (
    fit_selected_candidate_on_train,
    load_phase4d_selected_configurations,
)


def _toy_dataset(
    *,
    expose_test_targets=False,
):
    train_X = np.asarray(
        [
            [15.0, 10.0],
            [30.0, 20.0],
            [45.0, 30.0],
            [60.0, 40.0],
            [75.0, 50.0],
            [90.0, 60.0],
        ],
        dtype=float,
    )

    density = (
        1.0e16
        + 1.0e15 * train_X[:, 0]
        + 2.0e14 * train_X[:, 1]
    )

    temperature = (
        2.2
        - 0.005 * train_X[:, 1]
    )

    train_y = np.column_stack(
        (
            density,
            temperature,
        )
    )

    test_y = (
        train_y[:2].copy()
        if expose_test_targets
        else None
    )

    return SurrogateDataset(
        feature_names=(
            "nominal_absorbed_power_W",
            "target_pressure_mTorr",
        ),
        target_names=(
            DENSITY_TARGET,
            TEMPERATURE_TARGET,
        ),
        splits={
            "train": SurrogateSplit(
                X=train_X,
                y=train_y,
            ),
            "validation": SurrogateSplit(
                X=train_X[:2],
                y=train_y[:2],
            ),
            "test": SurrogateSplit(
                X=train_X[:2],
                y=test_y,
            ),
        },
    )


def test_reconstructs_exact_frozen_phase4d_winners():
    density, temperature = (
        load_phase4d_selected_configurations()
    )

    assert density.target_name == DENSITY_TARGET
    assert density.transform_name == "log10"

    assert density.candidate_spec.model_name == (
        EXTRA_TREES_MODEL
    )

    assert dict(
        density.candidate_spec.parameters
    ) == {
        "n_estimators": 500,
        "max_depth": 16,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }

    assert temperature.target_name == (
        TEMPERATURE_TARGET
    )
    assert temperature.transform_name == "identity"

    assert temperature.candidate_spec.model_name == (
        EXTRA_TREES_MODEL
    )

    assert dict(
        temperature.candidate_spec.parameters
    ) == {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }


def test_train_only_fit_uses_selected_density_transform():
    density, _ = (
        load_phase4d_selected_configurations()
    )

    fitted = fit_selected_candidate_on_train(
        _toy_dataset(),
        density,
    )

    assert fitted.fit_rows == 6
    assert fitted.transform.name == "log10"

    predictions = fitted.predict_physical(
        np.asarray(
            [
                [25.0, 15.0],
                [70.0, 45.0],
            ],
            dtype=float,
        )
    )

    assert predictions.shape == (2,)
    assert np.isfinite(
        predictions
    ).all()
    assert np.all(
        predictions > 0.0
    )


def test_train_only_fit_uses_selected_temperature_transform():
    _, temperature = (
        load_phase4d_selected_configurations()
    )

    fitted = fit_selected_candidate_on_train(
        _toy_dataset(),
        temperature,
    )

    assert fitted.fit_rows == 6
    assert fitted.transform.name == "identity"

    predictions = fitted.predict_physical(
        np.asarray(
            [
                [25.0, 15.0],
                [70.0, 45.0],
            ],
            dtype=float,
        )
    )

    assert predictions.shape == (2,)
    assert np.isfinite(
        predictions
    ).all()


def test_phase4e_fit_rejects_exposed_test_targets():
    density, _ = (
        load_phase4d_selected_configurations()
    )

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        fit_selected_candidate_on_train(
            _toy_dataset(
                expose_test_targets=True
            ),
            density,
        )


def test_selection_loader_rejects_non_phase4d_artifact(
    tmp_path,
):
    path = tmp_path / "selection.json"

    path.write_text(
        json.dumps(
            {
                "phase": "4E",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Phase 4D",
    ):
        load_phase4d_selected_configurations(
            path
        )


def test_prediction_rejects_wrong_feature_width():
    density, _ = (
        load_phase4d_selected_configurations()
    )

    fitted = fit_selected_candidate_on_train(
        _toy_dataset(),
        density,
    )

    with pytest.raises(
        ValueError,
        match="exactly two frozen features",
    ):
        fitted.predict_physical(
            np.ones(
                (3, 3),
                dtype=float,
            )
        )


def test_prediction_rejects_nonfinite_features():
    density, _ = (
        load_phase4d_selected_configurations()
    )

    fitted = fit_selected_candidate_on_train(
        _toy_dataset(),
        density,
    )

    X = np.asarray(
        [
            [15.0, 10.0],
            [np.nan, 20.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        fitted.predict_physical(
            X
        )
