"""Tests for Phase 4E-R TRAIN/VALIDATION orchestration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)
from plasma_ai.surrogate.data import (
    SurrogateDataset,
    SurrogateSplit,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4er_density import (
    enumerate_phase4er_density_specs,
)
import plasma_ai.surrogate.phase4er_experiment as experiment


FEATURE_NAMES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
)

TARGET_NAMES = (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)


def _dataset(
    *,
    expose_test_targets=False,
    train_rows=4096,
    validation_rows=2048,
    test_rows=2048,
):
    train_X = np.ones(
        (train_rows, 2),
        dtype=float,
    )

    validation_X = np.ones(
        (validation_rows, 2),
        dtype=float,
    )

    test_X = np.ones(
        (test_rows, 2),
        dtype=float,
    )

    train_y = np.column_stack(
        (
            np.full(
                train_rows,
                1.0e17,
            ),
            np.full(
                train_rows,
                2.0,
            ),
        )
    )

    validation_y = np.column_stack(
        (
            np.full(
                validation_rows,
                1.0e17,
            ),
            np.full(
                validation_rows,
                2.0,
            ),
        )
    )

    test_y = (
        np.column_stack(
            (
                np.full(
                    test_rows,
                    1.0e17,
                ),
                np.full(
                    test_rows,
                    2.0,
                ),
            )
        )
        if expose_test_targets
        else None
    )

    return SurrogateDataset(
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        splits={
            "train": SurrogateSplit(
                X=train_X,
                y=train_y,
            ),
            "validation": SurrogateSplit(
                X=validation_X,
                y=validation_y,
            ),
            "test": SurrogateSplit(
                X=test_X,
                y=test_y,
            ),
        },
    )


def _fake_fit_factory():
    order = {
        spec.candidate_id: index
        for index, spec in enumerate(
            enumerate_phase4er_density_specs()
        )
    }

    def fake_fit(
        *,
        train_X,
        train_y,
        validation_X,
        validation_y,
        spec,
    ):
        index = order[
            spec.candidate_id
        ]

        return ClassicalValidationResult(
            target_name=DENSITY_TARGET,
            transform_name="log10",
            candidate_id=spec.candidate_id,
            model_name=spec.model_name,
            parameters=dict(
                spec.parameters
            ),
            metrics={
                "mae": 1.0e14 + index,
                "rmse": 2.0e14 + index,
                "r2": 0.99,
                "mean_absolute_relative_error": (
                    0.002 + index * 0.001
                ),
                "median_absolute_relative_error": (
                    0.001 + index * 0.001
                ),
                "p95_absolute_relative_error": (
                    0.003 + index * 0.001
                ),
                "maximum_absolute_relative_error": (
                    0.01 + index * 0.001
                ),
            },
            prediction_sanity={
                "all_finite": True,
                "minimum": 1.0e16,
                "maximum": 2.0e17,
                "negative_count": 0,
                "non_positive_count": 0,
            },
        )

    return fake_fit


def test_phase4er_orchestration_preserves_test_lock(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _dataset(),
    )

    monkeypatch.setattr(
        experiment,
        "fit_phase4er_density_candidate",
        _fake_fit_factory(),
    )

    payload = (
        experiment.run_phase4er_validation_selection()
    )

    assert payload["phase"] == "4E-R"
    assert payload["phase4f_status"] == "locked"

    assert payload["data_usage"][
        "test_targets_accessed"
    ] is False

    assert payload["data_usage"][
        "train_rows"
    ] == 4096

    assert payload["data_usage"][
        "validation_rows"
    ] == 2048

    assert payload["data_usage"][
        "test_rows"
    ] == 2048

    redevelopment = payload[
        "density_redevelopment"
    ]

    assert redevelopment[
        "candidate_count"
    ] == 16

    assert redevelopment[
        "monotonic_cst"
    ] == [1, 1]

    assert redevelopment[
        "early_stopping"
    ] is False

    assert redevelopment[
        "selection"
    ][
        "selected_candidate_id"
    ] == enumerate_phase4er_density_specs()[
        0
    ].candidate_id

    assert payload["gates"][
        "physics_acceptance_performed"
    ] is False

    assert payload["gates"][
        "final_train_validation_refit_allowed"
    ] is False

    assert payload["gates"][
        "locked_test_evaluation_performed"
    ] is False


def test_phase4er_orchestration_rejects_exposed_test_targets(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _dataset(
            expose_test_targets=True
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        experiment.run_phase4er_validation_selection()


def test_phase4er_orchestration_rejects_wrong_split_size(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _dataset(
            train_rows=4095
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected frozen Phase 4 split sizes",
    ):
        experiment.run_phase4er_validation_selection()


def test_phase4er_orchestration_rejects_wrong_feature_contract(
    monkeypatch,
):
    dataset = _dataset()

    invalid = SurrogateDataset(
        feature_names=(
            "wrong_feature",
            "target_pressure_mTorr",
        ),
        target_names=dataset.target_names,
        splits=dataset.splits,
    )

    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: invalid,
    )

    with pytest.raises(
        RuntimeError,
        match="feature contract",
    ):
        experiment.run_phase4er_validation_selection()


def test_phase4er_writer_serializes_payload(
    tmp_path,
    monkeypatch,
):
    payload = {
        "phase": "4E-R",
        "phase4f_status": "locked",
        "data_usage": {
            "test_targets_accessed": False,
            "train_validation_refit_performed": False,
        },
        "gates": {
            "physics_acceptance_performed": False,
            "final_train_validation_refit_allowed": False,
            "locked_test_evaluation_performed": False,
        },
    }

    monkeypatch.setattr(
        experiment,
        "run_phase4er_validation_selection",
        lambda: payload,
    )

    destination = (
        tmp_path
        / "phase4er_validation_selection.json"
    )

    result = (
        experiment.write_phase4er_validation_selection(
            destination
        )
    )

    assert result == destination

    observed = json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    )

    assert observed == payload

    assert observed[
        "phase4f_status"
    ] == "locked"

    assert observed[
        "data_usage"
    ][
        "test_targets_accessed"
    ] is False

    assert observed[
        "gates"
    ][
        "final_train_validation_refit_allowed"
    ] is False
