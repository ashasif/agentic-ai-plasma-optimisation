"""Tests for Phase 4E-R physics-aware acceptance orchestration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
import plasma_ai.surrogate.phase4er_physics_acceptance_experiment as experiment


def _grid():
    power = np.asarray(
        [15.0, 30.0, 45.0],
        dtype=float,
    )

    pressure = np.asarray(
        [10.0, 20.0, 30.0],
        dtype=float,
    )

    features = np.asarray(
        [
            [p, q]
            for p in power
            for q in pressure
        ],
        dtype=float,
    )

    return SimpleNamespace(
        absorbed_power_W=power,
        target_pressure_mTorr=pressure,
        features=features,
        total_points=9,
    )


def _source_arrays():
    grid = _grid()

    density = np.asarray(
        [
            1.0, 2.0, 3.0,
            2.0, 3.0, 4.0,
            3.0, 4.0, 5.0,
        ],
        dtype=float,
    )

    temperature = np.asarray(
        [
            5.0, 4.0, 3.0,
            5.5, 4.5, 3.5,
            6.0, 5.0, 4.0,
        ],
        dtype=float,
    )

    return (
        grid.features.copy(),
        density,
        temperature,
    )


class _DummyFittedCandidate:
    def __init__(
        self,
        *,
        target,
        transform,
        candidate_id,
        model_name,
        parameters,
        prediction,
        fit_rows=4096,
    ):
        self.configuration = SimpleNamespace(
            target_name=target,
            transform_name=transform,
            candidate_spec=SimpleNamespace(
                candidate_id=candidate_id,
                model_name=model_name,
                parameters=dict(parameters),
            ),
        )

        self.fit_rows = fit_rows
        self._prediction = np.asarray(
            prediction,
            dtype=float,
        )

    def predict_physical(
        self,
        X,
    ):
        assert np.asarray(X).shape == (
            9,
            2,
        )

        return self._prediction.copy()


def _passing_models():
    density = _DummyFittedCandidate(
        target=DENSITY_TARGET,
        transform="log10",
        candidate_id=(
            "phase4er_hist_gradient_boosting_"
            "lr0.05_iter400_leaves31_l20.1"
        ),
        model_name="hist_gradient_boosting",
        parameters={
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
        },
        prediction=[
            1.1, 2.1, 3.1,
            2.1, 3.1, 4.1,
            3.1, 4.1, 5.1,
        ],
    )

    temperature = _DummyFittedCandidate(
        target=TEMPERATURE_TARGET,
        transform="identity",
        candidate_id="phase4d_temperature_candidate",
        model_name="extra_trees",
        parameters={
            "n_estimators": 500,
        },
        prediction=[
            5.1, 4.1, 3.1,
            5.6, 4.6, 3.6,
            6.1, 5.1, 4.1,
        ],
    )

    return (
        density,
        temperature,
    )


def _patch_common(
    monkeypatch,
    models,
):
    grid = _grid()

    monkeypatch.setattr(
        experiment,
        "build_phase4e_probe_grid",
        lambda: grid,
    )

    monkeypatch.setattr(
        experiment,
        "_load_frozen_source_reference",
        lambda path: _source_arrays(),
    )

    monkeypatch.setattr(
        experiment,
        "fit_phase4er_acceptance_candidates",
        lambda **kwargs: models,
    )


def test_phase4er_physics_acceptance_pass(
    monkeypatch,
):
    _patch_common(
        monkeypatch,
        _passing_models(),
    )

    payload = (
        experiment.run_phase4er_physics_acceptance()
    )

    assert payload["phase"] == "4E-R"
    assert payload["phase4f_status"] == "locked"

    assert payload[
        "data_usage"
    ][
        "test_targets_accessed"
    ] is False

    assert payload[
        "data_usage"
    ][
        "train_validation_refit_performed"
    ] is False

    assert payload[
        "density"
    ][
        "passed"
    ] is True

    assert payload[
        "temperature"
    ][
        "passed"
    ] is True

    assert payload[
        "acceptance"
    ][
        "overall_passed"
    ] is True

    assert payload[
        "acceptance"
    ][
        "final_train_validation_refit_allowed"
    ] is True

    assert payload[
        "gates"
    ][
        "locked_test_evaluation_performed"
    ] is False


def test_phase4er_physics_acceptance_density_failure_blocks_refit(
    monkeypatch,
):
    (
        density,
        temperature,
    ) = _passing_models()

    density._prediction = np.asarray(
        [
            1.1, 2.1, 3.1,
            2.1, 3.1, 2.9,
            3.1, 4.1, 5.1,
        ],
        dtype=float,
    )

    _patch_common(
        monkeypatch,
        (
            density,
            temperature,
        ),
    )

    payload = (
        experiment.run_phase4er_physics_acceptance()
    )

    assert payload[
        "density"
    ][
        "passed"
    ] is False

    assert payload[
        "acceptance"
    ][
        "overall_passed"
    ] is False

    assert payload[
        "acceptance"
    ][
        "final_train_validation_refit_allowed"
    ] is False

    assert payload[
        "gates"
    ][
        "final_train_validation_refit_allowed"
    ] is False

    assert payload[
        "gates"
    ][
        "final_train_validation_refit_performed"
    ] is False


def test_phase4er_physics_acceptance_rejects_grid_mismatch(
    monkeypatch,
):
    grid = _grid()

    monkeypatch.setattr(
        experiment,
        "build_phase4e_probe_grid",
        lambda: grid,
    )

    (
        features,
        density,
        temperature,
    ) = _source_arrays()

    features = features.copy()
    features[0, 0] = 999.0

    monkeypatch.setattr(
        experiment,
        "_load_frozen_source_reference",
        lambda path: (
            features,
            density,
            temperature,
        ),
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="feature ordering",
    ):
        experiment.run_phase4er_physics_acceptance()


def test_phase4er_physics_acceptance_writer(
    tmp_path,
    monkeypatch,
):
    payload = {
        "phase": "4E-R",
        "phase4f_status": "locked",
        "data_usage": {
            "test_targets_accessed": False,
        },
        "acceptance": {
            "overall_passed": True,
            "final_train_validation_refit_allowed": True,
        },
        "gates": {
            "physics_acceptance_performed": True,
            "final_train_validation_refit_performed": False,
            "locked_test_evaluation_performed": False,
        },
    }

    monkeypatch.setattr(
        experiment,
        "run_phase4er_physics_acceptance",
        lambda **kwargs: payload,
    )

    destination = (
        tmp_path
        / "phase4er_physics_acceptance.json"
    )

    result = (
        experiment.write_phase4er_physics_acceptance(
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
        "data_usage"
    ][
        "test_targets_accessed"
    ] is False
