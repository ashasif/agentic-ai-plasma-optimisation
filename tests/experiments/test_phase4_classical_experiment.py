"""Tests for the Phase 4C classical benchmark experiment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import plasma_ai.surrogate.classical_experiment as experiment
from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)


class _FakeDataset:
    def __init__(self):
        self._splits = {
            "train": SimpleNamespace(
                X=np.zeros((8, 2)),
                y=np.ones((8, 2)),
            ),
            "validation": SimpleNamespace(
                X=np.zeros((4, 2)),
                y=np.ones((4, 2)),
            ),
            "test": SimpleNamespace(
                X=np.zeros((4, 2)),
                y=None,
            ),
        }

    def split(self, name):
        return self._splits[name]


def _fake_result(
    *,
    target_name,
    transform_name,
    spec,
):
    if target_name == (
        "true_electron_density_m3"
    ):
        metrics = {
            "mae": 1.0,
            "rmse": 2.0,
            "r2": 0.9,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": 0.01,
            "p95_absolute_relative_error": 0.02,
            "max_absolute_relative_error": 0.03,
        }
    else:
        metrics = {
            "mae_eV": 0.01,
            "rmse_eV": 0.02,
            "r2": 0.9,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": 0.01,
            "p95_absolute_error_eV": 0.03,
            "max_absolute_error_eV": 0.04,
        }

    return ClassicalValidationResult(
        target_name=target_name,
        transform_name=transform_name,
        candidate_id=spec.candidate_id,
        model_name=spec.model_name,
        parameters=dict(spec.parameters),
        metrics=metrics,
        prediction_sanity={
            "all_finite": True,
            "minimum": 1.0,
            "maximum": 2.0,
            "negative_count": 0,
            "non_positive_count": 0,
        },
    )


def _fake_reference_baselines():
    return {
        "phase": "4B",
        "experiment": "reference_baselines",
        "data_usage": {
            "test_targets_accessed": False,
        },
        "results": [
            {
                "target": "true_electron_density_m3",
                "transform": "identity",
                "model": "linear_regression",
                "metrics": {
                    "p95_absolute_relative_error": 0.40,
                },
            },
            {
                "target": "true_electron_density_m3",
                "transform": "log10",
                "model": "linear_regression",
                "metrics": {
                    "p95_absolute_relative_error": 0.30,
                },
            },
            {
                "target": "true_electron_temperature_eV",
                "transform": "identity",
                "model": "linear_regression",
                "metrics": {
                    "rmse_eV": 0.05,
                },
            },
        ],
    }


def test_phase4c_plan_has_twelve_evaluations(
    monkeypatch,
):
    calls = []

    def fake_loader():
        return _FakeDataset()

    def fake_fit(
        *,
        train_X,
        train_y,
        validation_X,
        validation_y,
        target_name,
        transform_name,
        spec,
    ):
        calls.append(
            (
                target_name,
                transform_name,
                spec.candidate_id,
            )
        )

        return _fake_result(
            target_name=target_name,
            transform_name=transform_name,
            spec=spec,
        )

    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        fake_loader,
    )

    monkeypatch.setattr(
        experiment,
        "fit_classical_candidate",
        fake_fit,
    )

    monkeypatch.setattr(
        experiment,
        "_load_reference_baselines",
        lambda: _fake_reference_baselines(),
    )

    payload = experiment.run_classical_benchmark()

    assert len(calls) == 12
    assert len(payload["results"]) == 12

    assert payload["data_usage"] == {
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_targets_accessed": False,
        "train_rows": 8,
        "validation_rows": 4,
        "test_rows": 4,
    }

    design = payload["benchmark_design"]

    assert design[
        "candidate_configurations"
    ] == 4
    assert design[
        "target_transform_combinations"
    ] == 3
    assert design["total_fits"] == 12
    assert design[
        "full_hyperparameter_search"
    ] is False
    assert design[
        "final_model_selection_performed"
    ] is False
    assert design[
        "final_density_transform_selected"
    ] is False
    assert design[
        "production_model_persisted"
    ] is False
    assert design[
        "phase4d_reserved_for_selection"
    ] is True


def test_phase4c_loader_is_called_without_test_unlock(
    monkeypatch,
):
    loader_calls = []

    def fake_loader():
        loader_calls.append(True)
        return _FakeDataset()

    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        fake_loader,
    )

    monkeypatch.setattr(
        experiment,
        "fit_classical_candidate",
        lambda **kwargs: _fake_result(
            target_name=kwargs["target_name"],
            transform_name=kwargs["transform_name"],
            spec=kwargs["spec"],
        ),
    )

    monkeypatch.setattr(
        experiment,
        "_load_reference_baselines",
        lambda: _fake_reference_baselines(),
    )

    experiment.run_classical_benchmark()

    assert loader_calls == [True]


def test_phase4c_refuses_unlocked_test_targets(
    monkeypatch,
):
    dataset = _FakeDataset()

    dataset._splits["test"] = SimpleNamespace(
        X=np.zeros((4, 2)),
        y=np.ones((4, 2)),
    )

    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: dataset,
    )

    import pytest

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        experiment.run_classical_benchmark()


def test_phase4c_scientific_scope_preserves_claim_boundaries(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _FakeDataset(),
    )

    monkeypatch.setattr(
        experiment,
        "fit_classical_candidate",
        lambda **kwargs: _fake_result(
            target_name=kwargs["target_name"],
            transform_name=kwargs["transform_name"],
            spec=kwargs["spec"],
        ),
    )

    monkeypatch.setattr(
        experiment,
        "_load_reference_baselines",
        lambda: _fake_reference_baselines(),
    )

    payload = experiment.run_classical_benchmark()

    scope = payload["scientific_scope"]

    assert scope["data_type"] == "synthetic data"
    assert scope["source_model"] == (
        "reduced-order argon plasma model"
    )
    assert scope["domain"] == (
        "numerically qualified model envelope"
    )
    assert scope["surrogate_role"] == (
        "surrogate of the reduced-order simulator"
    )

    assert scope["experimental_validation"] is False
    assert scope["industrial_validation"] is False
    assert scope[
        "oipt_operating_range_claim"
    ] is False
    assert scope[
        "reactive_etch_or_deposition_prediction"
    ] is False
    assert scope[
        "wafer_scale_spatial_modelling"
    ] is False
    assert scope[
        "absorbed_power_is_generator_rf_power"
    ] is False


def test_phase4c_baseline_comparison_is_descriptive_only(
    monkeypatch,
):
    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _FakeDataset(),
    )

    monkeypatch.setattr(
        experiment,
        "fit_classical_candidate",
        lambda **kwargs: _fake_result(
            target_name=kwargs["target_name"],
            transform_name=kwargs["transform_name"],
            spec=kwargs["spec"],
        ),
    )

    monkeypatch.setattr(
        experiment,
        "_load_reference_baselines",
        lambda: _fake_reference_baselines(),
    )

    payload = experiment.run_classical_benchmark()

    comparisons = payload[
        "baseline_comparison"
    ]

    assert len(comparisons) == 12

    for comparison in comparisons:
        assert comparison[
            "selection_claim"
        ] is False

        assert (
            comparison["primary_metric"]
            in {
                "p95_absolute_relative_error",
                "rmse_eV",
            }
        )

        assert np.isfinite(
            comparison[
                "relative_improvement_fraction"
            ]
        )
