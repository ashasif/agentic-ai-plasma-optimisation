"""Tests for the Phase 4D validation-selection experiment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import plasma_ai.surrogate.selection_experiment as experiment

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)


DENSITY_TARGET = "true_electron_density_m3"
TEMPERATURE_TARGET = "true_electron_temperature_eV"


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


def _fake_fit(
    *,
    target_name,
    transform_name,
    spec,
    **kwargs,
):
    if target_name == DENSITY_TARGET:
        base = (
            0.01
            if transform_name == "identity"
            else 0.02
        )

        metrics = {
            "mae": 1.0,
            "rmse": 2.0,
            "r2": 0.99,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": base,
            "p95_absolute_relative_error": base,
            "max_absolute_relative_error": 0.05,
        }
    else:
        metrics = {
            "mae_eV": 0.002,
            "rmse_eV": 0.001,
            "r2": 0.99,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": 0.01,
            "p95_absolute_error_eV": 0.003,
            "max_absolute_error_eV": 0.005,
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


def test_phase4d_runs_exactly_108_validation_fits(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        experiment,
        "load_phase4_dataset",
        lambda: _FakeDataset(),
    )

    def recording_fit(**kwargs):
        calls.append(
            (
                kwargs["target_name"],
                kwargs["transform_name"],
                kwargs["spec"].candidate_id,
            )
        )
        return _fake_fit(**kwargs)

    monkeypatch.setattr(
        experiment,
        "fit_classical_candidate",
        recording_fit,
    )

    payload = experiment.run_validation_selection()

    assert len(calls) == 108
    assert len(payload["results"]) == 108

    counts = {}

    for target_name, transform_name, _ in calls:
        key = (
            target_name,
            transform_name,
        )
        counts[key] = counts.get(key, 0) + 1

    assert counts == {
        (DENSITY_TARGET, "identity"): 36,
        (DENSITY_TARGET, "log10"): 36,
        (TEMPERATURE_TARGET, "identity"): 36,
    }


def test_phase4d_preserves_locked_test_targets(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    assert loader_calls == [True]

    assert payload["data_usage"] == {
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_targets_accessed": False,
        "train_rows": 8,
        "validation_rows": 4,
        "test_rows": 4,
    }


def test_phase4d_refuses_unlocked_test_targets(
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

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        experiment.run_validation_selection()


def test_phase4d_selects_density_across_both_transforms(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    density = payload["selection"]["density"]

    assert density["target"] == DENSITY_TARGET
    assert density["selected_transform"] == "identity"

    assert density[
        "candidate_pool_size"
    ] == 72


def test_phase4d_selects_temperature_from_identity_pool(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    temperature = payload[
        "selection"
    ]["temperature"]

    assert temperature[
        "target"
    ] == TEMPERATURE_TARGET

    assert temperature[
        "selected_transform"
    ] == "identity"

    assert temperature[
        "candidate_pool_size"
    ] == 36


def test_phase4d_payload_marks_validation_selection_not_final_test(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    assert payload["phase"] == "4D"
    assert payload["experiment"] == (
        "controlled_validation_selection"
    )

    design = payload["selection_design"]

    assert design[
        "classical_configurations_per_target_transform"
    ] == 36

    assert design[
        "target_transform_combinations"
    ] == 3

    assert design["total_fits"] == 108

    assert design[
        "validation_selection_performed"
    ] is True

    assert design[
        "physics_acceptance_performed"
    ] is False

    assert design[
        "locked_test_evaluation_performed"
    ] is False

    assert design[
        "train_validation_refit_performed"
    ] is False

    assert design[
        "production_model_persisted"
    ] is False


def test_phase4d_scientific_scope_preserves_claim_boundaries(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

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



def test_phase4d_selection_payload_preserves_transform_qualified_trace(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    density = payload["selection"]["density"]

    qualified = density[
        "primary_equivalent_candidates"
    ]

    assert len(qualified) >= 1

    for item in qualified:
        assert set(item) == {
            "candidate_id",
            "transform",
        }

        assert item["transform"] in {
            "identity",
            "log10",
        }



def test_phase4d_selection_payload_records_primary_equivalence_boundary(
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
        _fake_fit,
    )

    payload = experiment.run_validation_selection()

    for key in ("density", "temperature"):
        decision = payload["selection"][key]

        assert "best_primary_value" in decision
        assert (
            "practical_equivalence_relative_fraction"
            in decision
        )
        assert (
            "primary_strict_upper_boundary"
            in decision
        )

        assert decision[
            "practical_equivalence_relative_fraction"
        ] == 0.02

        assert decision[
            "primary_strict_upper_boundary"
        ] >= decision["best_primary_value"]
