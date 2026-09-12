"""Tests for the Phase 4E-R final TRAIN+VALIDATION refit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from plasma_ai.surrogate.classical import (
    ClassicalCandidateSpec,
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
    SelectedCandidateConfiguration,
)
import plasma_ai.surrogate.phase4er_final_refit as refit


def _accepted_payload():
    return {
        "phase": "4E-R",
        "phase4f_status": "locked",
        "data_usage": {
            "test_targets_accessed": False,
            "train_validation_refit_performed": False,
        },
        "acceptance": {
            "density_passed": True,
            "temperature_passed": True,
            "overall_passed": True,
            "final_train_validation_refit_allowed": True,
        },
        "gates": {
            "locked_test_evaluation_performed": False,
        },
    }


def _write_acceptance(
    path,
    payload=None,
):
    path.write_text(
        json.dumps(
            payload or _accepted_payload(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dataset(
    *,
    expose_test_targets=False,
    train_rows=4096,
    validation_rows=2048,
):
    train_X = np.column_stack(
        (
            np.linspace(
                15.0,
                90.0,
                train_rows,
            ),
            np.linspace(
                10.0,
                60.0,
                train_rows,
            ),
        )
    )

    validation_X = np.column_stack(
        (
            np.linspace(
                15.0,
                90.0,
                validation_rows,
            ),
            np.linspace(
                10.0,
                60.0,
                validation_rows,
            ),
        )
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

    test_X = np.ones(
        (2048, 2),
        dtype=float,
    )

    test_y = (
        np.column_stack(
            (
                np.full(
                    2048,
                    1.0e17,
                ),
                np.full(
                    2048,
                    2.0,
                ),
            )
        )
        if expose_test_targets
        else None
    )

    return SurrogateDataset(
        feature_names=refit.EXPECTED_FEATURE_NAMES,
        target_names=refit.EXPECTED_TARGET_NAMES,
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


def _density_configuration():
    return SelectedCandidateConfiguration(
        target_name=DENSITY_TARGET,
        transform_name="log10",
        candidate_spec=ClassicalCandidateSpec(
            candidate_id="density_candidate",
            model_name="hist_gradient_boosting",
            parameters={
                "learning_rate": 0.05,
                "max_iter": 400,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.1,
            },
        ),
    )


def _temperature_configuration():
    return SelectedCandidateConfiguration(
        target_name=TEMPERATURE_TARGET,
        transform_name="identity",
        candidate_spec=ClassicalCandidateSpec(
            candidate_id="temperature_candidate",
            model_name="extra_trees",
            parameters={
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": 1.0,
            },
        ),
    )


class _DummyModel:
    def __init__(self):
        self.fit_X = None
        self.fit_y = None

    def fit(
        self,
        X,
        y,
    ):
        self.fit_X = np.asarray(
            X
        ).copy()

        self.fit_y = np.asarray(
            y
        ).copy()

        return self

    def predict(
        self,
        X,
    ):
        return np.zeros(
            np.asarray(X).shape[0],
            dtype=float,
        )


def test_validate_phase4er_refit_gate_accepts_frozen_pass(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "phase4er_physics_acceptance.json"
    )

    _write_acceptance(
        path
    )

    monkeypatch.setattr(
        refit,
        "file_sha256",
        lambda value: refit.EXPECTED_ACCEPTANCE_SHA256,
    )

    payload = (
        refit.validate_phase4er_refit_gate(
            path
        )
    )

    assert payload[
        "acceptance"
    ][
        "overall_passed"
    ] is True

    assert payload[
        "phase4f_status"
    ] == "locked"


def test_validate_phase4er_refit_gate_rejects_hash_mismatch(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "phase4er_physics_acceptance.json"
    )

    _write_acceptance(
        path
    )

    monkeypatch.setattr(
        refit,
        "file_sha256",
        lambda value: "wrong-hash",
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        refit.validate_phase4er_refit_gate(
            path
        )


def test_validate_phase4er_refit_gate_rejects_failed_acceptance(
    tmp_path,
    monkeypatch,
):
    payload = _accepted_payload()

    payload[
        "acceptance"
    ][
        "overall_passed"
    ] = False

    path = (
        tmp_path
        / "phase4er_physics_acceptance.json"
    )

    _write_acceptance(
        path,
        payload,
    )

    monkeypatch.setattr(
        refit,
        "file_sha256",
        lambda value: refit.EXPECTED_ACCEPTANCE_SHA256,
    )

    with pytest.raises(
        ValueError,
        match="Overall Phase 4E-R physics acceptance did not pass",
    ):
        refit.validate_phase4er_refit_gate(
            path
        )


def test_final_refit_uses_exactly_6144_rows_and_preserves_test_lock(
    monkeypatch,
):
    dataset = _dataset()

    monkeypatch.setattr(
        refit,
        "validate_phase4er_refit_gate",
        lambda path: _accepted_payload(),
    )

    monkeypatch.setattr(
        refit,
        "load_phase4_dataset",
        lambda: dataset,
    )

    monkeypatch.setattr(
        refit,
        "load_phase4er_density_configuration",
        lambda path: _density_configuration(),
    )

    monkeypatch.setattr(
        refit,
        "load_phase4d_selected_configurations",
        lambda path: (
            SimpleNamespace(
                target_name=DENSITY_TARGET,
            ),
            _temperature_configuration(),
        ),
    )

    density_model = _DummyModel()
    temperature_model = _DummyModel()

    monkeypatch.setattr(
        refit,
        "build_phase4er_density_candidate",
        lambda spec: density_model,
    )

    monkeypatch.setattr(
        refit,
        "build_classical_candidate",
        lambda spec: temperature_model,
    )

    result = (
        refit.run_phase4er_final_refit()
    )

    assert result.fit_rows == 6144
    assert result.test_targets_accessed is False

    assert result.density.fit_rows == 6144
    assert result.temperature.fit_rows == 6144

    assert density_model.fit_X.shape == (
        6144,
        2,
    )

    assert temperature_model.fit_X.shape == (
        6144,
        2,
    )

    assert density_model.fit_y.shape == (
        6144,
    )

    assert temperature_model.fit_y.shape == (
        6144,
    )

    assert np.isfinite(
        density_model.fit_y
    ).all()

    assert np.isfinite(
        temperature_model.fit_y
    ).all()


def test_final_refit_rejects_exposed_test_targets(
    monkeypatch,
):
    monkeypatch.setattr(
        refit,
        "validate_phase4er_refit_gate",
        lambda path: _accepted_payload(),
    )

    monkeypatch.setattr(
        refit,
        "load_phase4_dataset",
        lambda: _dataset(
            expose_test_targets=True
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        refit.run_phase4er_final_refit()


def test_final_refit_rejects_wrong_combined_row_count(
    monkeypatch,
):
    monkeypatch.setattr(
        refit,
        "validate_phase4er_refit_gate",
        lambda path: _accepted_payload(),
    )

    monkeypatch.setattr(
        refit,
        "load_phase4_dataset",
        lambda: _dataset(
            validation_rows=2047
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="VALIDATION row count",
    ):
        refit.run_phase4er_final_refit()
