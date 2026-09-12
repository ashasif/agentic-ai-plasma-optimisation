"""Tests for Phase 4E-R acceptance-candidate reconstruction."""

from __future__ import annotations

import json

import numpy as np
import pytest

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
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    enumerate_phase4er_density_specs,
)
import plasma_ai.surrogate.phase4er_candidates as candidates


SELECTED_ID = (
    "phase4er_hist_gradient_boosting"
    "_lr0.05_iter400_leaves31_l20.1"
)


def _selected_spec():
    matches = [
        spec
        for spec in enumerate_phase4er_density_specs()
        if spec.candidate_id == SELECTED_ID
    ]

    assert len(matches) == 1

    return matches[0]


def _selection_payload():
    spec = _selected_spec()

    return {
        "phase": "4E-R",
        "phase4f_status": "locked",
        "data_usage": {
            "test_targets_accessed": False,
            "train_validation_refit_performed": False,
        },
        "density_redevelopment": {
            "selection": {
                "target": DENSITY_TARGET,
                "selected_transform": PHASE4ER_DENSITY_TRANSFORM,
                "selected_candidate_id": spec.candidate_id,
                "selected_model": spec.model_name,
                "selected_parameters": dict(
                    spec.parameters
                ),
                "validation_selected_only": True,
            },
        },
    }


def _write_selection(
    path,
):
    path.write_text(
        json.dumps(
            _selection_payload(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dataset(
    *,
    expose_test_targets=False,
):
    train_rows = 4096
    validation_rows = 8
    test_rows = 8

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

    validation_X = np.ones(
        (validation_rows, 2),
        dtype=float,
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
        (test_rows, 2),
        dtype=float,
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
        feature_names=candidates.EXPECTED_FEATURE_NAMES,
        target_names=candidates.EXPECTED_TARGET_NAMES,
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


def test_load_phase4er_density_configuration(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "phase4er_validation_selection.json"
    )

    _write_selection(
        path
    )

    monkeypatch.setattr(
        candidates,
        "file_sha256",
        lambda value: (
            candidates.EXPECTED_PHASE4ER_SELECTION_SHA256
        ),
    )

    configuration = (
        candidates.load_phase4er_density_configuration(
            path
        )
    )

    spec = _selected_spec()

    assert configuration.target_name == DENSITY_TARGET

    assert (
        configuration.transform_name
        == PHASE4ER_DENSITY_TRANSFORM
    )

    assert (
        configuration.candidate_spec.candidate_id
        == spec.candidate_id
    )

    assert (
        configuration.candidate_spec.model_name
        == spec.model_name
    )

    assert (
        dict(
            configuration.candidate_spec.parameters
        )
        == dict(
            spec.parameters
        )
    )


def test_load_phase4er_density_configuration_rejects_hash_mismatch(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "phase4er_validation_selection.json"
    )

    _write_selection(
        path
    )

    monkeypatch.setattr(
        candidates,
        "file_sha256",
        lambda value: "wrong-hash",
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        candidates.load_phase4er_density_configuration(
            path
        )


def test_load_phase4er_density_configuration_rejects_unlocked_phase4f(
    tmp_path,
    monkeypatch,
):
    payload = _selection_payload()

    payload[
        "phase4f_status"
    ] = "unlocked"

    path = (
        tmp_path
        / "phase4er_validation_selection.json"
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        candidates,
        "file_sha256",
        lambda value: (
            candidates.EXPECTED_PHASE4ER_SELECTION_SHA256
        ),
    )

    with pytest.raises(
        ValueError,
        match="Phase 4F must remain locked",
    ):
        candidates.load_phase4er_density_configuration(
            path
        )


def test_fit_phase4er_density_on_train_preserves_test_lock(
    monkeypatch,
):
    spec = _selected_spec()

    configuration = SelectedCandidateConfiguration(
        target_name=DENSITY_TARGET,
        transform_name=PHASE4ER_DENSITY_TRANSFORM,
        candidate_spec=spec,
    )

    class DummyModel:
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
            return np.full(
                np.asarray(X).shape[0],
                17.0,
            )

    model = DummyModel()

    monkeypatch.setattr(
        candidates,
        "build_phase4er_density_candidate",
        lambda candidate_spec: model,
    )

    fitted = (
        candidates.fit_phase4er_density_on_train(
            _dataset(),
            configuration,
        )
    )

    assert fitted.fit_rows == 4096

    assert (
        fitted.configuration.candidate_spec.candidate_id
        == SELECTED_ID
    )

    assert model.fit_X.shape == (
        4096,
        2,
    )

    assert model.fit_y.shape == (
        4096,
    )

    assert np.isfinite(
        model.fit_y
    ).all()


def test_fit_phase4er_density_on_train_rejects_exposed_test_targets():
    spec = _selected_spec()

    configuration = SelectedCandidateConfiguration(
        target_name=DENSITY_TARGET,
        transform_name=PHASE4ER_DENSITY_TRANSFORM,
        candidate_spec=spec,
    )

    with pytest.raises(
        RuntimeError,
        match="TEST targets must remain locked",
    ):
        candidates.fit_phase4er_density_on_train(
            _dataset(
                expose_test_targets=True
            ),
            configuration,
        )
