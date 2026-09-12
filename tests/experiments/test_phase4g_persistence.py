"""Tests for Phase 4G persistence and reproducible inference."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from plasma_ai.surrogate.classical import (
    ClassicalCandidateSpec,
    build_classical_candidate,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4e_candidates import (
    FittedSelectedCandidate,
    SelectedCandidateConfiguration,
)
from plasma_ai.surrogate.phase4er_density import (
    build_phase4er_density_candidate,
)
from plasma_ai.surrogate.phase4er_final_refit import (
    Phase4ERFinalRefit,
)
from plasma_ai.surrogate.transforms import TargetTransform
import plasma_ai.surrogate.phase4g_persistence as persistence


def _final_refit_fixture():
    X = np.asarray(
        [
            [15.0, 10.0],
            [15.0, 60.0],
            [30.0, 25.0],
            [45.0, 35.0],
            [60.0, 45.0],
            [75.0, 55.0],
            [90.0, 10.0],
            [90.0, 60.0],
        ],
        dtype=np.float64,
    )

    density_y = np.asarray(
        [
            1.8e16,
            4.0e16,
            5.0e16,
            8.0e16,
            1.1e17,
            1.5e17,
            1.7e17,
            2.2e17,
        ],
        dtype=np.float64,
    )

    temperature_y = np.asarray(
        [
            2.05,
            1.55,
            1.88,
            1.78,
            1.70,
            1.63,
            2.02,
            1.57,
        ],
        dtype=np.float64,
    )

    density_spec = ClassicalCandidateSpec(
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
    )

    temperature_spec = ClassicalCandidateSpec(
        candidate_id=(
            "extra_trees_n500_depthnone_leaf1_features1"
        ),
        model_name="extra_trees",
        parameters={
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
    )

    density_config = SelectedCandidateConfiguration(
        target_name=DENSITY_TARGET,
        transform_name="log10",
        candidate_spec=density_spec,
    )

    temperature_config = SelectedCandidateConfiguration(
        target_name=TEMPERATURE_TARGET,
        transform_name="identity",
        candidate_spec=temperature_spec,
    )

    density_transform = TargetTransform(
        "log10"
    )

    temperature_transform = TargetTransform(
        "identity"
    )

    density_model = build_phase4er_density_candidate(
        density_spec
    )

    temperature_model = build_classical_candidate(
        temperature_spec
    )

    density_model.fit(
        X,
        density_transform.forward(
            density_y
        ),
    )

    temperature_model.fit(
        X,
        temperature_transform.forward(
            temperature_y
        ),
    )

    return Phase4ERFinalRefit(
        density=FittedSelectedCandidate(
            configuration=density_config,
            model=density_model,
            transform=density_transform,
            fit_rows=6144,
        ),
        temperature=FittedSelectedCandidate(
            configuration=temperature_config,
            model=temperature_model,
            transform=temperature_transform,
            fit_rows=6144,
        ),
        fit_rows=6144,
        test_targets_accessed=False,
    )


def test_phase4g_protocol_is_frozen_and_test_rerun_is_prohibited():
    protocol = persistence.load_phase4g_protocol()

    assert protocol[
        "status"
    ] == "frozen_before_benchmark_execution"

    assert protocol[
        "test_status"
    ][
        "consumed"
    ] is True

    assert protocol[
        "test_status"
    ][
        "rerun_permitted"
    ] is False

    assert protocol[
        "persistence"
    ][
        "pickle_protocol"
    ] == 5


def test_inference_matrix_contract_rejects_invalid_inputs():
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        persistence._validated_inference_matrix(
            [15.0, 10.0]
        )

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        persistence._validated_inference_matrix(
            np.empty(
                (0, 2)
            )
        )

    with pytest.raises(
        ValueError,
        match="exactly two frozen features",
    ):
        persistence._validated_inference_matrix(
            np.ones(
                (2, 3)
            )
        )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        persistence._validated_inference_matrix(
            [
                [15.0, 10.0],
                [np.nan, 20.0],
            ]
        )


def test_persist_load_and_predict_roundtrip_is_exact(
    tmp_path,
    monkeypatch,
):
    final_refit = _final_refit_fixture()

    monkeypatch.setattr(
        persistence,
        "_validate_source_evidence",
        lambda protocol: None,
    )

    monkeypatch.setattr(
        persistence,
        "_git_state",
        lambda: (
            "abc123",
            True,
        ),
    )

    monkeypatch.setattr(
        persistence,
        "run_phase4er_final_refit",
        lambda: final_refit,
    )

    manifest_path = (
        persistence.write_phase4g_persisted_surrogate(
            tmp_path
        )
    )

    assert manifest_path.exists()

    density_path = (
        tmp_path
        / persistence.DENSITY_MODEL_FILENAME
    )

    temperature_path = (
        tmp_path
        / persistence.TEMPERATURE_MODEL_FILENAME
    )

    assert density_path.exists()
    assert temperature_path.exists()

    loaded = persistence.load_phase4g_surrogate(
        manifest_path
    )

    X = np.asarray(
        [
            [15.0, 10.0],
            [52.5, 35.0],
            [90.0, 60.0],
        ],
        dtype=np.float64,
    )

    expected_density = (
        final_refit.density.predict_physical(
            X
        )
    )

    expected_temperature = (
        final_refit.temperature.predict_physical(
            X
        )
    )

    observed = loaded.predict_physical(
        X
    )

    assert np.array_equal(
        observed.electron_density_m3,
        expected_density,
    )

    assert np.array_equal(
        observed.electron_temperature_eV,
        expected_temperature,
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest[
        "final_fit"
    ][
        "rows"
    ] == 6144

    assert manifest[
        "final_fit"
    ][
        "test_targets_accessed"
    ] is False

    assert manifest[
        "models"
    ][
        "density"
    ][
        "target_transform"
    ] == "log10"

    assert manifest[
        "models"
    ][
        "temperature"
    ][
        "target_transform"
    ] == "identity"

    # Metadata corruption must be rejected before model loading.
    original_manifest_text = manifest_path.read_text(
        encoding="utf-8"
    )

    corrupt_manifest = dict(
        manifest
    )

    corrupt_manifest[
        "feature_contract"
    ] = dict(
        manifest[
            "feature_contract"
        ]
    )

    corrupt_manifest[
        "feature_contract"
    ][
        "feature_names"
    ] = [
        "target_pressure_mTorr",
        "nominal_absorbed_power_W",
    ]

    manifest_path.write_text(
        json.dumps(
            corrupt_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="feature order",
    ):
        persistence.load_phase4g_surrogate(
            manifest_path
        )

    manifest_path.write_text(
        original_manifest_text,
        encoding="utf-8",
    )

    # Binary corruption must be rejected by SHA-256 before pickle.loads.
    density_path.write_bytes(
        density_path.read_bytes()
        + b"corrupt"
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        persistence.load_phase4g_surrogate(
            manifest_path
        )


def test_persistence_rejects_final_refit_that_accessed_test(
    tmp_path,
    monkeypatch,
):
    final_refit = _final_refit_fixture()

    unsafe = Phase4ERFinalRefit(
        density=final_refit.density,
        temperature=final_refit.temperature,
        fit_rows=6144,
        test_targets_accessed=True,
    )

    monkeypatch.setattr(
        persistence,
        "_validate_source_evidence",
        lambda protocol: None,
    )

    monkeypatch.setattr(
        persistence,
        "_git_state",
        lambda: (
            "abc123",
            True,
        ),
    )

    monkeypatch.setattr(
        persistence,
        "run_phase4er_final_refit",
        lambda: unsafe,
    )

    with pytest.raises(
        RuntimeError,
        match="must not access TEST targets",
    ):
        persistence.write_phase4g_persisted_surrogate(
            tmp_path
        )


def test_production_persistence_requires_clean_repository(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        persistence,
        "_validate_source_evidence",
        lambda protocol: None,
    )

    monkeypatch.setattr(
        persistence,
        "_git_state",
        lambda: (
            "abc123",
            False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="clean committed repository",
    ):
        persistence.write_phase4g_persisted_surrogate(
            tmp_path
        )
