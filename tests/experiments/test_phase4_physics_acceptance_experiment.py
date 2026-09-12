"""Tests for the Phase 4E physics-aware acceptance experiment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from plasma_ai.surrogate.physics_acceptance_experiment import (
    EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256,
    _load_frozen_source_reference,
    write_phase4e_physics_acceptance,
)


def _write_reference(
    path,
    *,
    features,
    density,
    temperature,
    stored_hash=None,
):
    matrix = np.column_stack(
        (
            features,
            density,
            temperature,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    observed_hash = hashlib.sha256(
        matrix.tobytes(
            order="C"
        )
    ).hexdigest()

    rows = [
        {
            "nominal_absorbed_power_W": float(
                features[index, 0]
            ),
            "target_pressure_mTorr": float(
                features[index, 1]
            ),
            "true_electron_density_m3": float(
                density[index]
            ),
            "true_electron_temperature_eV": float(
                temperature[index]
            ),
        }
        for index in range(
            features.shape[0]
        )
    ]

    payload = {
        "phase": "4E",
        "artifact": "source_reference_grid",
        "source_reference_gate": {
            "all_targets_finite": True,
            "all_qualification_valid": True,
        },
        "reference_array_contract": {
            "sha256": (
                observed_hash
                if stored_hash is None
                else stored_hash
            ),
        },
        "rows": rows,
    }

    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    return observed_hash


def test_frozen_source_reference_loads_real_committed_artifact():
    (
        features,
        density,
        temperature,
    ) = _load_frozen_source_reference(
        "results/phase4/source_reference_grid.json"
    )

    assert features.shape == (
        1681,
        2,
    )

    assert density.shape == (
        1681,
    )

    assert temperature.shape == (
        1681,
    )

    assert np.isfinite(
        features
    ).all()

    assert np.all(
        density > 0.0
    )

    assert np.all(
        temperature > 0.0
    )


def test_expected_source_reference_hash_is_frozen():
    assert (
        EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256
        == "4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a"
    )


def test_source_reference_rejects_wrong_phase(
    tmp_path,
):
    source = json.loads(
        (
            __import__("pathlib").Path(
                "results/phase4/source_reference_grid.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )

    source["phase"] = "4F"

    path = tmp_path / "reference.json"

    path.write_text(
        json.dumps(
            source
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Phase 4E",
    ):
        _load_frozen_source_reference(
            path
        )


def test_source_reference_rejects_failed_gate(
    tmp_path,
):
    source = json.loads(
        (
            __import__("pathlib").Path(
                "results/phase4/source_reference_grid.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )

    source[
        "source_reference_gate"
    ][
        "all_qualification_valid"
    ] = False

    path = tmp_path / "reference.json"

    path.write_text(
        json.dumps(
            source
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="qualification",
    ):
        _load_frozen_source_reference(
            path
        )


def test_source_reference_rejects_row_count_drift(
    tmp_path,
):
    source = json.loads(
        (
            __import__("pathlib").Path(
                "results/phase4/source_reference_grid.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )

    source["rows"] = source[
        "rows"
    ][:-1]

    path = tmp_path / "reference.json"

    path.write_text(
        json.dumps(
            source
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="1681",
    ):
        _load_frozen_source_reference(
            path
        )


def test_source_reference_rejects_stored_hash_mismatch(
    tmp_path,
):
    source = json.loads(
        (
            __import__("pathlib").Path(
                "results/phase4/source_reference_grid.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )

    source[
        "reference_array_contract"
    ][
        "sha256"
    ] = "0" * 64

    path = tmp_path / "reference.json"

    path.write_text(
        json.dumps(
            source
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="stored artifact contract",
    ):
        _load_frozen_source_reference(
            path
        )


def test_source_reference_rejects_numerical_drift_even_with_updated_local_hash(
    tmp_path,
):
    source = json.loads(
        (
            __import__("pathlib").Path(
                "results/phase4/source_reference_grid.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )

    source["rows"][0][
        "true_electron_density_m3"
    ] *= 1.001

    features = np.asarray(
        [
            [
                row["nominal_absorbed_power_W"],
                row["target_pressure_mTorr"],
            ]
            for row in source["rows"]
        ],
        dtype=np.float64,
    )

    density = np.asarray(
        [
            row["true_electron_density_m3"]
            for row in source["rows"]
        ],
        dtype=np.float64,
    )

    temperature = np.asarray(
        [
            row["true_electron_temperature_eV"]
            for row in source["rows"]
        ],
        dtype=np.float64,
    )

    matrix = np.column_stack(
        (
            features,
            density,
            temperature,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    source[
        "reference_array_contract"
    ][
        "sha256"
    ] = hashlib.sha256(
        matrix.tobytes(
            order="C"
        )
    ).hexdigest()

    path = tmp_path / "reference.json"

    path.write_text(
        json.dumps(
            source
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="frozen Phase 4E reference",
    ):
        _load_frozen_source_reference(
            path
        )



def test_write_phase4e_acceptance_serializes_payload(
    tmp_path,
    monkeypatch,
):
    payload = {
        "phase": "4E",
        "stage": "physics_aware_pre_test_acceptance",
        "acceptance": {
            "density_passed": False,
            "temperature_passed": True,
            "overall_passed": False,
            "final_train_validation_refit_allowed": False,
        },
        "data_usage": {
            "test_targets_accessed": False,
        },
    }

    monkeypatch.setattr(
        "plasma_ai.surrogate.physics_acceptance_experiment."
        "run_phase4e_physics_acceptance",
        lambda **kwargs: payload,
    )

    destination = (
        tmp_path
        / "physics_acceptance.json"
    )

    result = write_phase4e_physics_acceptance(
        destination
    )

    assert result == destination

    observed = json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    )

    assert observed == payload


def test_written_stop_result_preserves_phase4e_gate(
    tmp_path,
    monkeypatch,
):
    payload = {
        "phase": "4E",
        "acceptance": {
            "density_passed": False,
            "temperature_passed": True,
            "overall_passed": False,
            "final_train_validation_refit_allowed": False,
        },
        "data_usage": {
            "test_targets_accessed": False,
            "validation_used_for_fitting": False,
            "validation_used_for_acceptance": False,
        },
    }

    monkeypatch.setattr(
        "plasma_ai.surrogate.physics_acceptance_experiment."
        "run_phase4e_physics_acceptance",
        lambda **kwargs: payload,
    )

    destination = (
        tmp_path
        / "physics_acceptance.json"
    )

    write_phase4e_physics_acceptance(
        destination
    )

    observed = json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    )

    acceptance = observed[
        "acceptance"
    ]

    assert acceptance[
        "overall_passed"
    ] is False

    assert acceptance[
        "final_train_validation_refit_allowed"
    ] is False

    assert observed[
        "data_usage"
    ][
        "test_targets_accessed"
    ] is False
