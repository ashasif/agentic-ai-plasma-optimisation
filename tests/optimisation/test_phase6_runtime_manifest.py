from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from plasma_ai.optimisation import (
    RuntimeManifestError,
    build_runtime_manifest_payload,
    effective_runtime_sha256,
    load_effective_runtime_protocol,
    load_runtime_manifest,
    supported_runtime_statuses,
)


def test_effective_runtime_protocol_identity() -> None:
    protocol = (
        load_effective_runtime_protocol()
    )

    assert protocol.base_sha256 == (
        "ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a"
    )

    assert protocol.amendment_sha256 == (
        "6294323543b9c4d121e4f4a8dcba34f8aad133924b12d63add1b37494232b87e"
    )

    assert protocol.effective_sha256 == (
        "7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83"
    )


def test_effective_runtime_hash_formula() -> None:
    observed = effective_runtime_sha256(
        "ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a",
        "6294323543b9c4d121e4f4a8dcba34f8aad133924b12d63add1b37494232b87e",
    )

    assert observed == (
        "7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83"
    )


def test_status_vocabulary_is_exactly_five_states() -> None:
    protocol = (
        load_effective_runtime_protocol()
    )

    assert supported_runtime_statuses(
        protocol
    ) == (
        "selected_method_accepted",
        "selected_method_accepted_grid_infeasible",
        "grid_fallback_selected_method_infeasible",
        "grid_fallback_selected_method_objective_regression",
        "no_feasible_point_found_under_search_protocol",
    )


def test_manifest_payload_contains_required_frozen_contract() -> None:
    runtime_source = Path(
        "src/plasma_ai/optimisation/runtime.py"
    )

    source_hash = hashlib.sha256(
        runtime_source.read_bytes()
    ).hexdigest()

    payload = build_runtime_manifest_payload(
        runtime_implementation_source_sha256=source_hash,
    )

    assert payload[
        "production_seed"
    ] == 20260914

    assert payload[
        "selected_method"
    ] == "differential_evolution"

    assert len(
        payload[
            "supported_status_vocabulary"
        ]
    ) == 5

    assert payload[
        "effective_runtime_protocol_sha256"
    ] == (
        "7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83"
    )


def _write_valid_temp_manifest(
    path: Path,
) -> dict:
    runtime_source = Path(
        "src/plasma_ai/optimisation/runtime.py"
    )

    source_hash = hashlib.sha256(
        runtime_source.read_bytes()
    ).hexdigest()

    payload = build_runtime_manifest_payload(
        runtime_implementation_source_sha256=source_hash,
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

    return payload


def test_temp_manifest_validates_without_surrogate_loading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    manifest = load_runtime_manifest(
        path
    )

    assert manifest.payload == payload

    assert manifest.sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_manifest_seed_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "production_seed"
    ] = 999

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="Production seed mismatch",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_status_vocabulary_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "supported_status_vocabulary"
    ] = [
        "selected_method_accepted",
    ]

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="status vocabulary",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_runtime_source_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "runtime_implementation_source_sha256"
    ] = "0" * 64

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="runtime_implementation_source_sha256",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_schema_version_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload["schema_version"] = "999.0.0"

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="schema version",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_reference_method_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "mandatory_reference_method"
    ] = "unexpected_reference"

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="Mandatory reference method",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_fallback_method_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "fallback_method"
    ] = "unexpected_fallback"

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="Fallback method",
    ):
        load_runtime_manifest(
            path
        )


@pytest.mark.parametrize(
    "field",
    [
        "grid_input_array_sha256",
        "grid_density_array_sha256",
        "grid_temperature_array_sha256",
    ],
)
def test_manifest_grid_array_hash_drift_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[field] = "0" * 64

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="Frozen grid-array hash mismatch",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_continuous_base_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "continuous_benchmark_base_sha256"
    ] = "0" * 64

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="continuous_benchmark_base_sha256",
    ):
        load_runtime_manifest(
            path
        )


def test_manifest_continuous_amendment_hash_drift_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_manifest.json"

    payload = _write_valid_temp_manifest(
        path
    )

    payload[
        "continuous_benchmark_amendment_sha256"
    ] = "0" * 64

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeManifestError,
        match="continuous_benchmark_amendment_sha256",
    ):
        load_runtime_manifest(
            path
        )
