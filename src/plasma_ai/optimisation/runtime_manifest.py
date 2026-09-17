"""Phase 6F production-runtime manifest contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from plasma_ai.optimisation.protocol import file_sha256


DEFAULT_RUNTIME_PROTOCOL_PATH = Path(
    "configs/phase6/runtime_protocol.json"
)

DEFAULT_RUNTIME_AMENDMENT_PATH = Path(
    "configs/phase6/runtime_protocol_amendment_001.json"
)

DEFAULT_RUNTIME_MANIFEST_PATH = Path(
    "artifacts/phase6/optimization_runtime_manifest.json"
)


EXPECTED_RUNTIME_BASE_SHA256 = (
    "ab7e7aebe73f57e7bf3801c24a18a42f3d86a3c0f2c4adb157707864d957656a"
)

EXPECTED_RUNTIME_AMENDMENT_SHA256 = (
    "6294323543b9c4d121e4f4a8dcba34f8aad133924b12d63add1b37494232b87e"
)

EXPECTED_EFFECTIVE_RUNTIME_SHA256 = (
    "7003151b7d8e549c73b76c27685e710df8a69d56e7e08536f3d97ee84a438d83"
)


class RuntimeManifestError(RuntimeError):
    """Raised when Phase 6 runtime persistence validation fails."""


@dataclass(frozen=True)
class EffectiveRuntimeProtocol:
    """Frozen Phase 6F1 + Phase 6F1.1 protocol pair."""

    base: dict[str, Any]
    amendment: dict[str, Any]
    base_sha256: str
    amendment_sha256: str
    effective_sha256: str


@dataclass(frozen=True)
class RuntimeManifest:
    """Validated Phase 6 production-runtime manifest."""

    payload: dict[str, Any]
    path: Path
    sha256: str


def effective_runtime_sha256(
    base_sha256: str,
    amendment_sha256: str,
) -> str:
    """Return frozen effective-runtime identity."""

    return hashlib.sha256(
        (
            base_sha256
            + "\n"
            + amendment_sha256
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def load_effective_runtime_protocol(
    base_path: str | Path = DEFAULT_RUNTIME_PROTOCOL_PATH,
    amendment_path: str | Path = DEFAULT_RUNTIME_AMENDMENT_PATH,
) -> EffectiveRuntimeProtocol:
    """Load and verify the frozen Phase 6 runtime contract."""

    base_source = Path(base_path)
    amendment_source = Path(amendment_path)

    base_hash = file_sha256(
        base_source
    )

    amendment_hash = file_sha256(
        amendment_source
    )

    if base_hash != EXPECTED_RUNTIME_BASE_SHA256:
        raise RuntimeManifestError(
            "Runtime base protocol SHA-256 mismatch."
        )

    if amendment_hash != EXPECTED_RUNTIME_AMENDMENT_SHA256:
        raise RuntimeManifestError(
            "Runtime amendment SHA-256 mismatch."
        )

    effective_hash = effective_runtime_sha256(
        base_hash,
        amendment_hash,
    )

    if effective_hash != EXPECTED_EFFECTIVE_RUNTIME_SHA256:
        raise RuntimeManifestError(
            "Effective runtime-protocol identity mismatch."
        )

    try:
        base = json.loads(
            base_source.read_text(
                encoding="utf-8"
            )
        )

        amendment = json.loads(
            amendment_source.read_text(
                encoding="utf-8"
            )
        )

    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(
            "Runtime protocol could not be parsed."
        ) from exc

    if base.get("phase") != "6F1":
        raise RuntimeManifestError(
            "Unexpected runtime base phase."
        )

    if amendment.get("phase") != "6F1.1":
        raise RuntimeManifestError(
            "Unexpected runtime amendment phase."
        )

    return EffectiveRuntimeProtocol(
        base=base,
        amendment=amendment,
        base_sha256=base_hash,
        amendment_sha256=amendment_hash,
        effective_sha256=effective_hash,
    )


def supported_runtime_statuses(
    protocol: EffectiveRuntimeProtocol,
) -> tuple[str, ...]:
    """Return complete frozen five-state runtime vocabulary."""

    statuses = tuple(
        str(value)
        for value
        in protocol.amendment[
            "effective_status_vocabulary"
        ]
    )

    if statuses != (
        "selected_method_accepted",
        "selected_method_accepted_grid_infeasible",
        "grid_fallback_selected_method_infeasible",
        "grid_fallback_selected_method_objective_regression",
        "no_feasible_point_found_under_search_protocol",
    ):
        raise RuntimeManifestError(
            "Unexpected effective runtime status vocabulary."
        )

    return statuses


def build_runtime_manifest_payload(
    *,
    runtime_implementation_source_sha256: str,
) -> dict[str, Any]:
    """Build canonical manifest payload for later Phase 6F3 freezing.

    This function does not write a file.
    """

    protocol = load_effective_runtime_protocol()

    base = protocol.base

    provenance = base[
        "provenance"
    ]

    initialization = base[
        "runtime_initialization"
    ]

    runtime_identity = base[
        "runtime_identity"
    ]

    return {
        "phase": "6F3",

        "artifact": (
            "production_optimisation_runtime_manifest"
        ),

        "schema_version": "1.0.0",

        "runtime_protocol_sha256": (
            protocol.base_sha256
        ),

        "runtime_protocol_amendment_sha256": (
            protocol.amendment_sha256
        ),

        "effective_runtime_protocol_sha256": (
            protocol.effective_sha256
        ),

        "runtime_implementation_source_sha256": (
            runtime_implementation_source_sha256
        ),

        "surrogate_manifest_sha256": (
            provenance[
                "surrogate_manifest_sha256"
            ]
        ),

        "density_model_sha256": (
            provenance[
                "density_model_sha256"
            ]
        ),

        "temperature_model_sha256": (
            provenance[
                "temperature_model_sha256"
            ]
        ),

        "optimizer_selection_sha256": (
            provenance[
                "optimizer_selection_sha256"
            ]
        ),

        "phase6e_result_sha256": (
            provenance[
                "phase6e_result_sha256"
            ]
        ),

        "continuous_benchmark_base_sha256": (
            provenance[
                "continuous_benchmark_base_sha256"
            ]
        ),

        "continuous_benchmark_amendment_sha256": (
            provenance[
                "continuous_benchmark_amendment_sha256"
            ]
        ),

        "continuous_benchmark_result_sha256": (
            provenance[
                "continuous_benchmark_result_sha256"
            ]
        ),

        "deterministic_grid_result_sha256": (
            provenance[
                "deterministic_grid_result_sha256"
            ]
        ),

        "production_seed": (
            runtime_identity[
                "production_seed"
            ]
        ),

        "selected_method": (
            runtime_identity[
                "selected_continuous_optimizer"
            ]
        ),

        "mandatory_reference_method": (
            runtime_identity[
                "mandatory_reference_method"
            ]
        ),

        "fallback_method": (
            runtime_identity[
                "fallback_method"
            ]
        ),

        "grid_input_array_sha256": (
            initialization[
                "expected_grid_input_array_sha256"
            ]
        ),

        "grid_density_array_sha256": (
            initialization[
                "expected_grid_density_array_sha256"
            ]
        ),

        "grid_temperature_array_sha256": (
            initialization[
                "expected_grid_temperature_array_sha256"
            ]
        ),

        "supported_status_vocabulary": list(
            supported_runtime_statuses(
                protocol
            )
        ),
    }


def _require_hash_match(
    payload: dict[str, Any],
    field: str,
    path: str | Path,
) -> None:
    expected = str(
        payload[field]
    )

    observed = file_sha256(
        path
    )

    if observed != expected:
        raise RuntimeManifestError(
            f"Manifest validation failed for {field!r}."
        )


def load_runtime_manifest(
    path: str | Path = DEFAULT_RUNTIME_MANIFEST_PATH,
) -> RuntimeManifest:
    """Load and fail-closed validate a runtime manifest."""

    protocol = load_effective_runtime_protocol()

    source = Path(path)

    try:
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(
            "Runtime manifest could not be loaded."
        ) from exc

    if payload.get("phase") != "6F3":
        raise RuntimeManifestError(
            "Unexpected runtime-manifest phase."
        )

    if (
        payload.get("artifact")
        != "production_optimisation_runtime_manifest"
    ):
        raise RuntimeManifestError(
            "Unexpected runtime-manifest artifact."
        )

    if payload.get("schema_version") != "1.0.0":
        raise RuntimeManifestError(
            "Unexpected runtime-manifest schema version."
        )

    if (
        payload.get(
            "runtime_protocol_sha256"
        )
        != protocol.base_sha256
    ):
        raise RuntimeManifestError(
            "Runtime base protocol hash mismatch in manifest."
        )

    if (
        payload.get(
            "runtime_protocol_amendment_sha256"
        )
        != protocol.amendment_sha256
    ):
        raise RuntimeManifestError(
            "Runtime amendment hash mismatch in manifest."
        )

    if (
        payload.get(
            "effective_runtime_protocol_sha256"
        )
        != protocol.effective_sha256
    ):
        raise RuntimeManifestError(
            "Effective runtime hash mismatch in manifest."
        )

    expected_statuses = list(
        supported_runtime_statuses(
            protocol
        )
    )

    if (
        payload.get(
            "supported_status_vocabulary"
        )
        != expected_statuses
    ):
        raise RuntimeManifestError(
            "Runtime status vocabulary mismatch."
        )

    expected_seed = int(
        protocol.base[
            "runtime_identity"
        ]["production_seed"]
    )

    if payload.get(
        "production_seed"
    ) != expected_seed:
        raise RuntimeManifestError(
            "Production seed mismatch in runtime manifest."
        )

    if payload.get(
        "selected_method"
    ) != "differential_evolution":
        raise RuntimeManifestError(
            "Selected method mismatch in runtime manifest."
        )

    if payload.get(
        "mandatory_reference_method"
    ) != "deterministic_grid":
        raise RuntimeManifestError(
            "Mandatory reference method mismatch in runtime manifest."
        )

    if payload.get(
        "fallback_method"
    ) != "deterministic_grid":
        raise RuntimeManifestError(
            "Fallback method mismatch in runtime manifest."
        )

    initialization = protocol.base[
        "runtime_initialization"
    ]

    expected_grid_hash_fields = {
        "grid_input_array_sha256": (
            "expected_grid_input_array_sha256"
        ),
        "grid_density_array_sha256": (
            "expected_grid_density_array_sha256"
        ),
        "grid_temperature_array_sha256": (
            "expected_grid_temperature_array_sha256"
        ),
    }

    for manifest_field, protocol_field in (
        expected_grid_hash_fields.items()
    ):
        if payload.get(
            manifest_field
        ) != initialization[
            protocol_field
        ]:
            raise RuntimeManifestError(
                f"Frozen grid-array hash mismatch for {manifest_field!r}."
            )

    _require_hash_match(
        payload,
        "surrogate_manifest_sha256",
        "artifacts/phase4/surrogate_manifest.json",
    )

    _require_hash_match(
        payload,
        "density_model_sha256",
        "artifacts/phase4/density_model.pkl",
    )

    _require_hash_match(
        payload,
        "temperature_model_sha256",
        "artifacts/phase4/temperature_model.pkl",
    )

    _require_hash_match(
        payload,
        "optimizer_selection_sha256",
        "configs/phase6/optimizer_selection.json",
    )

    _require_hash_match(
        payload,
        "phase6e_result_sha256",
        "results/phase6/robustness_qualification.json",
    )

    _require_hash_match(
        payload,
        "continuous_benchmark_base_sha256",
        "configs/phase6/continuous_optimizer_benchmark.json",
    )

    _require_hash_match(
        payload,
        "continuous_benchmark_amendment_sha256",
        "configs/phase6/continuous_optimizer_benchmark_amendment_001.json",
    )

    _require_hash_match(
        payload,
        "continuous_benchmark_result_sha256",
        "results/phase6/continuous_optimizer_benchmark.json",
    )

    _require_hash_match(
        payload,
        "deterministic_grid_result_sha256",
        "results/phase6/deterministic_grid_baseline.json",
    )

    runtime_source = Path(
        "src/plasma_ai/optimisation/runtime.py"
    )

    _require_hash_match(
        payload,
        "runtime_implementation_source_sha256",
        runtime_source,
    )

    return RuntimeManifest(
        payload=payload,
        path=source,
        sha256=file_sha256(
            source
        ),
    )
