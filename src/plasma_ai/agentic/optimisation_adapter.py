"""Trusted Phase 6 optimisation boundary for Phase 7."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from plasma_ai.agentic.errors import TrustedBoundaryError
from plasma_ai.optimisation.runtime import (
    load_phase6_optimisation_runtime,
    parse_runtime_request,
)


_BOUNDARY = "phase6_optimisation"

_RUNTIME_MANIFEST_PATH = (
    "artifacts/phase6/optimization_runtime_manifest.json"
)

_RUNTIME_MANIFEST_SHA256 = (
    "5f4d7446d5c5a504c793d38cecda854b5b36b9dd65eeba6dcfe5a0c64b8d7395"
)

_FROZEN_HASHES = {
    _RUNTIME_MANIFEST_PATH:
        _RUNTIME_MANIFEST_SHA256,
    "results/phase6/runtime_persistence_equivalence.json":
        "48600b44b196d4da5e1352722c65578770e802004df50158065f8945af62ce99",
    "src/plasma_ai/optimisation/runtime.py":
        "a391b196e0d73fe419d083e61d7bb92b87689e0adc02f3fed418b0a34cb3e082",
    "src/plasma_ai/optimisation/runtime_manifest.py":
        "80bbac4b1afa4a6357051048da7c0ca67939727f8572de0d87f8af43803aa931",
}

_STATUS_SOURCES = MappingProxyType(
    {
        "selected_method_accepted": "differential_evolution",
        "selected_method_accepted_grid_infeasible": (
            "differential_evolution"
        ),
        "grid_fallback_selected_method_infeasible": (
            "deterministic_grid"
        ),
        "grid_fallback_selected_method_objective_regression": (
            "deterministic_grid"
        ),
        "no_feasible_point_found_under_search_protocol": None,
    }
)

_REQUIRED_PROVENANCE_KEYS = frozenset(
    {
        "runtime_manifest_sha256",
        "effective_runtime_protocol_sha256",
        "phase4g_surrogate_manifest_sha256",
    }
)

_REQUIRED_RESPONSE_KEYS = frozenset(
    {
        "status",
        "scenario_id",
        "chosen_source",
        "chosen_operating_point",
        "chosen_predictions",
        "chosen_primary_objective",
        "selected_method_result",
        "grid_reference_result",
        "selected_minus_grid_primary_objective",
        "provenance",
    }
)


@dataclass(frozen=True)
class TrustedOptimisationEvidence:
    """Deeply immutable Phase 7-owned optimisation evidence."""

    runtime_response: Mapping[str, object]
    runtime_manifest_sha256: str


def _sha256_file(path: str) -> str:
    """Return SHA-256 for one required frozen file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _wrap_failure(stage: str, exc: Exception) -> TrustedBoundaryError:
    """Create the canonical Phase 7 optimisation-boundary failure."""

    return TrustedBoundaryError(
        boundary=_BOUNDARY,
        stage=stage,
        cause_type=type(exc).__name__,
    )


def _verify_frozen_identity() -> str:
    """Fail closed unless required Phase 6 identities are exact."""

    for path, expected in _FROZEN_HASHES.items():
        actual = _sha256_file(path)

        if actual != expected:
            raise ValueError(
                f"Frozen Phase 6 identity mismatch: {path}."
            )

    return _RUNTIME_MANIFEST_SHA256


def _copy_request(
    request: Mapping[str, object],
) -> dict[str, Any]:
    """Copy one request without rewriting its semantics."""

    if not isinstance(request, Mapping):
        raise TypeError("Optimisation request must be a mapping.")

    if not all(isinstance(key, str) for key in request):
        raise TypeError("Optimisation request keys must be strings.")

    return deepcopy(dict(request))


def _validate_provenance(
    provenance: object,
) -> Mapping[str, object]:
    """Validate mandatory frozen Phase 6 provenance."""

    if not isinstance(provenance, Mapping):
        raise TypeError("Runtime response provenance must be a mapping.")

    missing = _REQUIRED_PROVENANCE_KEYS - set(provenance.keys())

    if missing:
        raise ValueError(
            "Runtime response is missing required provenance: "
            f"{sorted(missing)}."
        )

    for key in _REQUIRED_PROVENANCE_KEYS:
        value = provenance[key]

        if not isinstance(value, str) or not value:
            raise TypeError(
                f"Runtime provenance {key!r} must be a non-empty string."
            )

    if provenance["runtime_manifest_sha256"] != _RUNTIME_MANIFEST_SHA256:
        raise ValueError(
            "Runtime response manifest provenance does not match the "
            "frozen Phase 6 runtime manifest."
        )

    return provenance


def _validate_response_contract(
    payload: Mapping[str, object],
) -> None:
    """Validate frozen Phase 6 response/status/provenance coherence."""

    missing = _REQUIRED_RESPONSE_KEYS - set(payload.keys())

    if missing:
        raise ValueError(
            "Runtime response is missing required fields: "
            f"{sorted(missing)}."
        )

    status = payload["status"]

    if not isinstance(status, str):
        raise TypeError("Runtime status must be a string.")

    if status not in _STATUS_SOURCES:
        raise ValueError("Runtime produced an unsupported status.")

    expected_source = _STATUS_SOURCES[status]

    if payload["chosen_source"] != expected_source:
        raise ValueError(
            "Runtime status and chosen_source are incoherent."
        )

    if status == "no_feasible_point_found_under_search_protocol":
        if payload["chosen_operating_point"] is not None:
            raise ValueError(
                "No-feasible runtime response must not contain a chosen "
                "operating point."
            )

        if payload["chosen_predictions"] is not None:
            raise ValueError(
                "No-feasible runtime response must not contain chosen "
                "predictions."
            )

        if payload["chosen_primary_objective"] is not None:
            raise ValueError(
                "No-feasible runtime response must not contain a chosen "
                "objective."
            )

    _validate_provenance(payload["provenance"])


def _deep_freeze(value: Any) -> object:
    """Recursively convert serialised runtime evidence to immutability."""

    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Runtime response mapping keys must be strings."
                )

            frozen[key] = _deep_freeze(item)

        return MappingProxyType(frozen)

    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Runtime response contains a non-finite float."
            )

        return value

    raise TypeError(
        "Runtime response contains a non-JSON immutable scalar type: "
        f"{type(value).__name__}."
    )


def _validated_frozen_response(
    response: Any,
) -> Mapping[str, object]:
    """Serialise, validate and deeply freeze one runtime response."""

    payload = response.to_dict()

    if not isinstance(payload, Mapping):
        raise TypeError("RuntimeResponse.to_dict() must return a mapping.")

    _validate_response_contract(payload)

    frozen = _deep_freeze(payload)

    if not isinstance(frozen, Mapping):
        raise TypeError("Frozen runtime response must remain a mapping.")

    return frozen


class TrustedOptimisationAdapter:
    """Narrow Phase 7 adapter over the frozen Phase 6 runtime."""

    __slots__ = ("_runtime", "_manifest_sha256")

    def __init__(self) -> None:
        """Verify frozen identity and eagerly load Phase 6 exactly once."""

        try:
            manifest_sha256 = _verify_frozen_identity()
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("integrity", exc) from exc

        try:
            runtime = load_phase6_optimisation_runtime()
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("load", exc) from exc

        self._runtime = runtime
        self._manifest_sha256 = manifest_sha256

    def evaluate(
        self,
        request: Mapping[str, object],
    ) -> TrustedOptimisationEvidence:
        """Return validated immutable evidence for one Phase 6 request."""

        try:
            raw_request = _copy_request(request)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("input_validation", exc) from exc

        try:
            scenario = parse_runtime_request(raw_request)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("parse", exc) from exc

        try:
            response = self._runtime.run(scenario)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("execution", exc) from exc

        try:
            frozen_response = _validated_frozen_response(response)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("output_validation", exc) from exc

        response_manifest_sha256 = frozen_response["provenance"][
            "runtime_manifest_sha256"
        ]

        if response_manifest_sha256 != self._manifest_sha256:
            exc = ValueError(
                "Evidence manifest SHA-256 does not match runtime response "
                "provenance."
            )
            raise _wrap_failure("output_validation", exc) from exc

        return TrustedOptimisationEvidence(
            runtime_response=frozen_response,
            runtime_manifest_sha256=self._manifest_sha256,
        )
