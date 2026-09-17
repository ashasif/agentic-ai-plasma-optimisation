"""Trusted Phase 5 monitoring boundary for Phase 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from plasma_ai.agentic.errors import TrustedBoundaryError
from plasma_ai.monitoring.phase5g_persistence import load_phase5g_monitoring


_BOUNDARY = "phase5_monitoring"

_FEATURE_ORDER = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
    "nominal_flow_sccm",
    "measured_absorbed_power_W",
    "measured_flow_sccm",
    "measured_pressure_mTorr",
)

_STRICTLY_POSITIVE_FIELDS = frozenset(
    {
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
        "nominal_flow_sccm",
    }
)

_ACTIVE_STATES = frozenset(
    {
        "flow_delivery",
        "power_coupling",
        "pressure_path_anomaly",
    }
)

_ALL_STATES = frozenset({"none"}) | _ACTIVE_STATES
_THRESHOLD = 0.4

_FROZEN_HASHES = {
    "configs/phase5/phase5g_protocol.json":
        "2ca15b486bf15c3c439e19607812358f632151c94de15541ae6d27e04fe1574c",
    "artifacts/phase5/monitoring_manifest.json":
        "a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259",
    "artifacts/phase5/fault_detector.pkl":
        "4a00e785d7e7f0a8429020399967cdd82c18da275d58e5b83bc11707eeac8a6a",
    "artifacts/phase5/fault_diagnoser.pkl":
        "2e9dea016137594c707ecd3c32116691cc90d0127133f11c70372c93e6fcca84",
    "results/phase5/persistence_equivalence.json":
        "6f53ac1c9b919172bde8a2030ecbedaabab94f0aa5dbac3e6cd68f534cb1feb9",
}

_MANIFEST_PATH = "artifacts/phase5/monitoring_manifest.json"
_MANIFEST_SHA256 = _FROZEN_HASHES[_MANIFEST_PATH]


@dataclass(frozen=True)
class TrustedMonitoringEvidence:
    """Immutable Phase 7-owned monitoring evidence."""

    active_probability: float
    fault_active: bool
    diagnostic_state: str
    monitoring_manifest_sha256: str


def _sha256_file(path: str) -> str:
    """Return SHA-256 for one required frozen file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _wrap_failure(stage: str, exc: Exception) -> TrustedBoundaryError:
    """Create the canonical Phase 7 trusted-boundary failure."""

    return TrustedBoundaryError(
        boundary=_BOUNDARY,
        stage=stage,
        cause_type=type(exc).__name__,
    )


def _verify_frozen_identity() -> str:
    """Fail closed unless all required Phase 5 identities are exact."""

    for path, expected in _FROZEN_HASHES.items():
        actual = _sha256_file(path)

        if actual != expected:
            raise ValueError(
                f"Frozen Phase 5 identity mismatch: {path}."
            )

    return _MANIFEST_SHA256


def _validate_observation(
    observation: Mapping[str, object],
) -> np.ndarray:
    """Validate and encode exactly one authoritative observation."""

    if not isinstance(observation, Mapping):
        raise TypeError("Monitoring observation must be a mapping.")

    keys = set(observation.keys())
    expected = set(_FEATURE_ORDER)

    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            "Monitoring observation fields do not match the frozen contract; "
            f"missing={missing}; extra={extra}."
        )

    values: list[float] = []

    for name in _FEATURE_ORDER:
        value = observation[name]

        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(
                f"Monitoring value {name!r} must be a real numeric scalar."
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                f"Monitoring value {name!r} must be finite."
            )

        if name in _STRICTLY_POSITIVE_FIELDS and numeric <= 0.0:
            raise ValueError(
                f"Monitoring value {name!r} must be strictly positive."
            )

        values.append(numeric)

    matrix = np.asarray([values], dtype=np.float64)

    if matrix.shape != (1, 6):
        raise ValueError("Monitoring row did not encode to shape (1, 6).")

    matrix.setflags(write=False)

    return matrix


def _single_prediction_value(values: Any, field: str) -> Any:
    """Extract exactly one scalar from one upstream prediction field."""

    array = np.asarray(values)

    if array.shape != (1,):
        raise ValueError(
            f"Prediction field {field!r} must have shape (1,)."
        )

    return array[0]


def _validate_prediction(
    prediction: Any,
    manifest_sha256: str,
) -> TrustedMonitoringEvidence:
    """Validate one Phase 5 prediction before crossing the boundary."""

    probability_raw = _single_prediction_value(
        prediction.active_probability,
        "active_probability",
    )

    active_raw = _single_prediction_value(
        prediction.fault_active,
        "fault_active",
    )

    state_raw = _single_prediction_value(
        prediction.diagnostic_state,
        "diagnostic_state",
    )

    if isinstance(probability_raw, (bool, np.bool_)) or not isinstance(
        probability_raw,
        Real,
    ):
        raise TypeError("active_probability must be numeric.")

    probability = float(probability_raw)

    if not math.isfinite(probability):
        raise ValueError("active_probability must be finite.")

    if probability < 0.0 or probability > 1.0:
        raise ValueError("active_probability must be within [0, 1].")

    if not isinstance(active_raw, (bool, np.bool_)):
        raise TypeError("fault_active must be boolean.")

    fault_active = bool(active_raw)
    state = str(state_raw)

    expected_fault_active = probability >= _THRESHOLD

    if fault_active != expected_fault_active:
        raise ValueError(
            "fault_active does not match the frozen 0.4 threshold rule."
        )

    if state not in _ALL_STATES:
        raise ValueError("Unknown Phase 5 diagnostic state.")

    if not fault_active and state != "none":
        raise ValueError(
            "Inactive monitoring evidence must use diagnostic state none."
        )

    if fault_active and state not in _ACTIVE_STATES:
        raise ValueError(
            "Active monitoring evidence must use an active diagnostic state."
        )

    return TrustedMonitoringEvidence(
        active_probability=probability,
        fault_active=fault_active,
        diagnostic_state=state,
        monitoring_manifest_sha256=manifest_sha256,
    )


class TrustedMonitoringAdapter:
    """Narrow Phase 7 adapter over the frozen Phase 5G runtime."""

    __slots__ = ("_runtime", "_manifest_sha256")

    def __init__(self) -> None:
        """Verify frozen identity and eagerly load Phase 5G exactly once."""

        try:
            manifest_sha256 = _verify_frozen_identity()
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("integrity", exc) from exc

        try:
            runtime = load_phase5g_monitoring()
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("load", exc) from exc

        self._runtime = runtime
        self._manifest_sha256 = manifest_sha256

    def evaluate(
        self,
        observation: Mapping[str, object],
    ) -> TrustedMonitoringEvidence:
        """Return validated immutable evidence for one observation."""

        try:
            raw_row = _validate_observation(observation)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("input_validation", exc) from exc

        try:
            prediction = self._runtime.predict(raw_row)
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("prediction", exc) from exc

        try:
            return _validate_prediction(
                prediction,
                self._manifest_sha256,
            )
        except TrustedBoundaryError:
            raise
        except Exception as exc:
            raise _wrap_failure("output_validation", exc) from exc
