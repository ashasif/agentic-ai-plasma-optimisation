"""Deterministic recording primitives for frozen Phase 7C orchestration."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from types import MappingProxyType


RECORD_SCHEMA_VERSION = 1
PHASE7C_PROTOCOL_SHA256 = "271b6620e7b86f34276a3735fbc248d6580a99f39115eb2f5732d9ed10744318"
PHASE5_MANIFEST_SHA256 = "a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259"
PHASE6_RUNTIME_MANIFEST_SHA256 = "5f4d7446d5c5a504c793d38cecda854b5b36b9dd65eeba6dcfe5a0c64b8d7395"

DECISION_MODES = frozenset(
    {
        "monitoring_driven",
        "operator_initiated_optimisation",
    }
)

REPLAY_EXCLUDED_FIELDS = frozenset(
    {
        "replay_fingerprint",
        "wall_clock_timestamp",
        "timing_metadata",
        "optional_natural_language_explanation",
        "approval_events",
    }
)

DECISION_ID_FIELDS = (
    "record_schema_version",
    "decision_mode",
    "phase7_protocol_sha256",
    "phase5_manifest_sha256",
    "phase6_runtime_manifest_sha256",
    "raw_monitoring_observation",
    "optimisation_request_or_null",
)


def to_json_native(value: object) -> object:
    """Recursively copy supported immutable evidence into JSON-native values."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}

        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON-native mapping keys must be exact strings.")

            result[key] = to_json_native(item)

        return result

    if isinstance(value, (tuple, list)):
        return [to_json_native(item) for item in value]

    if value is None:
        return None

    if type(value) is str:
        return value

    if type(value) is bool:
        return value

    if type(value) is int:
        return value

    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite floats are not permitted in authoritative records.")

        return value

    raise TypeError(
        f"Unsupported authoritative-record value type: {type(value).__name__}."
    )


def canonical_json(payload: object) -> str:
    """Return the exact frozen canonical JSON representation."""

    native = to_json_native(payload)

    return json.dumps(
        native,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(payload: object) -> str:
    """Hash canonical UTF-8 JSON without a trailing newline."""

    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decision_id_payload(
    *,
    decision_mode: str,
    raw_monitoring_observation: object,
    optimisation_request_or_null: object,
) -> dict[str, object]:
    """Build the exact frozen Phase 7 decision-id identity payload."""

    if decision_mode not in DECISION_MODES:
        raise ValueError(f"Unsupported Phase 7 decision mode: {decision_mode}.")

    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "decision_mode": decision_mode,
        "phase7_protocol_sha256": PHASE7C_PROTOCOL_SHA256,
        "phase5_manifest_sha256": PHASE5_MANIFEST_SHA256,
        "phase6_runtime_manifest_sha256": PHASE6_RUNTIME_MANIFEST_SHA256,
        "raw_monitoring_observation": to_json_native(raw_monitoring_observation),
        "optimisation_request_or_null": to_json_native(optimisation_request_or_null),
    }


def build_decision_id(
    *,
    decision_mode: str,
    raw_monitoring_observation: object,
    optimisation_request_or_null: object,
) -> str:
    """Return deterministic content-addressed identity for one decision unit."""

    payload = decision_id_payload(
        decision_mode=decision_mode,
        raw_monitoring_observation=raw_monitoring_observation,
        optimisation_request_or_null=optimisation_request_or_null,
    )

    return canonical_sha256(payload)


def replay_payload(authoritative_record: object) -> dict[str, object]:
    """Return the canonical replay payload with frozen exclusions removed."""

    native = to_json_native(authoritative_record)

    if not isinstance(native, dict):
        raise TypeError("Authoritative record must be a mapping.")

    return {
        key: value
        for key, value in native.items()
        if key not in REPLAY_EXCLUDED_FIELDS
    }


def build_replay_fingerprint(authoritative_record: object) -> str:
    """Hash authoritative replay content under the frozen exclusion policy."""

    return canonical_sha256(replay_payload(authoritative_record))


def _freeze_native(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _freeze_native(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(_freeze_native(item) for item in value)

    return value


def deep_freeze(payload: object) -> object:
    """Return a recursively immutable JSON-compatible representation."""

    return _freeze_native(to_json_native(payload))
