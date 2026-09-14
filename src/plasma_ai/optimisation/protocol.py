"""Frozen Phase 6 optimisation-contract loading.

This module combines the immutable Phase 6A base protocol with the
controlled Phase 6A.1 amendment.

It performs no surrogate loading, inference, optimisation, source-model
execution, or TEST-data access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_PROTOCOL_PATH = Path(
    "configs/phase6/optimisation_protocol.json"
)

DEFAULT_AMENDMENT_PATH = Path(
    "configs/phase6/optimisation_protocol_amendment_001.json"
)

EXPECTED_BASE_PROTOCOL_SHA256 = (
    "57fee45c69c5f42f9e94b786e895a5d89b163d141367326706eac2d3d2ed1cb1"
)

EXPECTED_AMENDMENT_001_SHA256 = (
    "9cf31578c6adddb195766fefa38ada89c6678e2df60bea708ddded0383c53912"
)

EXPECTED_EFFECTIVE_CONTRACT_SHA256 = (
    "91a6915a1773ca96364d555a4b2c9cd62ab8a692b6825f8859dc7331c90cf0b7"
)

EXPECTED_BASE_FREEZE_COMMIT = (
    "36e0f2a61c05bfece207449a23d2fee2d8e15e52"
)


class ProtocolIntegrityError(ValueError):
    """Raised when the frozen Phase 6 contract fails integrity checks."""


@dataclass(frozen=True)
class EffectivePhase6Protocol:
    """Validated Phase 6 base protocol plus controlled amendment."""

    base: dict[str, Any]
    amendment: dict[str, Any]
    base_sha256: str
    amendment_sha256: str
    effective_sha256: str


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 for one file."""

    source = Path(path)
    digest = hashlib.sha256()

    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def effective_contract_sha256(
    base_sha256: str,
    amendment_sha256: str,
) -> str:
    """Return the frozen effective-contract identity."""

    payload = (
        base_sha256
        + "\n"
        + amendment_sha256
        + "\n"
    ).encode("ascii")

    return hashlib.sha256(payload).hexdigest()


def _load_json_object(
    path: str | Path,
) -> dict[str, Any]:
    """Load one JSON object without mutating it."""

    source = Path(path)

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolIntegrityError(
            f"Could not load protocol JSON from {source}."
        ) from exc

    if not isinstance(payload, dict):
        raise ProtocolIntegrityError(
            f"Protocol JSON at {source} must be an object."
        )

    return payload


def load_effective_phase6_protocol(
    base_protocol_path: str | Path = DEFAULT_BASE_PROTOCOL_PATH,
    amendment_path: str | Path = DEFAULT_AMENDMENT_PATH,
) -> EffectivePhase6Protocol:
    """Load and validate the exact frozen Phase 6 effective contract."""

    base_path = Path(base_protocol_path)
    amendment_source = Path(amendment_path)

    try:
        observed_base_hash = file_sha256(base_path)
        observed_amendment_hash = file_sha256(
            amendment_source
        )
    except OSError as exc:
        raise ProtocolIntegrityError(
            "A frozen Phase 6 protocol file could not be read."
        ) from exc

    if observed_base_hash != EXPECTED_BASE_PROTOCOL_SHA256:
        raise ProtocolIntegrityError(
            "Frozen Phase 6A base protocol SHA-256 mismatch."
        )

    if (
        observed_amendment_hash
        != EXPECTED_AMENDMENT_001_SHA256
    ):
        raise ProtocolIntegrityError(
            "Frozen Phase 6A.1 amendment SHA-256 mismatch."
        )

    effective_hash = effective_contract_sha256(
        observed_base_hash,
        observed_amendment_hash,
    )

    if effective_hash != EXPECTED_EFFECTIVE_CONTRACT_SHA256:
        raise ProtocolIntegrityError(
            "Effective Phase 6 contract identity mismatch."
        )

    base = _load_json_object(base_path)
    amendment = _load_json_object(amendment_source)

    if base.get("phase") != "6A":
        raise ProtocolIntegrityError(
            "Unexpected Phase 6 base-protocol phase."
        )

    if base.get("protocol_schema_version") != "1.0.0":
        raise ProtocolIntegrityError(
            "Unsupported Phase 6 base-protocol schema."
        )

    if amendment.get("phase") != "6A.1":
        raise ProtocolIntegrityError(
            "Unexpected Phase 6 amendment phase."
        )

    if (
        amendment.get("amendment_id")
        != "phase6a_amendment_001"
    ):
        raise ProtocolIntegrityError(
            "Unexpected Phase 6 amendment identifier."
        )

    base_reference = amendment.get("base_protocol")

    if not isinstance(base_reference, dict):
        raise ProtocolIntegrityError(
            "Amendment base-protocol reference is invalid."
        )

    if (
        base_reference.get("sha256")
        != EXPECTED_BASE_PROTOCOL_SHA256
    ):
        raise ProtocolIntegrityError(
            "Amendment references the wrong base-protocol hash."
        )

    if (
        base_reference.get("freeze_commit")
        != EXPECTED_BASE_FREEZE_COMMIT
    ):
        raise ProtocolIntegrityError(
            "Amendment references the wrong base freeze commit."
        )

    if (
        base_reference.get(
            "base_protocol_must_remain_unmodified"
        )
        is not True
    ):
        raise ProtocolIntegrityError(
            "Amendment does not preserve base-protocol immutability."
        )

    return EffectivePhase6Protocol(
        base=base,
        amendment=amendment,
        base_sha256=observed_base_hash,
        amendment_sha256=observed_amendment_hash,
        effective_sha256=effective_hash,
    )
