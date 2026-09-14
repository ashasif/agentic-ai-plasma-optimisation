from __future__ import annotations

import json

import pytest

from plasma_ai.optimisation.protocol import (
    EXPECTED_AMENDMENT_001_SHA256,
    EXPECTED_BASE_PROTOCOL_SHA256,
    EXPECTED_EFFECTIVE_CONTRACT_SHA256,
    ProtocolIntegrityError,
    effective_contract_sha256,
    load_effective_phase6_protocol,
)


def test_load_effective_protocol_exact_hashes() -> None:
    protocol = load_effective_phase6_protocol()

    assert (
        protocol.base_sha256
        == EXPECTED_BASE_PROTOCOL_SHA256
    )

    assert (
        protocol.amendment_sha256
        == EXPECTED_AMENDMENT_001_SHA256
    )

    assert (
        protocol.effective_sha256
        == EXPECTED_EFFECTIVE_CONTRACT_SHA256
    )


def test_effective_identity_definition() -> None:
    assert (
        effective_contract_sha256(
            EXPECTED_BASE_PROTOCOL_SHA256,
            EXPECTED_AMENDMENT_001_SHA256,
        )
        == EXPECTED_EFFECTIVE_CONTRACT_SHA256
    )


def test_loaded_contract_has_expected_phases() -> None:
    protocol = load_effective_phase6_protocol()

    assert protocol.base["phase"] == "6A"
    assert protocol.amendment["phase"] == "6A.1"


def test_loaded_contract_preserves_test_separation() -> None:
    protocol = load_effective_phase6_protocol()

    assert (
        protocol.base["governance"][
            "phase4_test_may_not_be_reopened"
        ]
        is True
    )

    assert (
        protocol.base["governance"][
            "phase5_test_may_not_be_reopened"
        ]
        is True
    )


def test_modified_base_copy_is_rejected(
    tmp_path,
) -> None:
    protocol = load_effective_phase6_protocol()

    base_copy = tmp_path / "base.json"

    payload = dict(protocol.base)
    payload["phase"] = "tampered"

    base_copy.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ProtocolIntegrityError,
        match="SHA-256 mismatch",
    ):
        load_effective_phase6_protocol(
            base_protocol_path=base_copy,
        )


def test_modified_amendment_copy_is_rejected(
    tmp_path,
) -> None:
    protocol = load_effective_phase6_protocol()

    amendment_copy = tmp_path / "amendment.json"

    payload = dict(protocol.amendment)
    payload["phase"] = "tampered"

    amendment_copy.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ProtocolIntegrityError,
        match="SHA-256 mismatch",
    ):
        load_effective_phase6_protocol(
            amendment_path=amendment_copy,
        )
