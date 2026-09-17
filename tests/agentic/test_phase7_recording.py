from types import MappingProxyType

import pytest

from plasma_ai.agentic.recording import (
    DECISION_ID_FIELDS,
    PHASE5_MANIFEST_SHA256,
    PHASE6_RUNTIME_MANIFEST_SHA256,
    PHASE7C_PROTOCOL_SHA256,
    RECORD_SCHEMA_VERSION,
    REPLAY_EXCLUDED_FIELDS,
    build_decision_id,
    build_replay_fingerprint,
    canonical_json,
    canonical_sha256,
    decision_id_payload,
    deep_freeze,
    replay_payload,
    to_json_native,
)


def test_frozen_identity_constants_are_exact():
    assert RECORD_SCHEMA_VERSION == 1
    assert PHASE7C_PROTOCOL_SHA256 == "271b6620e7b86f34276a3735fbc248d6580a99f39115eb2f5732d9ed10744318"
    assert PHASE5_MANIFEST_SHA256 == "a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259"
    assert PHASE6_RUNTIME_MANIFEST_SHA256 == "5f4d7446d5c5a504c793d38cecda854b5b36b9dd65eeba6dcfe5a0c64b8d7395"


def test_decision_id_field_contract_is_exact():
    assert DECISION_ID_FIELDS == (
        "record_schema_version",
        "decision_mode",
        "phase7_protocol_sha256",
        "phase5_manifest_sha256",
        "phase6_runtime_manifest_sha256",
        "raw_monitoring_observation",
        "optimisation_request_or_null",
    )


def test_replay_exclusion_contract_is_exact():
    assert REPLAY_EXCLUDED_FIELDS == frozenset(
        {
            "replay_fingerprint",
            "wall_clock_timestamp",
            "timing_metadata",
            "optional_natural_language_explanation",
            "approval_events",
        }
    )


def test_to_json_native_converts_mappingproxy_and_tuple_recursively():
    source = MappingProxyType(
        {
            "a": (1, 2),
            "nested": MappingProxyType({"x": True}),
        }
    )

    result = to_json_native(source)

    assert result == {"a": [1, 2], "nested": {"x": True}}
    assert type(result) is dict
    assert type(result["a"]) is list
    assert type(result["nested"]) is dict


def test_to_json_native_does_not_mutate_source():
    source = {"a": [1, {"b": 2}]}
    before = {"a": [1, {"b": 2}]}

    result = to_json_native(source)
    result["a"][1]["b"] = 99

    assert source == before


def test_to_json_native_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="mapping keys"):
        to_json_native({1: "value"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_to_json_native_rejects_nonfinite_floats(value):
    with pytest.raises(ValueError, match="Non-finite"):
        to_json_native({"value": value})


def test_to_json_native_rejects_unsupported_values():
    with pytest.raises(TypeError, match="Unsupported"):
        to_json_native({"value": object()})


def test_canonical_json_exact_formatting_and_ascii_policy():
    payload = {"z": 1, "a": "é"}

    assert canonical_json(payload) == "{\"a\":\"\\u00e9\",\"z\":1}"


def test_canonical_sha256_is_insertion_order_independent():
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}

    assert canonical_sha256(left) == canonical_sha256(right)


def test_deep_freeze_returns_recursive_immutable_structure():
    frozen = deep_freeze({"nested": {"x": [1, 2]}})

    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["nested"], MappingProxyType)
    assert frozen["nested"]["x"] == (1, 2)

    with pytest.raises(TypeError):
        frozen["new"] = 1

    with pytest.raises(TypeError):
        frozen["nested"]["x"] = (3,)


def test_decision_id_payload_contains_exact_frozen_fields():
    payload = decision_id_payload(
        decision_mode="monitoring_driven",
        raw_monitoring_observation={"x": 1},
        optimisation_request_or_null=None,
    )

    assert tuple(payload.keys()) == DECISION_ID_FIELDS
    assert payload["record_schema_version"] == 1
    assert payload["phase7_protocol_sha256"] == PHASE7C_PROTOCOL_SHA256
    assert payload["phase5_manifest_sha256"] == PHASE5_MANIFEST_SHA256
    assert payload["phase6_runtime_manifest_sha256"] == PHASE6_RUNTIME_MANIFEST_SHA256


def test_decision_id_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported Phase 7 decision mode"):
        build_decision_id(
            decision_mode="unknown",
            raw_monitoring_observation={"x": 1},
            optimisation_request_or_null=None,
        )


def test_decision_id_is_deterministic_and_order_independent():
    left = build_decision_id(
        decision_mode="operator_initiated_optimisation",
        raw_monitoring_observation={"b": 2, "a": 1},
        optimisation_request_or_null={"objective": "electron_density"},
    )

    right = build_decision_id(
        decision_mode="operator_initiated_optimisation",
        raw_monitoring_observation={"a": 1, "b": 2},
        optimisation_request_or_null={"objective": "electron_density"},
    )

    assert left == right
    assert len(left) == 64
    int(left, 16)


def test_decision_id_changes_when_authoritative_input_changes():
    baseline = build_decision_id(
        decision_mode="monitoring_driven",
        raw_monitoring_observation={"x": 1},
        optimisation_request_or_null=None,
    )

    changed = build_decision_id(
        decision_mode="monitoring_driven",
        raw_monitoring_observation={"x": 2},
        optimisation_request_or_null=None,
    )

    assert baseline != changed


def test_replay_payload_removes_exact_excluded_top_level_fields():
    record = {
        "decision_id": "abc",
        "decision_outcome": "no_action",
        "replay_fingerprint": "old",
        "wall_clock_timestamp": "now",
        "timing_metadata": {"ms": 3},
        "optional_natural_language_explanation": "text",
        "approval_events": [{"approval_state": "approved"}],
    }

    result = replay_payload(record)

    assert result == {
        "decision_id": "abc",
        "decision_outcome": "no_action",
    }


def test_replay_fingerprint_ignores_excluded_fields():
    left = {
        "decision_id": "abc",
        "decision_outcome": "no_action",
        "wall_clock_timestamp": "t1",
        "optional_natural_language_explanation": "one",
    }

    right = {
        "decision_id": "abc",
        "decision_outcome": "no_action",
        "wall_clock_timestamp": "t2",
        "optional_natural_language_explanation": "two",
    }

    assert build_replay_fingerprint(left) == build_replay_fingerprint(right)


def test_replay_fingerprint_changes_for_authoritative_content():
    left = {
        "decision_id": "abc",
        "decision_outcome": "no_action",
    }

    right = {
        "decision_id": "abc",
        "decision_outcome": "request_human_input",
    }

    assert build_replay_fingerprint(left) != build_replay_fingerprint(right)
