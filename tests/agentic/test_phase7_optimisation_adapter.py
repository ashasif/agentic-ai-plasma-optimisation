"""Unit tests for the Phase 7 trusted optimisation adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil
from types import MappingProxyType

import pytest

import plasma_ai.agentic.optimisation_adapter as module
from plasma_ai.agentic.errors import TrustedBoundaryError
from plasma_ai.agentic.optimisation_adapter import (
    TrustedOptimisationAdapter,
    TrustedOptimisationEvidence,
)


PROJECT_ROOT = Path.cwd()


def _request() -> dict[str, object]:
    return {
        "scenario_id": "phase7b-test",
        "objective_priority": ["electron_density_m3"],
        "constraints": {
            "electron_temperature_eV": {
                "maximum": 5.0,
            }
        },
    }


def _payload(
    status: str = "selected_method_accepted",
    chosen_source: str | None = "differential_evolution",
) -> dict[str, object]:
    no_feasible = (
        status == "no_feasible_point_found_under_search_protocol"
    )

    return {
        "status": status,
        "scenario_id": "phase7b-test",
        "chosen_source": chosen_source,
        "chosen_operating_point": (
            None
            if no_feasible
            else {
                "nominal_absorbed_power_W": 50.0,
                "target_pressure_mTorr": 30.0,
            }
        ),
        "chosen_predictions": (
            None
            if no_feasible
            else {
                "electron_density_m3": 1.0e17,
                "electron_temperature_eV": 3.0,
            }
        ),
        "chosen_primary_objective": (
            None if no_feasible else 1.0e17
        ),
        "selected_method_result": {
            "run_pass": not no_feasible,
            "final_phase6_feasible": not no_feasible,
            "constraint_values": {
                "temperature_max": 3.0,
            },
        },
        "grid_reference_result": {
            "feasible_candidate_count": 5,
            "scenario_feasible": not no_feasible,
        },
        "selected_minus_grid_primary_objective": (
            None if no_feasible else 0.0
        ),
        "provenance": {
            "runtime_manifest_sha256": module._RUNTIME_MANIFEST_SHA256,
            "effective_runtime_protocol_sha256": "a" * 64,
            "phase4g_surrogate_manifest_sha256": "b" * 64,
        },
    }


class FakeResponse:
    def __init__(
        self,
        payload: object,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.to_dict_calls = 0

    def to_dict(self):
        self.to_dict_calls += 1

        if self.error is not None:
            raise self.error

        return self.payload


class FakeRuntime:
    def __init__(
        self,
        response: FakeResponse,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.scenarios = []

    def run(self, scenario):
        self.calls += 1
        self.scenarios.append(scenario)

        if self.error is not None:
            raise self.error

        return self.response


def _install_doubles(
    monkeypatch,
    payload: object | None = None,
):
    if payload is None:
        payload = _payload()

    response = FakeResponse(payload)
    runtime = FakeRuntime(response)
    parsed = object()

    calls = {
        "loader": 0,
        "parser": 0,
        "parser_inputs": [],
    }

    def loader():
        calls["loader"] += 1
        return runtime

    def parser(raw):
        calls["parser"] += 1
        calls["parser_inputs"].append(raw)
        return parsed

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        loader,
    )

    monkeypatch.setattr(
        module,
        "parse_runtime_request",
        parser,
    )

    adapter = TrustedOptimisationAdapter()

    assert calls["loader"] == 1

    return adapter, runtime, response, parsed, calls


def _copy_integrity_files(destination: Path) -> None:
    for relative in module._FROZEN_HASHES:
        source = PROJECT_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_thaw(item) for item in value]

    return value


@pytest.mark.parametrize(
    ("status", "source"),
    [
        ("selected_method_accepted", "differential_evolution"),
        (
            "selected_method_accepted_grid_infeasible",
            "differential_evolution",
        ),
        (
            "grid_fallback_selected_method_infeasible",
            "deterministic_grid",
        ),
        (
            "grid_fallback_selected_method_objective_regression",
            "deterministic_grid",
        ),
        (
            "no_feasible_point_found_under_search_protocol",
            None,
        ),
    ],
)
def test_all_five_valid_status_source_contracts(
    monkeypatch,
    status: str,
    source: str | None,
) -> None:
    payload = _payload(status, source)
    adapter, runtime, response, parsed, calls = _install_doubles(
        monkeypatch,
        payload,
    )

    evidence = adapter.evaluate(_request())

    assert isinstance(evidence, TrustedOptimisationEvidence)
    assert evidence.runtime_response["status"] == status
    assert evidence.runtime_response["chosen_source"] == source
    assert evidence.runtime_manifest_sha256 == module._RUNTIME_MANIFEST_SHA256
    assert calls["parser"] == 1
    assert runtime.calls == 1
    assert runtime.scenarios == [parsed]
    assert response.to_dict_calls == 1


def test_evidence_dataclass_is_frozen(monkeypatch) -> None:
    adapter, _, _, _, _ = _install_doubles(monkeypatch)
    evidence = adapter.evaluate(_request())

    with pytest.raises(FrozenInstanceError):
        evidence.runtime_manifest_sha256 = "changed"


def test_runtime_response_is_deeply_immutable(monkeypatch) -> None:
    payload = _payload()
    payload["extra"] = {
        "nested": [1, 2, {"value": 3}],
    }

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)
    evidence = adapter.evaluate(_request())

    assert isinstance(evidence.runtime_response, MappingProxyType)
    assert isinstance(evidence.runtime_response["extra"], MappingProxyType)
    assert evidence.runtime_response["extra"]["nested"] == (
        1,
        2,
        MappingProxyType({"value": 3}),
    )

    with pytest.raises(TypeError):
        evidence.runtime_response["status"] = "changed"

    with pytest.raises(TypeError):
        evidence.runtime_response["extra"]["changed"] = 1


def test_complete_runtime_response_preserved(monkeypatch) -> None:
    payload = _payload()
    payload["extra_preserved_field"] = {
        "values": [1, 2, 3],
        "label": "preserved",
    }

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)
    evidence = adapter.evaluate(_request())

    assert _thaw(evidence.runtime_response) == {
        **payload,
        "extra_preserved_field": {
            "values": [1, 2, 3],
            "label": "preserved",
        },
    }


def test_loader_called_once_per_instance(monkeypatch) -> None:
    adapter, runtime, _, _, calls = _install_doubles(monkeypatch)

    adapter.evaluate(_request())
    adapter.evaluate(_request())

    assert calls["loader"] == 1
    assert calls["parser"] == 2
    assert runtime.calls == 2


def test_one_parse_and_one_run_per_adapter_call(monkeypatch) -> None:
    adapter, runtime, response, _, calls = _install_doubles(monkeypatch)

    adapter.evaluate(_request())

    assert calls["parser"] == 1
    assert runtime.calls == 1
    assert response.to_dict_calls == 1


def test_input_request_is_not_mutated(monkeypatch) -> None:
    runtime = FakeRuntime(FakeResponse(_payload()))
    original = _request()
    before = {
        "scenario_id": original["scenario_id"],
        "objective_priority": list(original["objective_priority"]),
        "constraints": {
            "electron_temperature_eV": {
                "maximum": original["constraints"]["electron_temperature_eV"]["maximum"],
            }
        },
    }

    def parser(raw):
        raw["scenario_id"] = "mutated-copy"
        raw["objective_priority"].append("mutation")
        raw["constraints"]["electron_temperature_eV"]["maximum"] = 999.0
        return object()

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(module, "parse_runtime_request", parser)

    adapter = TrustedOptimisationAdapter()
    adapter.evaluate(original)

    assert original == before


def test_non_mapping_request_rejected(monkeypatch) -> None:
    adapter, runtime, _, _, calls = _install_doubles(monkeypatch)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(["not", "a", "mapping"])

    assert captured.value.stage == "input_validation"
    assert captured.value.cause_type == "TypeError"
    assert calls["parser"] == 0
    assert runtime.calls == 0


def test_non_string_request_key_rejected(monkeypatch) -> None:
    adapter, runtime, _, _, calls = _install_doubles(monkeypatch)

    request = _request()
    request[7] = "invalid"

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(request)

    assert captured.value.stage == "input_validation"
    assert calls["parser"] == 0
    assert runtime.calls == 0


def test_manifest_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    _copy_integrity_files(tmp_path)

    manifest = tmp_path / module._RUNTIME_MANIFEST_PATH
    manifest.write_bytes(manifest.read_bytes() + b"tamper")

    monkeypatch.chdir(tmp_path)

    calls = {"loader": 0}

    def loader():
        calls["loader"] += 1
        return FakeRuntime(FakeResponse(_payload()))

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        loader,
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        TrustedOptimisationAdapter()

    assert captured.value.stage == "integrity"
    assert captured.value.cause_type == "ValueError"
    assert calls["loader"] == 0


def test_loader_failure(monkeypatch) -> None:
    def loader():
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        loader,
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        TrustedOptimisationAdapter()

    assert captured.value.stage == "load"
    assert captured.value.cause_type == "RuntimeError"
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_parse_failure(monkeypatch) -> None:
    runtime = FakeRuntime(FakeResponse(_payload()))

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        lambda: runtime,
    )

    def parser(raw):
        raise ValueError("synthetic parse failure")

    monkeypatch.setattr(module, "parse_runtime_request", parser)

    adapter = TrustedOptimisationAdapter()

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "parse"
    assert captured.value.cause_type == "ValueError"
    assert isinstance(captured.value.__cause__, ValueError)
    assert runtime.calls == 0


def test_execution_failure(monkeypatch) -> None:
    runtime = FakeRuntime(
        FakeResponse(_payload()),
        error=RuntimeError("synthetic execution failure"),
    )

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        module,
        "parse_runtime_request",
        lambda raw: object(),
    )

    adapter = TrustedOptimisationAdapter()

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "execution"
    assert captured.value.cause_type == "RuntimeError"
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert runtime.calls == 1


def test_response_serialisation_failure(monkeypatch) -> None:
    response = FakeResponse(
        _payload(),
        error=RuntimeError("synthetic to_dict failure"),
    )
    runtime = FakeRuntime(response)

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        module,
        "parse_runtime_request",
        lambda raw: object(),
    )

    adapter = TrustedOptimisationAdapter()

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"
    assert captured.value.cause_type == "RuntimeError"


def test_unknown_status(monkeypatch) -> None:
    adapter, _, _, _, _ = _install_doubles(
        monkeypatch,
        _payload("unknown_status", None),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize(
    ("status", "wrong_source"),
    [
        ("selected_method_accepted", "deterministic_grid"),
        (
            "selected_method_accepted_grid_infeasible",
            "deterministic_grid",
        ),
        (
            "grid_fallback_selected_method_infeasible",
            "differential_evolution",
        ),
        (
            "grid_fallback_selected_method_objective_regression",
            "differential_evolution",
        ),
        (
            "no_feasible_point_found_under_search_protocol",
            "deterministic_grid",
        ),
    ],
)
def test_status_source_incoherence(
    monkeypatch,
    status: str,
    wrong_source: str,
) -> None:
    adapter, _, _, _, _ = _install_doubles(
        monkeypatch,
        _payload(status, wrong_source),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "chosen_operating_point",
            {
                "nominal_absorbed_power_W": 50.0,
                "target_pressure_mTorr": 30.0,
            },
        ),
        (
            "chosen_predictions",
            {
                "electron_density_m3": 1.0e17,
                "electron_temperature_eV": 3.0,
            },
        ),
        ("chosen_primary_objective", 1.0),
    ],
)
def test_no_feasible_null_contract(
    monkeypatch,
    field: str,
    value: object,
) -> None:
    payload = _payload(
        "no_feasible_point_found_under_search_protocol",
        None,
    )
    payload[field] = value

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_missing_required_response_field(monkeypatch, field: str) -> None:
    payload = _payload()
    del payload[field]

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize(
    "key",
    [
        "runtime_manifest_sha256",
        "effective_runtime_protocol_sha256",
        "phase4g_surrogate_manifest_sha256",
    ],
)
def test_missing_required_provenance(monkeypatch, key: str) -> None:
    payload = _payload()
    del payload["provenance"][key]

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


def test_runtime_manifest_provenance_mismatch(monkeypatch) -> None:
    payload = _payload()
    payload["provenance"]["runtime_manifest_sha256"] = "0" * 64

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


def test_evidence_manifest_matches_response_provenance(monkeypatch) -> None:
    adapter, _, _, _, _ = _install_doubles(monkeypatch)
    evidence = adapter.evaluate(_request())

    assert evidence.runtime_manifest_sha256 == (
        evidence.runtime_response["provenance"]["runtime_manifest_sha256"]
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_runtime_response_rejected(
    monkeypatch,
    value: float,
) -> None:
    payload = _payload()
    payload["extra_nonfinite"] = {"value": value}

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"


def test_non_json_runtime_object_rejected(monkeypatch) -> None:
    payload = _payload()
    payload["unsupported"] = object()

    adapter, _, _, _, _ = _install_doubles(monkeypatch, payload)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"
    assert captured.value.cause_type == "TypeError"


def test_to_dict_must_return_mapping(monkeypatch) -> None:
    adapter, _, _, _, _ = _install_doubles(
        monkeypatch,
        ["not", "a", "mapping"],
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_request())

    assert captured.value.stage == "output_validation"
    assert captured.value.cause_type == "TypeError"


def test_keyboard_interrupt_from_parser_is_not_caught(monkeypatch) -> None:
    runtime = FakeRuntime(FakeResponse(_payload()))

    monkeypatch.setattr(
        module,
        "load_phase6_optimisation_runtime",
        lambda: runtime,
    )

    def parser(raw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "parse_runtime_request", parser)

    adapter = TrustedOptimisationAdapter()

    with pytest.raises(KeyboardInterrupt):
        adapter.evaluate(_request())
