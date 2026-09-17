"""Unit tests for the Phase 7 trusted monitoring adapter."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import plasma_ai.agentic.monitoring_adapter as module
from plasma_ai.agentic.errors import TrustedBoundaryError
from plasma_ai.agentic.monitoring_adapter import (
    TrustedMonitoringAdapter,
    TrustedMonitoringEvidence,
)


PROJECT_ROOT = Path.cwd()


def _observation() -> dict[str, float]:
    return {
        "nominal_absorbed_power_W": 50.0,
        "target_pressure_mTorr": 30.0,
        "nominal_flow_sccm": 20.0,
        "measured_absorbed_power_W": 49.0,
        "measured_flow_sccm": 19.5,
        "measured_pressure_mTorr": 30.5,
    }


def _prediction(
    probability: float = 0.2,
    active: bool = False,
    state: str = "none",
):
    return SimpleNamespace(
        active_probability=np.asarray([probability]),
        fault_active=np.asarray([active], dtype=bool),
        diagnostic_state=np.asarray([state]),
    )


class FakeRuntime:
    def __init__(self, prediction=None, error: Exception | None = None):
        self.prediction = _prediction() if prediction is None else prediction
        self.error = error
        self.calls = 0
        self.raw_values = None

    def predict(self, raw_values):
        self.calls += 1
        self.raw_values = np.asarray(raw_values)

        if self.error is not None:
            raise self.error

        return self.prediction


def _adapter(monkeypatch, prediction=None) -> tuple[TrustedMonitoringAdapter, FakeRuntime]:
    runtime = FakeRuntime(prediction=prediction)
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return runtime

    monkeypatch.setattr(module, "load_phase5g_monitoring", loader)

    adapter = TrustedMonitoringAdapter()

    assert calls["count"] == 1

    return adapter, runtime


def _copy_integrity_files(destination: Path) -> None:
    for relative in module._FROZEN_HASHES:
        source = PROJECT_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_valid_single_observation(monkeypatch) -> None:
    adapter, runtime = _adapter(monkeypatch)

    evidence = adapter.evaluate(_observation())

    assert isinstance(evidence, TrustedMonitoringEvidence)
    assert evidence.active_probability == 0.2
    assert evidence.fault_active is False
    assert evidence.diagnostic_state == "none"
    assert evidence.monitoring_manifest_sha256 == module._MANIFEST_SHA256

    assert runtime.calls == 1
    assert runtime.raw_values.shape == (1, 6)
    assert runtime.raw_values.dtype == np.float64
    assert runtime.raw_values.tolist() == [[50.0, 30.0, 20.0, 49.0, 19.5, 30.5]]


def test_evidence_is_frozen(monkeypatch) -> None:
    adapter, _ = _adapter(monkeypatch)
    evidence = adapter.evaluate(_observation())

    with pytest.raises(FrozenInstanceError):
        evidence.diagnostic_state = "flow_delivery"


def test_loader_called_once_per_instance(monkeypatch) -> None:
    runtime = FakeRuntime()
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return runtime

    monkeypatch.setattr(module, "load_phase5g_monitoring", loader)

    adapter = TrustedMonitoringAdapter()
    adapter.evaluate(_observation())
    adapter.evaluate(_observation())

    assert calls["count"] == 1
    assert runtime.calls == 2


def test_missing_field(monkeypatch) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    del observation["measured_pressure_mTorr"]

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert runtime.calls == 0


def test_extra_field(monkeypatch) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    observation["unexpected"] = 1.0

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert runtime.calls == 0


def test_non_numeric_value(monkeypatch) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    observation["measured_flow_sccm"] = "bad"

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert captured.value.cause_type == "TypeError"
    assert runtime.calls == 0


def test_boolean_numeric_rejected(monkeypatch) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    observation["measured_flow_sccm"] = True

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert runtime.calls == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_value(monkeypatch, value: float) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    observation["measured_pressure_mTorr"] = value

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert runtime.calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
        "nominal_flow_sccm",
    ],
)
def test_non_positive_required_nominal_input(monkeypatch, field: str) -> None:
    adapter, runtime = _adapter(monkeypatch)
    observation = _observation()
    observation[field] = 0.0

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(observation)

    assert captured.value.stage == "input_validation"
    assert runtime.calls == 0


def test_wrong_prediction_cardinality(monkeypatch) -> None:
    prediction = SimpleNamespace(
        active_probability=np.asarray([0.2, 0.3]),
        fault_active=np.asarray([False, False]),
        diagnostic_state=np.asarray(["none", "none"]),
    )

    adapter, _ = _adapter(monkeypatch, prediction)

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_out_of_range(monkeypatch, probability: float) -> None:
    adapter, _ = _adapter(
        monkeypatch,
        _prediction(probability=probability),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


@pytest.mark.parametrize(
    ("probability", "active"),
    [
        (0.39, True),
        (0.40, False),
    ],
)
def test_threshold_incoherence(
    monkeypatch,
    probability: float,
    active: bool,
) -> None:
    state = "flow_delivery" if active else "none"

    adapter, _ = _adapter(
        monkeypatch,
        _prediction(probability, active, state),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


def test_inactive_state_incoherence(monkeypatch) -> None:
    adapter, _ = _adapter(
        monkeypatch,
        _prediction(0.2, False, "flow_delivery"),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


def test_active_state_incoherence(monkeypatch) -> None:
    adapter, _ = _adapter(
        monkeypatch,
        _prediction(0.8, True, "none"),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


def test_unknown_diagnostic_state(monkeypatch) -> None:
    adapter, _ = _adapter(
        monkeypatch,
        _prediction(0.8, True, "unknown_state"),
    )

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "output_validation"


def test_active_valid_state(monkeypatch) -> None:
    adapter, _ = _adapter(
        monkeypatch,
        _prediction(0.8, True, "power_coupling"),
    )

    evidence = adapter.evaluate(_observation())

    assert evidence.active_probability == 0.8
    assert evidence.fault_active is True
    assert evidence.diagnostic_state == "power_coupling"


def test_manifest_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    _copy_integrity_files(tmp_path)

    manifest = tmp_path / module._MANIFEST_PATH
    manifest.write_bytes(manifest.read_bytes() + b"tamper")

    monkeypatch.chdir(tmp_path)

    loader_calls = {"count": 0}

    def loader():
        loader_calls["count"] += 1
        return FakeRuntime()

    monkeypatch.setattr(module, "load_phase5g_monitoring", loader)

    with pytest.raises(TrustedBoundaryError) as captured:
        TrustedMonitoringAdapter()

    assert captured.value.stage == "integrity"
    assert captured.value.cause_type == "ValueError"
    assert loader_calls["count"] == 0


def test_loader_failure(monkeypatch) -> None:
    def loader():
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(module, "load_phase5g_monitoring", loader)

    with pytest.raises(TrustedBoundaryError) as captured:
        TrustedMonitoringAdapter()

    assert captured.value.stage == "load"
    assert captured.value.cause_type == "RuntimeError"
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_prediction_failure(monkeypatch) -> None:
    runtime = FakeRuntime(error=RuntimeError("synthetic prediction failure"))
    monkeypatch.setattr(
        module,
        "load_phase5g_monitoring",
        lambda: runtime,
    )

    adapter = TrustedMonitoringAdapter()

    with pytest.raises(TrustedBoundaryError) as captured:
        adapter.evaluate(_observation())

    assert captured.value.stage == "prediction"
    assert captured.value.cause_type == "RuntimeError"
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_input_mapping_is_not_mutated(monkeypatch) -> None:
    adapter, _ = _adapter(monkeypatch)
    observation = _observation()
    before = dict(observation)

    adapter.evaluate(observation)

    assert observation == before
