from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import plasma_ai.optimisation.surrogate_adapter as adapter_module

from plasma_ai.optimisation.primitives import (
    DecisionPointError,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
)
from plasma_ai.optimisation.surrogate_adapter import (
    Phase6SurrogateAdapter,
    PredictionContractError,
    load_phase6_surrogate_adapter,
)


class FakeRuntime:
    def __init__(
        self,
        *,
        density=None,
        temperature=None,
    ) -> None:
        self.calls: list[np.ndarray] = []
        self._density = density
        self._temperature = temperature

    def predict_physical(
        self,
        X,
    ):
        matrix = np.asarray(
            X,
            dtype=float,
        )

        self.calls.append(
            matrix.copy()
        )

        if self._density is None:
            density = (
                1.0e17
                + matrix[:, 0] * 1.0e14
                + matrix[:, 1] * 1.0e13
            )
        else:
            density = self._density

        if self._temperature is None:
            temperature = (
                1.5
                + matrix[:, 0] * 1.0e-3
                + matrix[:, 1] * 1.0e-4
            )
        else:
            temperature = self._temperature

        return SimpleNamespace(
            electron_density_m3=density,
            electron_temperature_eV=temperature,
        )


def _loaded_adapter(
    monkeypatch,
    *,
    runtime=None,
):
    fake = runtime or FakeRuntime()
    seen = {}

    def fake_loader(
        manifest_path,
        **kwargs,
    ):
        seen["manifest_path"] = manifest_path
        seen["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(
        adapter_module,
        "load_phase4g_surrogate",
        fake_loader,
    )

    adapter = load_phase6_surrogate_adapter()

    return adapter, fake, seen


def test_loader_uses_trusted_phase4g_manifest(
    monkeypatch,
) -> None:
    adapter, _, seen = _loaded_adapter(
        monkeypatch
    )

    assert isinstance(
        adapter,
        Phase6SurrogateAdapter,
    )

    assert (
        seen["manifest_path"].as_posix()
        == "artifacts/phase4/surrogate_manifest.json"
    )

    assert seen["kwargs"] == {}


def test_feature_order_is_frozen(
    monkeypatch,
) -> None:
    adapter, _, _ = _loaded_adapter(
        monkeypatch
    )

    assert adapter.feature_order == (
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
    )


def test_batch_prediction_preserves_feature_order(
    monkeypatch,
) -> None:
    adapter, runtime, _ = _loaded_adapter(
        monkeypatch
    )

    result = adapter.predict_batch(
        [
            [15.0, 10.0],
            [90.0, 60.0],
        ]
    )

    assert len(runtime.calls) == 1

    np.testing.assert_array_equal(
        runtime.calls[0],
        np.array(
            [
                [15.0, 10.0],
                [90.0, 60.0],
            ],
            dtype=float,
        ),
    )

    assert result.input_matrix.shape == (2, 2)
    assert result.electron_density_m3.shape == (2,)
    assert result.electron_temperature_eV.shape == (2,)


def test_point_prediction_uses_one_row(
    monkeypatch,
) -> None:
    adapter, runtime, _ = _loaded_adapter(
        monkeypatch
    )

    result = adapter.predict_point(
        OperatingPoint(
            nominal_absorbed_power_W=52.5,
            target_pressure_mTorr=35.0,
        )
    )

    assert len(runtime.calls) == 1
    assert runtime.calls[0].shape == (1, 2)

    assert result.operating_point == OperatingPoint(
        52.5,
        35.0,
    )

    assert result.electron_density_m3 > 0.0
    assert result.electron_temperature_eV > 0.0


@pytest.mark.parametrize(
    "values",
    [
        [[14.9, 35.0]],
        [[90.1, 35.0]],
        [[52.5, 9.9]],
        [[52.5, 60.1]],
    ],
)
def test_out_of_domain_input_is_rejected_before_inference(
    monkeypatch,
    values,
) -> None:
    adapter, runtime, _ = _loaded_adapter(
        monkeypatch
    )

    with pytest.raises(
        DecisionPointError,
        match="outside",
    ):
        adapter.predict_batch(values)

    assert runtime.calls == []


def test_boolean_input_is_rejected_before_inference(
    monkeypatch,
) -> None:
    adapter, runtime, _ = _loaded_adapter(
        monkeypatch
    )

    with pytest.raises(
        DecisionPointError,
        match="real number",
    ):
        adapter.predict_batch(
            [
                [True, 35.0],
            ]
        )

    assert runtime.calls == []


@pytest.mark.parametrize(
    "values",
    [
        [52.5, 35.0],
        [[[52.5, 35.0]]],
        [[52.5]],
        [[52.5, 35.0, 1.0]],
        [],
    ],
)
def test_invalid_batch_shape_is_rejected(
    monkeypatch,
    values,
) -> None:
    adapter, runtime, _ = _loaded_adapter(
        monkeypatch
    )

    with pytest.raises(
        PredictionContractError,
    ):
        adapter.predict_batch(values)

    assert runtime.calls == []


def test_nonfinite_prediction_is_rejected(
    monkeypatch,
) -> None:
    runtime = FakeRuntime(
        density=np.array([np.nan]),
        temperature=np.array([1.8]),
    )

    adapter, runtime, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    with pytest.raises(
        PredictionContractError,
        match="non-finite",
    ):
        adapter.predict_batch(
            [
                [52.5, 35.0],
            ]
        )

    assert len(runtime.calls) == 1


def test_nonpositive_density_prediction_is_rejected(
    monkeypatch,
) -> None:
    runtime = FakeRuntime(
        density=np.array([0.0]),
        temperature=np.array([1.8]),
    )

    adapter, _, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    with pytest.raises(
        PredictionContractError,
        match="non-positive",
    ):
        adapter.predict_batch(
            [
                [52.5, 35.0],
            ]
        )


def test_nonpositive_temperature_prediction_is_rejected(
    monkeypatch,
) -> None:
    runtime = FakeRuntime(
        density=np.array([1.0e17]),
        temperature=np.array([-1.0]),
    )

    adapter, _, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    with pytest.raises(
        PredictionContractError,
        match="non-positive",
    ):
        adapter.predict_batch(
            [
                [52.5, 35.0],
            ]
        )


def test_wrong_density_shape_is_rejected(
    monkeypatch,
) -> None:
    runtime = FakeRuntime(
        density=np.array([[1.0e17]]),
        temperature=np.array([1.8]),
    )

    adapter, _, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    with pytest.raises(
        PredictionContractError,
        match="shape",
    ):
        adapter.predict_batch(
            [
                [52.5, 35.0],
            ]
        )


def test_wrong_temperature_shape_is_rejected(
    monkeypatch,
) -> None:
    runtime = FakeRuntime(
        density=np.array([1.0e17]),
        temperature=np.array([[1.8]]),
    )

    adapter, _, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    with pytest.raises(
        PredictionContractError,
        match="shape",
    ):
        adapter.predict_batch(
            [
                [52.5, 35.0],
            ]
        )


def test_returned_arrays_are_read_only(
    monkeypatch,
) -> None:
    adapter, _, _ = _loaded_adapter(
        monkeypatch
    )

    result = adapter.predict_batch(
        [
            [52.5, 35.0],
        ]
    )

    assert result.input_matrix.flags.writeable is False
    assert (
        result.electron_density_m3.flags.writeable
        is False
    )
    assert (
        result.electron_temperature_eV.flags.writeable
        is False
    )

    with pytest.raises(ValueError):
        result.input_matrix[0, 0] = 20.0

    with pytest.raises(ValueError):
        result.electron_density_m3[0] = 2.0e17


def test_prediction_arrays_do_not_alias_runtime_arrays(
    monkeypatch,
) -> None:
    density = np.array([1.0e17])
    temperature = np.array([1.8])

    runtime = FakeRuntime(
        density=density,
        temperature=temperature,
    )

    adapter, _, _ = _loaded_adapter(
        monkeypatch,
        runtime=runtime,
    )

    result = adapter.predict_batch(
        [
            [52.5, 35.0],
        ]
    )

    density[0] = 2.0e17
    temperature[0] = 2.5

    assert (
        result.electron_density_m3[0]
        == pytest.approx(1.0e17)
    )

    assert (
        result.electron_temperature_eV[0]
        == pytest.approx(1.8)
    )
