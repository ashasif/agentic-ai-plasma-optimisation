"""Trusted Phase 4G surrogate adapter for Phase 6.

Phase 6 may use the frozen surrogate only through the trusted Phase 4G
persistence layer. This module adds Phase 6 input/output validation around
that already-qualified runtime.

No optimisation search or source-model execution occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plasma_ai.optimisation.primitives import (
    validate_decision_point,
)
from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    file_sha256,
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
)
from plasma_ai.surrogate.phase4g_persistence import (
    LoadedPhase4GSurrogate,
    load_phase4g_surrogate,
)


class SurrogateAdapterError(RuntimeError):
    """Raised when the trusted Phase 6 surrogate adapter cannot load."""


class PredictionContractError(ValueError):
    """Raised when frozen surrogate output violates its Phase 6 contract."""


@dataclass(frozen=True)
class Phase6Prediction:
    """Physical surrogate prediction for one validated operating point."""

    operating_point: OperatingPoint
    electron_density_m3: float
    electron_temperature_eV: float


@dataclass(frozen=True)
class Phase6PredictionBatch:
    """Validated immutable batch of Phase 6 surrogate predictions."""

    input_matrix: NDArray[np.float64]
    electron_density_m3: NDArray[np.float64]
    electron_temperature_eV: NDArray[np.float64]


@dataclass(frozen=True)
class Phase6SurrogateAdapter:
    """Validated Phase 6 wrapper around the trusted Phase 4G runtime."""

    protocol: EffectivePhase6Protocol
    surrogate: LoadedPhase4GSurrogate
    manifest_path: Path
    manifest_sha256: str

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return the exact frozen model feature order."""

        return tuple(
            self.protocol.base[
                "frozen_phase4_contract"
            ]["feature_order"]
        )

    def predict_batch(
        self,
        values: ArrayLike,
    ) -> Phase6PredictionBatch:
        """Predict a validated in-domain two-feature batch."""

        matrix = _validated_phase6_input_matrix(
            values,
            protocol=self.protocol,
        )

        predictions = self.surrogate.predict_physical(
            matrix
        )

        try:
            density_raw = predictions.electron_density_m3
            temperature_raw = (
                predictions.electron_temperature_eV
            )
        except AttributeError as exc:
            raise PredictionContractError(
                "Trusted Phase 4G prediction container does not "
                "provide the frozen physical output fields."
            ) from exc

        density = _validated_prediction_vector(
            density_raw,
            rows=matrix.shape[0],
            field="electron_density_m3",
        )

        temperature = _validated_prediction_vector(
            temperature_raw,
            rows=matrix.shape[0],
            field="electron_temperature_eV",
        )

        return Phase6PredictionBatch(
            input_matrix=_immutable_float_array(
                matrix
            ),
            electron_density_m3=_immutable_float_array(
                density
            ),
            electron_temperature_eV=_immutable_float_array(
                temperature
            ),
        )

    def predict_point(
        self,
        point: OperatingPoint,
    ) -> Phase6Prediction:
        """Predict one validated operating point."""

        validated = validate_decision_point(
            point.nominal_absorbed_power_W,
            point.target_pressure_mTorr,
            protocol=self.protocol,
        )

        batch = self.predict_batch(
            [
                [
                    validated.nominal_absorbed_power_W,
                    validated.target_pressure_mTorr,
                ]
            ]
        )

        return Phase6Prediction(
            operating_point=validated,
            electron_density_m3=float(
                batch.electron_density_m3[0]
            ),
            electron_temperature_eV=float(
                batch.electron_temperature_eV[0]
            ),
        )


def _immutable_float_array(
    values: ArrayLike,
) -> NDArray[np.float64]:
    """Return an independent read-only float64 array."""

    output = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    output.setflags(write=False)

    return output


def _validated_phase6_input_matrix(
    values: ArrayLike,
    *,
    protocol: EffectivePhase6Protocol,
) -> NDArray[np.float64]:
    """Validate the Phase 6 batch before any surrogate call."""

    try:
        raw = np.asarray(
            values,
            dtype=object,
        )
    except Exception as exc:
        raise PredictionContractError(
            "Phase 6 surrogate input could not be represented "
            "as a matrix."
        ) from exc

    if raw.ndim != 2:
        raise PredictionContractError(
            "Phase 6 batch input must be a two-dimensional matrix."
        )

    if raw.shape[1] != 2:
        raise PredictionContractError(
            "Phase 6 batch input must contain exactly two columns."
        )

    if raw.shape[0] == 0:
        raise PredictionContractError(
            "Phase 6 batch input must contain at least one row."
        )

    validated_rows: list[list[float]] = []

    for row_index in range(raw.shape[0]):
        point = validate_decision_point(
            raw[row_index, 0],
            raw[row_index, 1],
            protocol=protocol,
        )

        validated_rows.append(
            [
                point.nominal_absorbed_power_W,
                point.target_pressure_mTorr,
            ]
        )

    matrix = np.asarray(
        validated_rows,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(matrix)):
        raise PredictionContractError(
            "Validated Phase 6 input unexpectedly contains "
            "non-finite values."
        )

    return matrix


def _validated_prediction_vector(
    values: Any,
    *,
    rows: int,
    field: str,
) -> NDArray[np.float64]:
    """Validate one frozen physical prediction vector."""

    try:
        vector = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise PredictionContractError(
            f"{field} could not be converted to float64."
        ) from exc

    expected_shape = (rows,)

    if vector.shape != expected_shape:
        raise PredictionContractError(
            f"{field} has shape {vector.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(np.isfinite(vector)):
        raise PredictionContractError(
            f"{field} contains non-finite predictions."
        )

    if np.any(vector <= 0.0):
        raise PredictionContractError(
            f"{field} contains non-positive predictions."
        )

    return vector


def load_phase6_surrogate_adapter() -> Phase6SurrogateAdapter:
    """Load Phase 6 inference through the trusted frozen Phase 4G path."""

    protocol = load_effective_phase6_protocol()

    interface = protocol.base[
        "trusted_surrogate_interface"
    ]

    if (
        interface["module"]
        != "plasma_ai.surrogate.phase4g_persistence"
    ):
        raise SurrogateAdapterError(
            "Frozen trusted-surrogate module contract drift."
        )

    if (
        interface["loader_function"]
        != "load_phase4g_surrogate"
    ):
        raise SurrogateAdapterError(
            "Frozen trusted-surrogate loader contract drift."
        )

    if (
        interface["prediction_method"]
        != "predict_physical"
    ):
        raise SurrogateAdapterError(
            "Frozen prediction-method contract drift."
        )

    if tuple(
        interface["prediction_container_fields"]
    ) != (
        "electron_density_m3",
        "electron_temperature_eV",
    ):
        raise SurrogateAdapterError(
            "Frozen prediction-output contract drift."
        )

    manifest_path = Path(
        interface["production_manifest_path"]
    )

    persistence_source_path = Path(
        interface["persistence_source_path"]
    )

    try:
        observed_manifest_hash = file_sha256(
            manifest_path
        )

        observed_source_hash = file_sha256(
            persistence_source_path
        )
    except OSError as exc:
        raise SurrogateAdapterError(
            "Trusted Phase 4G inference input could not be read."
        ) from exc

    if (
        observed_manifest_hash
        != interface["production_manifest_sha256"]
    ):
        raise SurrogateAdapterError(
            "Frozen Phase 4G surrogate-manifest hash mismatch."
        )

    if (
        observed_source_hash
        != interface["persistence_source_sha256"]
    ):
        raise SurrogateAdapterError(
            "Trusted Phase 4G persistence-source hash mismatch."
        )

    try:
        surrogate = load_phase4g_surrogate(
            manifest_path=manifest_path,
        )
    except Exception as exc:
        raise SurrogateAdapterError(
            "Trusted Phase 4G surrogate loading failed."
        ) from exc

    return Phase6SurrogateAdapter(
        protocol=protocol,
        surrogate=surrogate,
        manifest_path=manifest_path,
        manifest_sha256=observed_manifest_hash,
    )
