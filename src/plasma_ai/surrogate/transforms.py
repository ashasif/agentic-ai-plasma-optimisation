"""Target transformations for Phase 4 surrogate modelling.

Transform choices are frozen by the Phase 4A protocol.

Electron-density candidates:
- identity
- log10

Electron-temperature candidate:
- identity

All final evaluation metrics must be calculated after predictions are
returned to the physical target scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUPPORTED_TARGET_TRANSFORMS = (
    "identity",
    "log10",
)


@dataclass(frozen=True)
class TargetTransform:
    """Deterministic target transformation."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_TARGET_TRANSFORMS:
            raise ValueError(
                f"Unsupported target transform {self.name!r}. "
                f"Supported transforms: "
                f"{SUPPORTED_TARGET_TRANSFORMS}"
            )

    def forward(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Transform physical target values for model fitting."""
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if not np.isfinite(array).all():
            raise ValueError(
                "Target values must all be finite."
            )

        if self.name == "identity":
            return array.copy()

        if np.any(array <= 0.0):
            raise ValueError(
                "log10 target transformation requires "
                "strictly positive values."
            )

        return np.log10(array)

    def inverse(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Return transformed predictions to physical units."""
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if not np.isfinite(array).all():
            raise ValueError(
                "Predicted transformed values must all be finite."
            )

        if self.name == "identity":
            return array.copy()

        physical = np.power(
            10.0,
            array,
        )

        if not np.isfinite(physical).all():
            raise ValueError(
                "Inverse log10 transformation produced "
                "non-finite physical values."
            )

        return physical


def allowed_transforms_for_target(
    target_name: str,
) -> tuple[str, ...]:
    """Return the frozen transformation candidates for a target."""
    if target_name == "true_electron_density_m3":
        return (
            "identity",
            "log10",
        )

    if target_name == "true_electron_temperature_eV":
        return ("identity",)

    raise KeyError(
        f"Unknown Phase 4 target {target_name!r}."
    )
