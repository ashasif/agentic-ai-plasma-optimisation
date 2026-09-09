"""Electron-argon reaction-rate coefficients.

The functions in this module implement the simplified Maxwellian-EEDF
rate-coefficient fits used in the argon global model described by
Jan et al. (2013).

Electron temperature must be supplied in electron-volts (eV).
Returned rate coefficients have units of m^3 s^-1.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


RateResult: TypeAlias = float | NDArray[np.float64]


def _validate_temperature(te_eV: ArrayLike) -> NDArray[np.float64]:
    """Convert electron temperature to an array and enforce Te > 0."""
    temperature = np.asarray(te_eV, dtype=float)

    if not np.all(np.isfinite(temperature)):
        raise ValueError("Electron temperature must contain only finite values.")

    if np.any(temperature <= 0.0):
        raise ValueError("Electron temperature must be strictly positive.")

    return temperature


def _return_scalar_if_scalar(
    original: ArrayLike,
    result: NDArray[np.float64],
) -> RateResult:
    """Return a Python float for scalar input, otherwise return an array."""
    if np.asarray(original).ndim == 0:
        return float(result)

    return result


def ionization_rate_m3_s(te_eV: ArrayLike) -> RateResult:
    """Electron-impact ionisation rate coefficient for argon."""
    te = _validate_temperature(te_eV)

    rate = 2.3e-14 * te**0.68 * np.exp(-15.76 / te)

    return _return_scalar_if_scalar(te_eV, rate)


def excitation_rate_m3_s(te_eV: ArrayLike) -> RateResult:
    """Electron-impact excitation rate coefficient for argon."""
    te = _validate_temperature(te_eV)

    rate = 2.5e-15 * te**0.74 * np.exp(-11.56 / te)

    return _return_scalar_if_scalar(te_eV, rate)


def elastic_rate_m3_s(te_eV: ArrayLike) -> RateResult:
    """Effective electron-argon elastic collision rate coefficient."""
    te = _validate_temperature(te_eV)
    log_te = np.log(te)

    rate = (
        2.3e-14
        * te**1.61
        * np.exp(
            0.06 * log_te**2
            - 0.12 * log_te**3
        )
    )

    return _return_scalar_if_scalar(te_eV, rate)
