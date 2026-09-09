"""Numerical integration utilities for the argon global model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from plasma_ai.physics.energy_losses import (
    electron_temperature_eV_from_energy_density,
)
from plasma_ai.physics.global_model import (
    GlobalModelParameters,
    global_model_rhs,
)


@dataclass(frozen=True)
class GlobalModelSolution:
    """Container for a time-dependent global-model simulation."""

    time_s: NDArray[np.float64]
    states: NDArray[np.float64]
    electron_temperature_eV: NDArray[np.float64]
    success: bool
    message: str
    converged: bool
    max_relative_rate_s: float

    @property
    def neutral_density_m3(self) -> NDArray[np.float64]:
        return self.states[0]

    @property
    def ion_density_m3(self) -> NDArray[np.float64]:
        return self.states[1]

    @property
    def electron_energy_density_j_m3(
        self,
    ) -> NDArray[np.float64]:
        return self.states[2]

    @property
    def final_state(self) -> NDArray[np.float64]:
        return self.states[:, -1]


def _validate_initial_state(
    initial_state: ArrayLike,
) -> NDArray[np.float64]:
    """Validate the initial physical state."""
    state = np.asarray(initial_state, dtype=float)

    if state.shape != (3,):
        raise ValueError(
            "Initial state must contain exactly [n0, ni, Pe]."
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            "Initial state must contain only finite values."
        )

    if np.any(state <= 0.0):
        raise ValueError(
            "Initial n0, ni, and Pe must all be strictly positive."
        )

    return state


def _log_state_rhs(
    time_s: float,
    log_state: NDArray[np.float64],
    params: GlobalModelParameters,
) -> NDArray[np.float64]:
    """Return derivatives of the logarithmic state variables."""
    state = np.exp(log_state)

    physical_rhs = global_model_rhs(
        time_s,
        state,
        params,
    )

    return physical_rhs / state


def maximum_relative_rate_s(
    state: ArrayLike,
    params: GlobalModelParameters,
) -> float:
    """Return max(|dy/dt| / |y|) across the three state variables."""
    physical_state = _validate_initial_state(state)

    rhs = global_model_rhs(
        0.0,
        physical_state,
        params,
    )

    return float(
        np.max(
            np.abs(rhs / physical_state)
        )
    )


def solve_global_model(
    initial_state: ArrayLike,
    params: GlobalModelParameters,
    *,
    end_time_s: float = 5.0,
    output_points: int = 301,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    convergence_threshold_s: float = 1.0e-6,
) -> GlobalModelSolution:
    """Integrate the global model using a positivity-preserving log state."""
    state0 = _validate_initial_state(initial_state)

    if not math.isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError(
            "end_time_s must be finite and strictly positive."
        )

    if output_points < 2:
        raise ValueError(
            "output_points must be at least 2."
        )

    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
    ):
        raise ValueError(
            "relative_tolerance must be finite and positive."
        )

    if (
        not math.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0.0
    ):
        raise ValueError(
            "absolute_tolerance must be finite and positive."
        )

    if (
        not math.isfinite(convergence_threshold_s)
        or convergence_threshold_s <= 0.0
    ):
        raise ValueError(
            "convergence_threshold_s must be finite and positive."
        )

    evaluation_times = np.linspace(
        0.0,
        end_time_s,
        output_points,
    )

    solution = solve_ivp(
        fun=lambda t, z: _log_state_rhs(
            t,
            z,
            params,
        ),
        t_span=(0.0, end_time_s),
        y0=np.log(state0),
        method="BDF",
        t_eval=evaluation_times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    if not solution.success:
        raise RuntimeError(
            f"Global-model integration failed: {solution.message}"
        )

    states = np.exp(solution.y)

    temperatures = np.array(
        [
            electron_temperature_eV_from_energy_density(
                states[2, index],
                states[1, index],
            )
            for index in range(states.shape[1])
        ],
        dtype=float,
    )

    final_relative_rate = maximum_relative_rate_s(
        states[:, -1],
        params,
    )

    return GlobalModelSolution(
        time_s=np.asarray(solution.t, dtype=float),
        states=np.asarray(states, dtype=float),
        electron_temperature_eV=temperatures,
        success=bool(solution.success),
        message=str(solution.message),
        converged=(
            final_relative_rate
            <= convergence_threshold_s
        ),
        max_relative_rate_s=final_relative_rate,
    )
