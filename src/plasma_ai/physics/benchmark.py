"""Literature-oriented trend benchmarks for the argon global model.

These sweeps test qualitative behaviour reported for inductively
coupled argon plasmas. Model absorbed power must not be interpreted
as identical to experimentally applied RF generator power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_ai.physics.energy_losses import (
    electron_energy_density_j_m3,
)
from plasma_ai.physics.gas_flow import (
    effective_pumping_speed_m3_s,
    mtorr_to_pa,
    neutral_density_m3_from_pressure_pa,
)
from plasma_ai.physics.global_model import GlobalModelParameters
from plasma_ai.physics.solver import solve_global_model


@dataclass(frozen=True)
class BenchmarkPoint:
    pressure_mtorr: float
    absorbed_power_W: float
    electron_density_m3: float
    electron_temperature_eV: float
    neutral_density_m3: float
    converged: bool
    max_relative_rate_s: float


def simulate_benchmark_point(
    pressure_mtorr: float,
    absorbed_power_W: float,
    *,
    flow_sccm: float = 20.0,
    gas_temperature_K: float = 300.0,
    radius_m: float = 0.17,
    length_m: float = 0.25,
    initial_electron_density_m3: float = 1.0e15,
    initial_electron_temperature_eV: float = 3.0,
) -> BenchmarkPoint:
    """Solve one reduced-order argon benchmark operating point."""
    pressure_pa = mtorr_to_pa(pressure_mtorr)

    initial_neutral_density = (
        neutral_density_m3_from_pressure_pa(
            pressure_pa,
            gas_temperature_K,
        )
    )

    pumping_speed = effective_pumping_speed_m3_s(
        flow_sccm,
        pressure_pa,
        gas_temperature_K,
    )

    initial_energy_density = electron_energy_density_j_m3(
        initial_electron_density_m3,
        initial_electron_temperature_eV,
    )

    initial_state = np.array(
        [
            initial_neutral_density,
            initial_electron_density_m3,
            initial_energy_density,
        ],
        dtype=float,
    )

    params = GlobalModelParameters(
        radius_m=radius_m,
        length_m=length_m,
        absorbed_power_W=absorbed_power_W,
        flow_sccm=flow_sccm,
        pumping_speed_m3_s=pumping_speed,
        gas_temperature_K=gas_temperature_K,
    )

    solution = solve_global_model(
        initial_state,
        params,
        end_time_s=20.0,
        convergence_threshold_s=1.0e-6,
    )

    return BenchmarkPoint(
        pressure_mtorr=float(pressure_mtorr),
        absorbed_power_W=float(absorbed_power_W),
        electron_density_m3=float(
            solution.ion_density_m3[-1]
        ),
        electron_temperature_eV=float(
            solution.electron_temperature_eV[-1]
        ),
        neutral_density_m3=float(
            solution.neutral_density_m3[-1]
        ),
        converged=bool(solution.converged),
        max_relative_rate_s=float(
            solution.max_relative_rate_s
        ),
    )


def pressure_sweep(
    *,
    absorbed_power_W: float = 50.0,
) -> list[BenchmarkPoint]:
    """Run the literature-inspired pressure sweep."""
    pressures_mtorr = (
        7.5,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
    )

    return [
        simulate_benchmark_point(
            pressure,
            absorbed_power_W,
        )
        for pressure in pressures_mtorr
    ]


def power_sweep(
    *,
    pressure_mtorr: float = 30.0,
) -> list[BenchmarkPoint]:
    """Run an absorbed-power trend sweep at fixed pressure."""
    powers_W = (
        10.0,
        20.0,
        50.0,
        100.0,
    )

    return [
        simulate_benchmark_point(
            pressure_mtorr,
            power,
        )
        for power in powers_W
    ]
