"""Operating-envelope qualification for Phase 3.

This module qualifies a deliberately limited synthetic design envelope
around the reduced-order argon global model. The ranges defined here
are model-study ranges, not Oxford Instruments operating ranges and
not experimentally validated industrial process limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.stats import qmc

from plasma_ai.physics.energy_losses import (
    electron_energy_density_j_m3,
)
from plasma_ai.physics.gas_flow import (
    effective_pumping_speed_m3_s,
    mtorr_to_pa,
    neutral_density_m3_from_pressure_pa,
    pa_to_mtorr,
    pressure_pa_from_neutral_density_m3,
)
from plasma_ai.physics.global_model import (
    GlobalModelParameters,
)
from plasma_ai.physics.solver import solve_global_model
from plasma_ai.physics.validation import (
    steady_state_balance_residuals,
)


RADIUS_M = 0.17
LENGTH_M = 0.25

GAS_TEMPERATURE_K = 300.0
ION_NEUTRAL_CROSS_SECTION_M2 = 1.0e-18

INITIAL_ELECTRON_DENSITY_M3 = 1.0e15
INITIAL_ELECTRON_TEMPERATURE_EV = 3.0

END_TIME_S = 20.0
CONVERGENCE_THRESHOLD_S = 1.0e-6

BALANCE_RESIDUAL_LIMIT = 1.0e-6
TOTAL_PARTICLE_IDENTITY_LIMIT = 1.0e-12

MIN_REASONABLE_ELECTRON_TEMPERATURE_EV = 0.5
MAX_REASONABLE_ELECTRON_TEMPERATURE_EV = 10.0

ABSORBED_POWER_LEVELS_W = (
    15.0,
    50.0,
    90.0,
)

PRESSURE_LEVELS_MTORR = (
    10.0,
    30.0,
    60.0,
)

FLOW_LEVELS_SCCM = (
    15.0,
    20.0,
    25.0,
)

# Frozen Phase-3A amendment:
# base surrogate design varies absorbed power and pressure only.
BASE_SURROGATE_FLOW_SCCM = 20.0

# Reproducible continuous-envelope pilot.
SOBOL_PILOT_SEED = 20260910
SOBOL_PILOT_POWER_MIN_W = 15.0
SOBOL_PILOT_POWER_MAX_W = 90.0
SOBOL_PILOT_PRESSURE_MIN_MTORR = 10.0
SOBOL_PILOT_PRESSURE_MAX_MTORR = 60.0
SOBOL_PILOT_EXPONENT = 8


@dataclass(frozen=True)
class QualificationInput:
    """One requested operating point in the Phase-3 qualification."""

    qualification_id: str
    absorbed_power_W: float
    target_pressure_mTorr: float
    flow_sccm: float


@dataclass(frozen=True)
class QualificationPoint:
    """Result of one operating-envelope qualification simulation."""

    qualification_id: str

    absorbed_power_W: float
    target_pressure_mTorr: float
    flow_sccm: float
    pumping_speed_m3_s: float

    integration_success: bool
    converged: bool
    max_relative_rate_s: float

    neutral_density_m3: float
    ion_density_m3: float
    electron_density_m3: float
    electron_energy_density_J_m3: float
    electron_temperature_eV: float

    solved_pressure_mTorr: float
    pressure_target_relative_error: float

    neutral_particle_balance: float
    ion_particle_balance: float
    electron_energy_balance: float
    total_particle_identity: float
    max_balance_residual: float

    physical_state_valid: bool
    balance_valid: bool
    qualification_valid: bool

    error_message: str


def qualification_design() -> list[QualificationInput]:
    """Return the frozen 3 x 3 x 3 Phase-3B qualification design."""
    combinations = product(
        ABSORBED_POWER_LEVELS_W,
        PRESSURE_LEVELS_MTORR,
        FLOW_LEVELS_SCCM,
    )

    return [
        QualificationInput(
            qualification_id=f"Q{index:03d}",
            absorbed_power_W=float(power),
            target_pressure_mTorr=float(pressure),
            flow_sccm=float(flow),
        )
        for index, (power, pressure, flow) in enumerate(
            combinations,
            start=1,
        )
    ]


def continuous_sobol_pilot_design(
    *,
    seed: int = SOBOL_PILOT_SEED,
) -> list[QualificationInput]:
    """Return the reproducible 256-point continuous Phase-3B pilot.

    Only absorbed plasma power and target pressure are sampled.
    Argon flow is fixed because the Phase-3B factorial experiment
    showed that it is structurally redundant for nominal steady-state
    plasma outputs when pumping speed is adjusted to hold pressure.
    """
    sampler = qmc.Sobol(
        d=2,
        scramble=True,
        seed=seed,
    )

    unit_design = sampler.random_base2(
        m=SOBOL_PILOT_EXPONENT,
    )

    scaled_design = qmc.scale(
        unit_design,
        l_bounds=[
            SOBOL_PILOT_POWER_MIN_W,
            SOBOL_PILOT_PRESSURE_MIN_MTORR,
        ],
        u_bounds=[
            SOBOL_PILOT_POWER_MAX_W,
            SOBOL_PILOT_PRESSURE_MAX_MTORR,
        ],
    )

    return [
        QualificationInput(
            qualification_id=f"S{index:03d}",
            absorbed_power_W=float(row[0]),
            target_pressure_mTorr=float(row[1]),
            flow_sccm=BASE_SURROGATE_FLOW_SCCM,
        )
        for index, row in enumerate(
            scaled_design,
            start=1,
        )
    ]


def _failure_result(
    point: QualificationInput,
    pumping_speed_m3_s: float,
    error_message: str,
) -> QualificationPoint:
    """Return an explicit record for a failed integration."""
    nan = float("nan")

    return QualificationPoint(
        qualification_id=point.qualification_id,
        absorbed_power_W=point.absorbed_power_W,
        target_pressure_mTorr=point.target_pressure_mTorr,
        flow_sccm=point.flow_sccm,
        pumping_speed_m3_s=pumping_speed_m3_s,
        integration_success=False,
        converged=False,
        max_relative_rate_s=nan,
        neutral_density_m3=nan,
        ion_density_m3=nan,
        electron_density_m3=nan,
        electron_energy_density_J_m3=nan,
        electron_temperature_eV=nan,
        solved_pressure_mTorr=nan,
        pressure_target_relative_error=nan,
        neutral_particle_balance=nan,
        ion_particle_balance=nan,
        electron_energy_balance=nan,
        total_particle_identity=nan,
        max_balance_residual=nan,
        physical_state_valid=False,
        balance_valid=False,
        qualification_valid=False,
        error_message=error_message,
    )


def run_qualification_point(
    point: QualificationInput,
) -> QualificationPoint:
    """Solve and independently validate one qualification point."""
    pressure_pa = mtorr_to_pa(
        point.target_pressure_mTorr,
    )

    initial_neutral_density_m3 = (
        neutral_density_m3_from_pressure_pa(
            pressure_pa,
            GAS_TEMPERATURE_K,
        )
    )

    pumping_speed_m3_s = effective_pumping_speed_m3_s(
        point.flow_sccm,
        pressure_pa,
        GAS_TEMPERATURE_K,
    )

    initial_energy_density_J_m3 = (
        electron_energy_density_j_m3(
            INITIAL_ELECTRON_DENSITY_M3,
            INITIAL_ELECTRON_TEMPERATURE_EV,
        )
    )

    initial_state = np.array(
        [
            initial_neutral_density_m3,
            INITIAL_ELECTRON_DENSITY_M3,
            initial_energy_density_J_m3,
        ],
        dtype=float,
    )

    params = GlobalModelParameters(
        radius_m=RADIUS_M,
        length_m=LENGTH_M,
        absorbed_power_W=point.absorbed_power_W,
        flow_sccm=point.flow_sccm,
        pumping_speed_m3_s=pumping_speed_m3_s,
        gas_temperature_K=GAS_TEMPERATURE_K,
        ion_neutral_cross_section_m2=(
            ION_NEUTRAL_CROSS_SECTION_M2
        ),
    )

    try:
        solution = solve_global_model(
            initial_state,
            params,
            end_time_s=END_TIME_S,
            convergence_threshold_s=(
                CONVERGENCE_THRESHOLD_S
            ),
        )
    except RuntimeError as exc:
        return _failure_result(
            point,
            pumping_speed_m3_s,
            str(exc),
        )

    final_state = solution.final_state

    neutral_density_m3 = float(final_state[0])
    ion_density_m3 = float(final_state[1])
    electron_energy_density_J_m3 = float(
        final_state[2],
    )

    # Bulk quasineutrality is a frozen model assumption.
    electron_density_m3 = ion_density_m3

    electron_temperature_eV = float(
        solution.electron_temperature_eV[-1],
    )

    solved_pressure_pa = (
        pressure_pa_from_neutral_density_m3(
            neutral_density_m3,
            GAS_TEMPERATURE_K,
        )
    )

    solved_pressure_mTorr = pa_to_mtorr(
        solved_pressure_pa,
    )

    pressure_target_relative_error = (
        abs(
            solved_pressure_mTorr
            - point.target_pressure_mTorr
        )
        / point.target_pressure_mTorr
    )

    residuals = steady_state_balance_residuals(
        final_state,
        params,
    )

    finite_state = all(
        math.isfinite(value)
        for value in (
            neutral_density_m3,
            ion_density_m3,
            electron_density_m3,
            electron_energy_density_J_m3,
            electron_temperature_eV,
            solved_pressure_mTorr,
            solution.max_relative_rate_s,
        )
    )

    physical_state_valid = (
        finite_state
        and neutral_density_m3 > 0.0
        and 0.0
        < electron_density_m3
        < neutral_density_m3
        and electron_energy_density_J_m3 > 0.0
        and (
            MIN_REASONABLE_ELECTRON_TEMPERATURE_EV
            < electron_temperature_eV
            < MAX_REASONABLE_ELECTRON_TEMPERATURE_EV
        )
        and solved_pressure_mTorr > 0.0
    )

    balance_valid = (
        residuals["neutral_particle_balance"]
        < BALANCE_RESIDUAL_LIMIT
        and residuals["ion_particle_balance"]
        < BALANCE_RESIDUAL_LIMIT
        and residuals["electron_energy_balance"]
        < BALANCE_RESIDUAL_LIMIT
        and residuals["max_balance_residual"]
        < BALANCE_RESIDUAL_LIMIT
        and residuals["total_particle_identity"]
        < TOTAL_PARTICLE_IDENTITY_LIMIT
    )

    qualification_valid = (
        bool(solution.success)
        and bool(solution.converged)
        and physical_state_valid
        and balance_valid
    )

    return QualificationPoint(
        qualification_id=point.qualification_id,
        absorbed_power_W=point.absorbed_power_W,
        target_pressure_mTorr=point.target_pressure_mTorr,
        flow_sccm=point.flow_sccm,
        pumping_speed_m3_s=pumping_speed_m3_s,
        integration_success=bool(solution.success),
        converged=bool(solution.converged),
        max_relative_rate_s=float(
            solution.max_relative_rate_s,
        ),
        neutral_density_m3=neutral_density_m3,
        ion_density_m3=ion_density_m3,
        electron_density_m3=electron_density_m3,
        electron_energy_density_J_m3=(
            electron_energy_density_J_m3
        ),
        electron_temperature_eV=(
            electron_temperature_eV
        ),
        solved_pressure_mTorr=solved_pressure_mTorr,
        pressure_target_relative_error=(
            pressure_target_relative_error
        ),
        neutral_particle_balance=float(
            residuals["neutral_particle_balance"],
        ),
        ion_particle_balance=float(
            residuals["ion_particle_balance"],
        ),
        electron_energy_balance=float(
            residuals["electron_energy_balance"],
        ),
        total_particle_identity=float(
            residuals["total_particle_identity"],
        ),
        max_balance_residual=float(
            residuals["max_balance_residual"],
        ),
        physical_state_valid=physical_state_valid,
        balance_valid=balance_valid,
        qualification_valid=qualification_valid,
        error_message="",
    )


def run_qualification_design() -> list[QualificationPoint]:
    """Run all 27 points in the frozen Phase-3B design."""
    return [
        run_qualification_point(point)
        for point in qualification_design()
    ]


def run_continuous_sobol_pilot(
    *,
    seed: int = SOBOL_PILOT_SEED,
) -> list[QualificationPoint]:
    """Run the continuous Phase-3B Sobol qualification pilot."""
    return [
        run_qualification_point(point)
        for point in continuous_sobol_pilot_design(
            seed=seed,
        )
    ]
