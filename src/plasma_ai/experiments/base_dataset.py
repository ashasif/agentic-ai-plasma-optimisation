"""Deterministic generation of the Phase-3 base simulator dataset."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
    SPLIT_ORDER,
)
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
    global_model_terms,
)
from plasma_ai.physics.solver import solve_global_model
from plasma_ai.physics.validation import (
    steady_state_balance_residuals,
)


@dataclass(frozen=True)
class BaseDesignPoint:
    """One deterministic operating point before simulation."""

    simulation_id: str
    split: str
    split_index: int
    design_seed: int

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float

    nominal_flow_sccm: float
    gas_temperature_K: float
    ion_neutral_cross_section_m2: float
    radius_m: float
    length_m: float


@dataclass(frozen=True)
class BaseSimulationRow:
    """One canonical deterministic base-dataset simulation row."""

    simulation_id: str
    split: str
    split_index: int
    design_seed: int

    nominal_absorbed_power_W: float
    target_pressure_mTorr: float
    nominal_flow_sccm: float
    derived_pumping_speed_m3_s: float

    gas_temperature_K: float
    ion_neutral_cross_section_m2: float
    radius_m: float
    length_m: float

    true_neutral_density_m3: float
    true_ion_density_m3: float
    true_electron_density_m3: float
    true_electron_energy_density_J_m3: float
    true_electron_temperature_eV: float
    true_pressure_mTorr: float

    ionization_rate_coefficient_m3_s: float
    ionization_density_rate_m3_s: float
    charged_wall_loss_rate_s: float
    charged_wall_loss_density_rate_m3_s: float

    ionization_fraction: float
    residence_time_proxy_s: float

    collisional_power_loss_W: float
    wall_power_loss_W: float

    integration_success: bool
    converged: bool
    max_relative_rate_s: float
    pressure_target_relative_error: float

    neutral_particle_balance: float
    ion_particle_balance: float
    electron_energy_balance: float
    total_particle_identity: float
    max_balance_residual: float

    physical_state_valid: bool
    balance_valid: bool

    domain_status: str
    model_validity_status: str

    qualification_valid: bool
    is_ml_eligible: bool

    error_message: str


def generate_split_design(
    config: BaseDatasetConfig,
    split_name: str,
) -> list[BaseDesignPoint]:
    """Generate one independently scrambled Sobol split."""
    if split_name not in config.splits:
        raise ValueError(
            f"Unknown split: {split_name!r}"
        )

    split = config.splits[split_name]

    exponent = split.rows.bit_length() - 1

    sampler = qmc.Sobol(
        d=2,
        scramble=True,
        seed=split.seed,
    )

    unit_design = sampler.random_base2(
        m=exponent,
    )

    scaled_design = qmc.scale(
        unit_design,
        l_bounds=[
            config.absorbed_power_min_W,
            config.target_pressure_min_mTorr,
        ],
        u_bounds=[
            config.absorbed_power_max_W,
            config.target_pressure_max_mTorr,
        ],
    )

    return [
        BaseDesignPoint(
            simulation_id=(
                f"base_{split_name}_{index:06d}"
            ),
            split=split_name,
            split_index=index,
            design_seed=split.seed,
            nominal_absorbed_power_W=float(row[0]),
            target_pressure_mTorr=float(row[1]),
            nominal_flow_sccm=config.fixed_flow_sccm,
            gas_temperature_K=config.gas_temperature_K,
            ion_neutral_cross_section_m2=(
                config.ion_neutral_cross_section_m2
            ),
            radius_m=config.radius_m,
            length_m=config.length_m,
        )
        for index, row in enumerate(
            scaled_design,
            start=1,
        )
    ]


def generate_base_design(
    config: BaseDatasetConfig,
) -> list[BaseDesignPoint]:
    """Generate the complete deterministic base experimental design."""
    design: list[BaseDesignPoint] = []

    for split_name in SPLIT_ORDER:
        design.extend(
            generate_split_design(
                config,
                split_name,
            )
        )

    return design


def classify_domain_status(
    point: BaseDesignPoint,
    config: BaseDatasetConfig,
) -> str:
    """Classify a point relative to the frozen Phase-3 envelope."""
    power = point.nominal_absorbed_power_W
    pressure = point.target_pressure_mTorr

    if (
        power < config.absorbed_power_min_W
        or power > config.absorbed_power_max_W
        or pressure < config.target_pressure_min_mTorr
        or pressure > config.target_pressure_max_mTorr
    ):
        return "ood"

    power_fraction = (
        (power - config.absorbed_power_min_W)
        / (
            config.absorbed_power_max_W
            - config.absorbed_power_min_W
        )
    )

    pressure_fraction = (
        (pressure - config.target_pressure_min_mTorr)
        / (
            config.target_pressure_max_mTorr
            - config.target_pressure_min_mTorr
        )
    )

    boundary = config.near_boundary_fraction

    if (
        power_fraction <= boundary
        or power_fraction >= 1.0 - boundary
        or pressure_fraction <= boundary
        or pressure_fraction >= 1.0 - boundary
    ):
        return "near_boundary"

    return "supported"


def _failed_simulation_row(
    point: BaseDesignPoint,
    pumping_speed_m3_s: float,
    domain_status: str,
    error_message: str,
) -> BaseSimulationRow:
    """Preserve an integration failure as an explicit dataset row."""
    nan = float("nan")

    return BaseSimulationRow(
        simulation_id=point.simulation_id,
        split=point.split,
        split_index=point.split_index,
        design_seed=point.design_seed,
        nominal_absorbed_power_W=(
            point.nominal_absorbed_power_W
        ),
        target_pressure_mTorr=(
            point.target_pressure_mTorr
        ),
        nominal_flow_sccm=point.nominal_flow_sccm,
        derived_pumping_speed_m3_s=pumping_speed_m3_s,
        gas_temperature_K=point.gas_temperature_K,
        ion_neutral_cross_section_m2=(
            point.ion_neutral_cross_section_m2
        ),
        radius_m=point.radius_m,
        length_m=point.length_m,
        true_neutral_density_m3=nan,
        true_ion_density_m3=nan,
        true_electron_density_m3=nan,
        true_electron_energy_density_J_m3=nan,
        true_electron_temperature_eV=nan,
        true_pressure_mTorr=nan,
        ionization_rate_coefficient_m3_s=nan,
        ionization_density_rate_m3_s=nan,
        charged_wall_loss_rate_s=nan,
        charged_wall_loss_density_rate_m3_s=nan,
        ionization_fraction=nan,
        residence_time_proxy_s=nan,
        collisional_power_loss_W=nan,
        wall_power_loss_W=nan,
        integration_success=False,
        converged=False,
        max_relative_rate_s=nan,
        pressure_target_relative_error=nan,
        neutral_particle_balance=nan,
        ion_particle_balance=nan,
        electron_energy_balance=nan,
        total_particle_identity=nan,
        max_balance_residual=nan,
        physical_state_valid=False,
        balance_valid=False,
        domain_status=domain_status,
        model_validity_status="integration_failed",
        qualification_valid=False,
        is_ml_eligible=False,
        error_message=error_message,
    )


def simulate_base_design_point(
    point: BaseDesignPoint,
    config: BaseDatasetConfig,
) -> BaseSimulationRow:
    """Run and independently validate one deterministic base point."""
    domain_status = classify_domain_status(
        point,
        config,
    )

    target_pressure_pa = mtorr_to_pa(
        point.target_pressure_mTorr,
    )

    pumping_speed_m3_s = effective_pumping_speed_m3_s(
        point.nominal_flow_sccm,
        target_pressure_pa,
        point.gas_temperature_K,
    )

    initial_neutral_density_m3 = (
        neutral_density_m3_from_pressure_pa(
            target_pressure_pa,
            point.gas_temperature_K,
        )
    )

    initial_energy_density_J_m3 = (
        electron_energy_density_j_m3(
            config.initial_electron_density_m3,
            config.initial_electron_temperature_eV,
        )
    )

    initial_state = np.array(
        [
            initial_neutral_density_m3,
            config.initial_electron_density_m3,
            initial_energy_density_J_m3,
        ],
        dtype=float,
    )

    params = GlobalModelParameters(
        radius_m=point.radius_m,
        length_m=point.length_m,
        absorbed_power_W=point.nominal_absorbed_power_W,
        flow_sccm=point.nominal_flow_sccm,
        pumping_speed_m3_s=pumping_speed_m3_s,
        gas_temperature_K=point.gas_temperature_K,
        ion_neutral_cross_section_m2=(
            point.ion_neutral_cross_section_m2
        ),
    )

    try:
        solution = solve_global_model(
            initial_state,
            params,
            end_time_s=config.solver_end_time_s,
            convergence_threshold_s=(
                config.convergence_threshold_s
            ),
        )
    except RuntimeError as exc:
        return _failed_simulation_row(
            point,
            pumping_speed_m3_s,
            domain_status,
            str(exc),
        )

    final_state = solution.final_state

    n0 = float(final_state[0])
    ni = float(final_state[1])
    pe = float(final_state[2])

    # Frozen electropositive bulk quasineutrality assumption.
    ne = ni

    te = float(
        solution.electron_temperature_eV[-1]
    )

    true_pressure_pa = (
        pressure_pa_from_neutral_density_m3(
            n0,
            point.gas_temperature_K,
        )
    )

    true_pressure_mTorr = pa_to_mtorr(
        true_pressure_pa,
    )

    pressure_target_relative_error = (
        abs(
            true_pressure_mTorr
            - point.target_pressure_mTorr
        )
        / point.target_pressure_mTorr
    )

    terms = global_model_terms(
        final_state,
        params,
    )

    residuals = steady_state_balance_residuals(
        final_state,
        params,
    )

    volume_m3 = float(
        terms["volume_m3"]
    )

    ionization_rate_coefficient_m3_s = float(
        terms["ionization_rate_coefficient_m3_s"]
    )

    ionization_density_rate_m3_s = float(
        terms["ionization_density_rate_m3_s"]
    )

    charged_wall_loss_rate_s = float(
        terms["charged_wall_loss_rate_s"]
    )

    charged_wall_loss_density_rate_m3_s = float(
        terms["charged_wall_loss_density_rate_m3_s"]
    )

    collisional_power_loss_W = float(
        terms["collisional_power_loss_density_W_m3"]
        * volume_m3
    )

    wall_power_loss_W = float(
        terms["wall_power_loss_density_W_m3"]
        * volume_m3
    )

    ionization_fraction = (
        ni / (n0 + ni)
    )

    # This is a global V/S engineering proxy, not a spatial
    # residence-time distribution.
    residence_time_proxy_s = (
        volume_m3 / pumping_speed_m3_s
    )

    values_requiring_finiteness = (
        n0,
        ni,
        ne,
        pe,
        te,
        true_pressure_mTorr,
        solution.max_relative_rate_s,
        ionization_rate_coefficient_m3_s,
        ionization_density_rate_m3_s,
        charged_wall_loss_rate_s,
        charged_wall_loss_density_rate_m3_s,
        ionization_fraction,
        residence_time_proxy_s,
        collisional_power_loss_W,
        wall_power_loss_W,
    )

    finite_values = all(
        math.isfinite(value)
        for value in values_requiring_finiteness
    )

    physical_state_valid = (
        finite_values
        and n0 > 0.0
        and 0.0 < ne < n0
        and pe > 0.0
        and (
            config.minimum_reasonable_electron_temperature_eV
            < te
            < config.maximum_reasonable_electron_temperature_eV
        )
        and true_pressure_mTorr > 0.0
        and pumping_speed_m3_s > 0.0
        and ionization_rate_coefficient_m3_s >= 0.0
        and ionization_density_rate_m3_s >= 0.0
        and charged_wall_loss_rate_s >= 0.0
        and charged_wall_loss_density_rate_m3_s >= 0.0
        and 0.0 < ionization_fraction < 1.0
        and residence_time_proxy_s > 0.0
        and collisional_power_loss_W >= 0.0
        and wall_power_loss_W >= 0.0
    )

    balance_valid = bool(
        residuals["neutral_particle_balance"]
        < config.balance_residual_limit
        and residuals["ion_particle_balance"]
        < config.balance_residual_limit
        and residuals["electron_energy_balance"]
        < config.balance_residual_limit
        and residuals["max_balance_residual"]
        < config.balance_residual_limit
        and residuals["total_particle_identity"]
        < config.total_particle_identity_limit
    )

    if not solution.converged:
        model_validity_status = "nonconverged"
    elif not physical_state_valid:
        model_validity_status = "invalid_physical_state"
    elif not balance_valid:
        model_validity_status = "invalid_balance"
    else:
        model_validity_status = "valid"

    qualification_valid = bool(
        bool(solution.success)
        and bool(solution.converged)
        and physical_state_valid
        and balance_valid
    )

    is_ml_eligible = (
        qualification_valid
        and domain_status
        in {
            "supported",
            "near_boundary",
        }
    )

    return BaseSimulationRow(
        simulation_id=point.simulation_id,
        split=point.split,
        split_index=point.split_index,
        design_seed=point.design_seed,
        nominal_absorbed_power_W=(
            point.nominal_absorbed_power_W
        ),
        target_pressure_mTorr=(
            point.target_pressure_mTorr
        ),
        nominal_flow_sccm=point.nominal_flow_sccm,
        derived_pumping_speed_m3_s=pumping_speed_m3_s,
        gas_temperature_K=point.gas_temperature_K,
        ion_neutral_cross_section_m2=(
            point.ion_neutral_cross_section_m2
        ),
        radius_m=point.radius_m,
        length_m=point.length_m,
        true_neutral_density_m3=n0,
        true_ion_density_m3=ni,
        true_electron_density_m3=ne,
        true_electron_energy_density_J_m3=pe,
        true_electron_temperature_eV=te,
        true_pressure_mTorr=true_pressure_mTorr,
        ionization_rate_coefficient_m3_s=(
            ionization_rate_coefficient_m3_s
        ),
        ionization_density_rate_m3_s=(
            ionization_density_rate_m3_s
        ),
        charged_wall_loss_rate_s=(
            charged_wall_loss_rate_s
        ),
        charged_wall_loss_density_rate_m3_s=(
            charged_wall_loss_density_rate_m3_s
        ),
        ionization_fraction=ionization_fraction,
        residence_time_proxy_s=residence_time_proxy_s,
        collisional_power_loss_W=(
            collisional_power_loss_W
        ),
        wall_power_loss_W=wall_power_loss_W,
        integration_success=bool(solution.success),
        converged=bool(solution.converged),
        max_relative_rate_s=float(
            solution.max_relative_rate_s
        ),
        pressure_target_relative_error=(
            pressure_target_relative_error
        ),
        neutral_particle_balance=float(
            residuals["neutral_particle_balance"]
        ),
        ion_particle_balance=float(
            residuals["ion_particle_balance"]
        ),
        electron_energy_balance=float(
            residuals["electron_energy_balance"]
        ),
        total_particle_identity=float(
            residuals["total_particle_identity"]
        ),
        max_balance_residual=float(
            residuals["max_balance_residual"]
        ),
        physical_state_valid=physical_state_valid,
        balance_valid=balance_valid,
        domain_status=domain_status,
        model_validity_status=model_validity_status,
        qualification_valid=qualification_valid,
        is_ml_eligible=is_ml_eligible,
        error_message="",
    )


def generate_base_rows(
    config: BaseDatasetConfig,
    design: Sequence[BaseDesignPoint] | None = None,
) -> list[BaseSimulationRow]:
    """Simulate a supplied design or the complete configured design."""
    points = (
        list(design)
        if design is not None
        else generate_base_design(config)
    )

    return [
        simulate_base_design_point(
            point,
            config,
        )
        for point in points
    ]
