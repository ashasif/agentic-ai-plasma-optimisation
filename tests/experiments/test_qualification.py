import numpy as np
import pytest

from plasma_ai.experiments.qualification import (
    ABSORBED_POWER_LEVELS_W,
    FLOW_LEVELS_SCCM,
    PRESSURE_LEVELS_MTORR,
    BASE_SURROGATE_FLOW_SCCM,
    SOBOL_PILOT_POWER_MAX_W,
    SOBOL_PILOT_POWER_MIN_W,
    SOBOL_PILOT_PRESSURE_MAX_MTORR,
    SOBOL_PILOT_PRESSURE_MIN_MTORR,
    QualificationInput,
    continuous_sobol_pilot_design,
    qualification_design,
    run_qualification_point,
)


def test_qualification_design_contains_27_unique_points():
    design = qualification_design()

    assert len(design) == 27

    combinations = {
        (
            point.absorbed_power_W,
            point.target_pressure_mTorr,
            point.flow_sccm,
        )
        for point in design
    }

    assert len(combinations) == 27


def test_qualification_design_uses_frozen_levels():
    design = qualification_design()

    assert {
        point.absorbed_power_W
        for point in design
    } == set(ABSORBED_POWER_LEVELS_W)

    assert {
        point.target_pressure_mTorr
        for point in design
    } == set(PRESSURE_LEVELS_MTORR)

    assert {
        point.flow_sccm
        for point in design
    } == set(FLOW_LEVELS_SCCM)


def test_reference_point_reproduces_phase2_state():
    point = QualificationInput(
        qualification_id="REFERENCE",
        absorbed_power_W=50.0,
        target_pressure_mTorr=30.0,
        flow_sccm=20.0,
    )

    result = run_qualification_point(point)

    assert result.integration_success
    assert result.converged
    assert result.physical_state_valid
    assert result.balance_valid
    assert result.qualification_valid

    assert result.electron_density_m3 == pytest.approx(
        9.371273e16,
        rel=1.0e-5,
    )

    assert result.electron_temperature_eV == pytest.approx(
        1.717831,
        rel=1.0e-5,
    )

    assert result.solved_pressure_mTorr == pytest.approx(
        30.0,
        rel=1.0e-5,
    )


def test_reference_point_meets_phase2_validation_limits():
    point = QualificationInput(
        qualification_id="REFERENCE",
        absorbed_power_W=50.0,
        target_pressure_mTorr=30.0,
        flow_sccm=20.0,
    )

    result = run_qualification_point(point)

    assert result.max_relative_rate_s <= 1.0e-6

    assert result.neutral_particle_balance < 1.0e-6
    assert result.ion_particle_balance < 1.0e-6
    assert result.electron_energy_balance < 1.0e-6
    assert result.max_balance_residual < 1.0e-6
    assert result.total_particle_identity < 1.0e-12

    assert np.isfinite(
        result.pressure_target_relative_error
    )


def test_continuous_sobol_pilot_has_256_unique_points():
    design = continuous_sobol_pilot_design()

    assert len(design) == 256

    combinations = {
        (
            point.absorbed_power_W,
            point.target_pressure_mTorr,
        )
        for point in design
    }

    assert len(combinations) == 256


def test_continuous_sobol_pilot_respects_frozen_bounds_and_flow():
    design = continuous_sobol_pilot_design()

    for point in design:
        assert (
            SOBOL_PILOT_POWER_MIN_W
            <= point.absorbed_power_W
            <= SOBOL_PILOT_POWER_MAX_W
        )

        assert (
            SOBOL_PILOT_PRESSURE_MIN_MTORR
            <= point.target_pressure_mTorr
            <= SOBOL_PILOT_PRESSURE_MAX_MTORR
        )

        assert point.flow_sccm == BASE_SURROGATE_FLOW_SCCM


def test_continuous_sobol_pilot_is_deterministic_for_seed():
    first = continuous_sobol_pilot_design(seed=20260910)
    second = continuous_sobol_pilot_design(seed=20260910)

    first_values = [
        (
            point.absorbed_power_W,
            point.target_pressure_mTorr,
            point.flow_sccm,
        )
        for point in first
    ]

    second_values = [
        (
            point.absorbed_power_W,
            point.target_pressure_mTorr,
            point.flow_sccm,
        )
        for point in second
    ]

    assert first_values == second_values
