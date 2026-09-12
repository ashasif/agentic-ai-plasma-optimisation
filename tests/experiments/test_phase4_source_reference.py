"""Tests for canonical Phase 4E source-reference evaluation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plasma_ai.experiments.base_dataset import (
    BaseSimulationRow,
)
from plasma_ai.surrogate.probe_grid import (
    Phase4EProbeGrid,
)
import plasma_ai.surrogate.source_reference as source_reference


def _tiny_grid() -> Phase4EProbeGrid:
    power = np.asarray(
        [15.0, 90.0],
        dtype=float,
    )
    pressure = np.asarray(
        [10.0, 60.0],
        dtype=float,
    )

    features = np.asarray(
        [
            [15.0, 10.0],
            [15.0, 60.0],
            [90.0, 10.0],
            [90.0, 60.0],
        ],
        dtype=float,
    )

    return Phase4EProbeGrid(
        absorbed_power_W=power,
        target_pressure_mTorr=pressure,
        features=features,
    )


def _fake_row(
    point,
    *,
    qualification_valid=True,
):
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
        derived_pumping_speed_m3_s=1.0,
        gas_temperature_K=point.gas_temperature_K,
        ion_neutral_cross_section_m2=(
            point.ion_neutral_cross_section_m2
        ),
        radius_m=point.radius_m,
        length_m=point.length_m,
        true_neutral_density_m3=1.0e20,
        true_ion_density_m3=1.0e17,
        true_electron_density_m3=(
            1.0e17
            + point.nominal_absorbed_power_W
        ),
        true_electron_energy_density_J_m3=1.0,
        true_electron_temperature_eV=(
            2.0
            + point.target_pressure_mTorr
            * 1.0e-3
        ),
        true_pressure_mTorr=point.target_pressure_mTorr,
        ionization_rate_coefficient_m3_s=1.0,
        ionization_density_rate_m3_s=1.0,
        charged_wall_loss_rate_s=1.0,
        charged_wall_loss_density_rate_m3_s=1.0,
        ionization_fraction=1.0e-3,
        residence_time_proxy_s=1.0,
        collisional_power_loss_W=1.0,
        wall_power_loss_W=1.0,
        integration_success=True,
        converged=True,
        max_relative_rate_s=0.0,
        pressure_target_relative_error=0.0,
        neutral_particle_balance=0.0,
        ion_particle_balance=0.0,
        electron_energy_balance=0.0,
        total_particle_identity=0.0,
        max_balance_residual=0.0,
        physical_state_valid=True,
        balance_valid=True,
        domain_status="near_boundary",
        model_validity_status="valid",
        qualification_valid=qualification_valid,
        is_ml_eligible=True,
        error_message="",
    )


def test_source_reference_uses_canonical_phase3_simulator(
    monkeypatch,
):
    calls = []

    def fake_simulate(point, config):
        calls.append(
            (
                point,
                config,
            )
        )
        return _fake_row(point)

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    grid = _tiny_grid()

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            grid
        )
    )

    assert len(calls) == 4
    assert result.total_points == 4

    for index, (point, _) in enumerate(calls):
        assert point.split == "phase4e_probe"
        assert point.split_index == index
        assert point.design_seed == 0
        assert point.simulation_id == (
            f"phase4e_probe_{index:04d}"
        )


def test_source_reference_preserves_probe_order(
    monkeypatch,
):
    def fake_simulate(point, config):
        return _fake_row(point)

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    grid = _tiny_grid()

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            grid
        )
    )

    observed = [
        (
            row.nominal_absorbed_power_W,
            row.target_pressure_mTorr,
        )
        for row in result.rows
    ]

    expected = [
        tuple(map(float, row))
        for row in grid.features
    ]

    assert observed == expected


def test_source_reference_uses_phase3_fixed_parameters(
    monkeypatch,
):
    captured = []

    def fake_simulate(point, config):
        captured.append(point)
        return _fake_row(point)

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    source_reference.evaluate_phase4e_source_reference(
        _tiny_grid()
    )

    for point in captured:
        assert point.nominal_flow_sccm == pytest.approx(
            20.0
        )
        assert point.gas_temperature_K == pytest.approx(
            300.0
        )
        assert (
            point.ion_neutral_cross_section_m2
            == pytest.approx(1.0e-18)
        )
        assert point.radius_m == pytest.approx(
            0.17
        )
        assert point.length_m == pytest.approx(
            0.25
        )


def test_source_reference_gate_accepts_valid_reference(
    monkeypatch,
):
    def fake_simulate(point, config):
        return _fake_row(point)

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            _tiny_grid()
        )
    )

    source_reference.validate_phase4e_source_reference(
        result,
        expected_points=4,
    )


def test_source_reference_gate_rejects_nonfinite_target(
    monkeypatch,
):
    def fake_simulate(point, config):
        row = _fake_row(point)

        if point.split_index == 2:
            return replace(
                row,
                true_electron_density_m3=float("nan"),
            )

        return row

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            _tiny_grid()
        )
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        source_reference.validate_phase4e_source_reference(
            result,
            expected_points=4,
        )


def test_source_reference_gate_rejects_invalid_source_row(
    monkeypatch,
):
    def fake_simulate(point, config):
        return _fake_row(
            point,
            qualification_valid=(
                point.split_index != 1
            ),
        )

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            _tiny_grid()
        )
    )

    with pytest.raises(
        ValueError,
        match="qualification-invalid",
    ):
        source_reference.validate_phase4e_source_reference(
            result,
            expected_points=4,
        )


def test_source_reference_gate_rejects_wrong_point_count(
    monkeypatch,
):
    def fake_simulate(point, config):
        return _fake_row(point)

    monkeypatch.setattr(
        source_reference,
        "simulate_base_design_point",
        fake_simulate,
    )

    result = (
        source_reference
        .evaluate_phase4e_source_reference(
            _tiny_grid()
        )
    )

    with pytest.raises(
        ValueError,
        match="point count",
    ):
        source_reference.validate_phase4e_source_reference(
            result,
            expected_points=1681,
        )
