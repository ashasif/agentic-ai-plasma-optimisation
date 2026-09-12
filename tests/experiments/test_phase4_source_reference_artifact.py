"""Tests for Phase 4E source-reference artifact construction."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from plasma_ai.experiments.base_dataset import (
    BaseSimulationRow,
)
from plasma_ai.surrogate.probe_grid import (
    Phase4EProbeGrid,
)
from plasma_ai.surrogate.source_reference import (
    Phase4ESourceReference,
)
from plasma_ai.surrogate.source_reference_artifact import (
    build_source_reference_payload,
    write_source_reference_artifact,
)


def _grid() -> Phase4EProbeGrid:
    return Phase4EProbeGrid(
        absorbed_power_W=np.asarray(
            [15.0, 90.0],
            dtype=float,
        ),
        target_pressure_mTorr=np.asarray(
            [10.0, 60.0],
            dtype=float,
        ),
        features=np.asarray(
            [
                [15.0, 10.0],
                [15.0, 60.0],
                [90.0, 10.0],
                [90.0, 60.0],
            ],
            dtype=float,
        ),
    )


def _row(index, power, pressure):
    return BaseSimulationRow(
        simulation_id=f"phase4e_probe_{index:04d}",
        split="phase4e_probe",
        split_index=index,
        design_seed=0,
        nominal_absorbed_power_W=power,
        target_pressure_mTorr=pressure,
        nominal_flow_sccm=20.0,
        derived_pumping_speed_m3_s=1.0,
        gas_temperature_K=300.0,
        ion_neutral_cross_section_m2=1.0e-18,
        radius_m=0.17,
        length_m=0.25,
        true_neutral_density_m3=1.0e20,
        true_ion_density_m3=1.0e17,
        true_electron_density_m3=(
            1.0e17 + power
        ),
        true_electron_energy_density_J_m3=1.0,
        true_electron_temperature_eV=(
            2.0 + pressure * 1.0e-3
        ),
        true_pressure_mTorr=pressure,
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
        qualification_valid=True,
        is_ml_eligible=True,
        error_message="",
    )


def _reference():
    grid = _grid()

    rows = tuple(
        _row(
            index,
            float(feature[0]),
            float(feature[1]),
        )
        for index, feature
        in enumerate(grid.features)
    )

    return Phase4ESourceReference(
        rows=rows,
        electron_density_m3=np.asarray(
            [
                row.true_electron_density_m3
                for row in rows
            ],
            dtype=float,
        ),
        electron_temperature_eV=np.asarray(
            [
                row.true_electron_temperature_eV
                for row in rows
            ],
            dtype=float,
        ),
    )


def test_source_reference_payload_records_source_gate():
    payload = build_source_reference_payload(
        _grid(),
        _reference(),
    )

    gate = payload["source_reference_gate"]

    assert gate["all_targets_finite"] is True
    assert gate["all_qualification_valid"] is True
    assert gate["qualification_valid_count"] == 4
    assert gate["integration_success_count"] == 4
    assert gate["converged_count"] == 4


def test_source_reference_payload_preserves_probe_order():
    payload = build_source_reference_payload(
        _grid(),
        _reference(),
    )

    observed = [
        (
            row["nominal_absorbed_power_W"],
            row["target_pressure_mTorr"],
        )
        for row in payload["rows"]
    ]

    expected = [
        tuple(map(float, row))
        for row in _grid().features
    ]

    assert observed == expected


def test_source_reference_payload_has_deterministic_array_hash():
    first = build_source_reference_payload(
        _grid(),
        _reference(),
    )

    second = build_source_reference_payload(
        _grid(),
        _reference(),
    )

    first_hash = first[
        "reference_array_contract"
    ]["sha256"]

    second_hash = second[
        "reference_array_contract"
    ]["sha256"]

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_source_reference_payload_rejects_order_mismatch():
    reference = _reference()

    bad_rows = (
        reference.rows[1],
        reference.rows[0],
        *reference.rows[2:],
    )

    bad_reference = Phase4ESourceReference(
        rows=bad_rows,
        electron_density_m3=(
            reference.electron_density_m3
        ),
        electron_temperature_eV=(
            reference.electron_temperature_eV
        ),
    )

    with pytest.raises(
        ValueError,
        match="ordering",
    ):
        build_source_reference_payload(
            _grid(),
            bad_reference,
        )


def test_source_reference_payload_rejects_invalid_gate():
    reference = _reference()

    bad_row = replace(
        reference.rows[0],
        qualification_valid=False,
    )

    bad_reference = Phase4ESourceReference(
        rows=(
            bad_row,
            *reference.rows[1:],
        ),
        electron_density_m3=(
            reference.electron_density_m3
        ),
        electron_temperature_eV=(
            reference.electron_temperature_eV
        ),
    )

    with pytest.raises(
        ValueError,
        match="qualification-invalid",
    ):
        build_source_reference_payload(
            _grid(),
            bad_reference,
        )


def test_write_source_reference_artifact_is_valid_json(
    tmp_path,
):
    destination = (
        tmp_path
        / "source_reference.json"
    )

    result = write_source_reference_artifact(
        _grid(),
        _reference(),
        destination,
    )

    assert result == destination
    assert destination.exists()

    payload = json.loads(
        destination.read_text(
            encoding="utf-8",
        )
    )

    assert payload["phase"] == "4E"
    assert (
        payload["artifact"]
        == "source_reference_grid"
    )
    assert len(payload["rows"]) == 4


def test_source_reference_artifact_preserves_test_lock_scope(
    tmp_path,
):
    destination = (
        tmp_path
        / "source_reference.json"
    )

    write_source_reference_artifact(
        _grid(),
        _reference(),
        destination,
    )

    payload = json.loads(
        destination.read_text(
            encoding="utf-8",
        )
    )

    text = json.dumps(payload)

    assert "test_targets" not in text
    assert "test_metrics" not in text

    scope = payload["scientific_scope"]

    assert scope["experimental_validation"] is False
    assert scope["industrial_validation"] is False
    assert scope["oipt_operating_range_claim"] is False
