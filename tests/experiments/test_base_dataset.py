from collections import Counter
from dataclasses import fields, replace
from functools import lru_cache
import math

from plasma_ai.experiments.base_dataset import (
    BaseSimulationRow,
    generate_base_design,
    generate_base_rows,
)
from plasma_ai.experiments.dataset_config import (
    SplitConfig,
    load_base_dataset_config,
)


CONFIG_PATH = "configs/phase3/base_dataset.json"


def _config():
    return load_base_dataset_config(
        CONFIG_PATH
    )


def _tiny_config():
    config = _config()

    return replace(
        config,
        splits={
            "train": SplitConfig(
                rows=8,
                seed=20260910,
            ),
            "validation": SplitConfig(
                rows=4,
                seed=20260911,
            ),
            "test": SplitConfig(
                rows=4,
                seed=20260912,
            ),
        },
    )


@lru_cache(maxsize=1)
def _tiny_regeneration_pair():
    config = _tiny_config()
    design = generate_base_design(config)

    first = tuple(
        generate_base_rows(
            config,
            design,
        )
    )

    second = tuple(
        generate_base_rows(
            config,
            design,
        )
    )

    return first, second


def test_base_dataset_config_has_frozen_split_counts():
    config = _config()

    assert config.splits["train"].rows == 4096
    assert config.splits["validation"].rows == 2048
    assert config.splits["test"].rows == 2048
    assert config.total_rows == 8192


def test_base_dataset_config_has_frozen_sampling_envelope():
    config = _config()

    assert config.absorbed_power_min_W == 15.0
    assert config.absorbed_power_max_W == 90.0

    assert config.target_pressure_min_mTorr == 10.0
    assert config.target_pressure_max_mTorr == 60.0

    assert config.fixed_flow_sccm == 20.0
    assert config.gas_temperature_K == 300.0

    assert (
        config.ion_neutral_cross_section_m2
        == 1.0e-18
    )


def test_complete_base_design_has_expected_size_and_split_counts():
    design = generate_base_design(
        _config()
    )

    assert len(design) == 8192

    split_counts = Counter(
        point.split for point in design
    )

    assert split_counts == {
        "train": 4096,
        "validation": 2048,
        "test": 2048,
    }


def test_complete_base_design_has_unique_ids_and_coordinates():
    design = generate_base_design(
        _config()
    )

    ids = [
        point.simulation_id
        for point in design
    ]

    coordinates = [
        (
            point.nominal_absorbed_power_W,
            point.target_pressure_mTorr,
        )
        for point in design
    ]

    assert len(set(ids)) == 8192
    assert len(set(coordinates)) == 8192


def test_complete_base_design_respects_bounds_and_fixed_parameters():
    config = _config()
    design = generate_base_design(config)

    for point in design:
        assert (
            config.absorbed_power_min_W
            <= point.nominal_absorbed_power_W
            <= config.absorbed_power_max_W
        )

        assert (
            config.target_pressure_min_mTorr
            <= point.target_pressure_mTorr
            <= config.target_pressure_max_mTorr
        )

        assert (
            point.nominal_flow_sccm
            == config.fixed_flow_sccm
        )

        assert (
            point.gas_temperature_K
            == config.gas_temperature_K
        )

        assert (
            point.ion_neutral_cross_section_m2
            == config.ion_neutral_cross_section_m2
        )

        assert point.radius_m == config.radius_m
        assert point.length_m == config.length_m


def test_complete_base_design_is_deterministic():
    config = _config()

    first = generate_base_design(config)
    second = generate_base_design(config)

    assert first == second


def test_split_order_and_ids_are_canonical():
    design = generate_base_design(
        _config()
    )

    assert design[0].simulation_id == (
        "base_train_000001"
    )

    assert design[4095].simulation_id == (
        "base_train_004096"
    )

    assert design[4096].simulation_id == (
        "base_validation_000001"
    )

    assert design[6143].simulation_id == (
        "base_validation_002048"
    )

    assert design[6144].simulation_id == (
        "base_test_000001"
    )

    assert design[-1].simulation_id == (
        "base_test_002048"
    )


def test_tiny_dataset_regeneration_is_exactly_deterministic():
    first, second = _tiny_regeneration_pair()

    assert len(first) == 16
    assert len(second) == 16
    assert first == second


def test_tiny_dataset_all_rows_are_valid_and_ml_eligible():
    first, _ = _tiny_regeneration_pair()

    assert all(
        row.integration_success
        for row in first
    )

    assert all(
        row.converged
        for row in first
    )

    assert all(
        row.physical_state_valid
        for row in first
    )

    assert all(
        row.balance_valid
        for row in first
    )

    assert all(
        row.qualification_valid
        for row in first
    )

    assert all(
        row.is_ml_eligible
        for row in first
    )

    assert all(
        row.model_validity_status == "valid"
        for row in first
    )


def test_tiny_dataset_physical_and_derived_outputs_are_consistent():
    first, _ = _tiny_regeneration_pair()

    for row in first:
        assert row.true_neutral_density_m3 > 0.0

        assert (
            0.0
            < row.true_electron_density_m3
            < row.true_neutral_density_m3
        )

        assert (
            row.true_electron_density_m3
            == row.true_ion_density_m3
        )

        assert (
            0.5
            < row.true_electron_temperature_eV
            < 10.0
        )

        assert (
            row.derived_pumping_speed_m3_s
            > 0.0
        )

        assert (
            0.0
            < row.ionization_fraction
            < 1.0
        )

        assert row.residence_time_proxy_s > 0.0

        assert row.collisional_power_loss_W >= 0.0
        assert row.wall_power_loss_W >= 0.0

        assert math.isfinite(
            row.pressure_target_relative_error
        )


def test_tiny_dataset_domain_status_is_supported_or_boundary():
    first, _ = _tiny_regeneration_pair()

    statuses = {
        row.domain_status
        for row in first
    }

    assert statuses <= {
        "supported",
        "near_boundary",
    }

    assert "ood" not in statuses


def test_base_schema_contains_no_monitoring_or_fault_columns():
    field_names = {
        field.name
        for field in fields(BaseSimulationRow)
    }

    assert not any(
        name.startswith("measured_")
        for name in field_names
    )

    assert not any(
        name.startswith("fault_")
        for name in field_names
    )

    assert "fault_present" not in field_names


def test_tiny_dataset_validity_flags_are_native_python_booleans():
    first, _ = _tiny_regeneration_pair()

    for row in first:
        assert type(row.integration_success) is bool
        assert type(row.converged) is bool
        assert type(row.physical_state_valid) is bool
        assert type(row.balance_valid) is bool
        assert type(row.qualification_valid) is bool
        assert type(row.is_ml_eligible) is bool
