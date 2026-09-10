import json
import math
from dataclasses import replace

from plasma_ai.experiments.dataset_artifacts import (
    file_sha256,
)
from plasma_ai.experiments.dataset_config import (
    load_base_dataset_config,
)
from plasma_ai.experiments.generate_monitoring_dataset import (
    build_monitoring_manifest,
    build_monitoring_summary,
    generate_monitoring_row,
    write_monitoring_artifacts,
)
from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_observation import (
    build_monitoring_observation,
)
from plasma_ai.experiments.monitoring_plan import (
    generate_monitoring_episode_plan,
)
from plasma_ai.experiments.monitoring_process import (
    simulate_monitoring_process_step,
)


MONITORING_CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)

BASE_CONFIG_PATH = (
    "configs/phase3/base_dataset.json"
)

FEATURE_MANIFEST_PATH = (
    "configs/phase3/monitoring_feature_manifest.json"
)

PHYSICS_ROOT = (
    "src/plasma_ai/physics"
)

EXPERIMENTS_ROOT = (
    "src/plasma_ai/experiments"
)


def _config():
    return load_monitoring_config(
        MONITORING_CONFIG_PATH
    )


def _base_config():
    return load_base_dataset_config(
        BASE_CONFIG_PATH
    )


def _normal_plan():
    config = _config()

    source = (
        generate_monitoring_episode_plan(
            config
        )[0]
    )

    return replace(
        source,
        nominal_absorbed_power_W=50.0,
        target_pressure_mTorr=30.0,
        nominal_flow_sccm=20.0,
        fault_present=False,
        fault_domain="none",
        fault_family="none",
        fault_profile="none",
        fault_severity="none",
        fault_direction="none",
        fault_magnitude_fraction=0.0,
        fault_onset_step=None,
    )


def _rows():
    config = _config()
    base = _base_config()
    plan = _normal_plan()

    return [
        generate_monitoring_row(
            config=config,
            base_config=base,
            plan=plan,
            step_index=step,
        )
        for step in (
            0,
            20,
        )
    ]


def test_generate_monitoring_row_is_deterministic():
    config = _config()
    base = _base_config()
    plan = _normal_plan()

    first = generate_monitoring_row(
        config=config,
        base_config=base,
        plan=plan,
        step_index=20,
    )

    second = generate_monitoring_row(
        config=config,
        base_config=base,
        plan=plan,
        step_index=20,
    )

    assert first == second


def test_monitoring_summary_reports_small_subset_correctly():
    rows = _rows()

    summary = build_monitoring_summary(
        rows
    )

    assert (
        summary["rows_total"]
        == 2
    )

    assert (
        summary["episodes_total"]
        == 1
    )

    assert (
        summary[
            "rows_qualification_valid"
        ]
        == 2
    )

    assert (
        summary[
            "rows_ml_eligible"
        ]
        == 2
    )

    assert (
        summary[
            "split_episode_counts"
        ][
            "train"
        ]
        == 1
    )

    assert (
        summary[
            "episode_fault_family_counts"
        ][
            "none"
        ]
        == 1
    )


def test_monitoring_manifest_preserves_scientific_scope():
    rows = _rows()

    manifest = (
        build_monitoring_manifest(
            config=_config(),
            base_config=_base_config(),
            rows=rows,
            config_path=(
                MONITORING_CONFIG_PATH
            ),
            base_config_path=(
                BASE_CONFIG_PATH
            ),
            feature_manifest_path=(
                FEATURE_MANIFEST_PATH
            ),
            physics_root=(
                PHYSICS_ROOT
            ),
            experiments_root=(
                EXPERIMENTS_ROOT
            ),
        )
    )

    scope = manifest[
        "scientific_scope"
    ]

    assert not scope[
        "experimental_data"
    ]

    assert not scope[
        "industrial_validation"
    ]

    assert not scope[
        "oipt_operating_range"
    ]

    assert not scope[
        "generator_power_equivalent"
    ]

    assert not scope[
        "reactive_etch_prediction"
    ]

    assert not scope[
        "compound_faults"
    ]

    assert scope[
        "synthetic_scenario_assumptions"
    ]

    assert not scope[
        "ordered_steps_are_calibrated_physical_time"
    ]


def test_monitoring_writer_produces_hash_consistent_artifacts(
    tmp_path,
):
    rows = _rows()

    dataset_path = (
        tmp_path
        / "monitoring.csv"
    )

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    summary_path = (
        tmp_path
        / "summary.json"
    )

    result = (
        write_monitoring_artifacts(
            rows=rows,
            config=_config(),
            base_config=_base_config(),
            config_path=(
                MONITORING_CONFIG_PATH
            ),
            base_config_path=(
                BASE_CONFIG_PATH
            ),
            feature_manifest_path=(
                FEATURE_MANIFEST_PATH
            ),
            physics_root=(
                PHYSICS_ROOT
            ),
            experiments_root=(
                EXPERIMENTS_ROOT
            ),
            dataset_path=(
                dataset_path
            ),
            manifest_path=(
                manifest_path
            ),
            summary_path=(
                summary_path
            ),
        )
    )

    assert dataset_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        file_sha256(
            dataset_path
        )
        == manifest[
            "provenance"
        ][
            "dataset_sha256"
        ]
    )

    assert (
        result[
            "dataset_sha256"
        ]
        == manifest[
            "provenance"
        ][
            "dataset_sha256"
        ]
    )

    assert (
        summary[
            "rows_total"
        ]
        == 2
    )


def test_failed_process_observation_is_preserved_not_raised():
    config = _config()
    base = _base_config()
    plan = _normal_plan()

    good_state = (
        simulate_monitoring_process_step(
            config,
            plan,
            20,
        )
    )

    failed_state = replace(
        good_state,
        true_neutral_density_m3=float(
            "nan"
        ),
        true_ion_density_m3=float(
            "nan"
        ),
        true_electron_density_m3=float(
            "nan"
        ),
        true_electron_energy_density_J_m3=float(
            "nan"
        ),
        true_electron_temperature_eV=float(
            "nan"
        ),
        true_pressure_mTorr=float(
            "nan"
        ),
        true_pressure_relative_deviation_from_target=float(
            "nan"
        ),
        integration_success=False,
        converged=False,
        physical_state_valid=False,
        balance_valid=False,
        model_validity_status=(
            "integration_failed"
        ),
        qualification_valid=False,
        error_message=(
            "synthetic test failure"
        ),
    )

    observation = (
        build_monitoring_observation(
            config,
            plan,
            failed_state,
        )
    )

    assert math.isnan(
        observation.measured_pressure_mTorr
    )

    from plasma_ai.experiments.monitoring_dataset import (
        build_monitoring_row,
    )

    row = build_monitoring_row(
        config=config,
        base_config=base,
        plan=plan,
        state=failed_state,
        observation=observation,
    )

    assert not row.integration_success
    assert not row.qualification_valid
    assert not row.is_ml_eligible

    assert (
        row.model_validity_status
        == "integration_failed"
    )

    assert (
        row.domain_status
        == "ood"
    )
