from dataclasses import replace

import pytest

from plasma_ai.experiments.dataset_config import (
    load_base_dataset_config,
)
from plasma_ai.experiments.monitoring_artifacts import (
    canonical_monitoring_csv_bytes,
    load_monitoring_feature_manifest,
    monitoring_config_sha256,
    monitoring_dataset_sha256,
)
from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_dataset import (
    APPROVED_BINARY_TARGET,
    APPROVED_MONITORING_FEATURES,
    APPROVED_MULTICLASS_TARGET,
    build_monitoring_row,
    classify_true_domain_status,
    monitoring_schema_fields,
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


def _monitoring_config():
    return load_monitoring_config(
        MONITORING_CONFIG_PATH
    )


def _base_config():
    return load_base_dataset_config(
        BASE_CONFIG_PATH
    )


def _normal_plan():
    config = _monitoring_config()

    source = generate_monitoring_episode_plan(
        config
    )[0]

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


def _build_row(
    plan,
    step=20,
):
    config = _monitoring_config()

    state = simulate_monitoring_process_step(
        config,
        plan,
        step,
    )

    observation = build_monitoring_observation(
        config,
        plan,
        state,
    )

    return build_monitoring_row(
        config=config,
        base_config=_base_config(),
        plan=plan,
        state=state,
        observation=observation,
    )


def test_monitoring_schema_contains_approved_features_and_targets():
    fields = set(
        monitoring_schema_fields()
    )

    assert set(
        APPROVED_MONITORING_FEATURES
    ) <= fields

    assert (
        APPROVED_BINARY_TARGET
        in fields
    )

    assert (
        APPROVED_MULTICLASS_TARGET
        in fields
    )


def test_feature_manifest_is_leakage_safe_and_valid():
    manifest = (
        load_monitoring_feature_manifest(
            FEATURE_MANIFEST_PATH
        )
    )

    assert tuple(
        manifest["features"]
    ) == APPROVED_MONITORING_FEATURES

    assert (
        manifest["binary_target"]
        == APPROVED_BINARY_TARGET
    )

    assert (
        manifest["multiclass_target"]
        == APPROVED_MULTICLASS_TARGET
    )

    features = set(
        manifest["features"]
    )

    assert not any(
        name.startswith(
            "true_"
        )
        for name in features
    )

    assert not any(
        "fault" in name
        for name in features
    )

    assert not any(
        "noise" in name
        for name in features
    )

    assert (
        "step_index"
        not in features
    )

    assert (
        "episode_id"
        not in features
    )


def test_true_domain_classifier_uses_phase3b_envelope():
    base = _base_config()

    assert classify_true_domain_status(
        true_absorbed_power_W=50.0,
        true_pressure_mTorr=30.0,
        base_config=base,
    ) == "supported"

    assert classify_true_domain_status(
        true_absorbed_power_W=15.1,
        true_pressure_mTorr=30.0,
        base_config=base,
    ) == "near_boundary"

    assert classify_true_domain_status(
        true_absorbed_power_W=91.0,
        true_pressure_mTorr=30.0,
        base_config=base,
    ) == "ood"

    assert classify_true_domain_status(
        true_absorbed_power_W=50.0,
        true_pressure_mTorr=61.0,
        base_config=base,
    ) == "ood"


def test_normal_monitoring_row_has_no_active_fault():
    row = _build_row(
        _normal_plan(),
        step=20,
    )

    assert not row.fault_present
    assert not row.fault_started
    assert not row.fault_effect_active

    assert (
        row.active_fault_domain
        == "none"
    )

    assert (
        row.active_fault_family
        == "none"
    )

    assert row.qualification_valid
    assert row.is_ml_eligible

    assert row.domain_status in {
        "supported",
        "near_boundary",
    }


def test_drift_onset_distinguishes_started_from_nonzero_effect():
    base = _normal_plan()

    plan = replace(
        base,
        fault_present=True,
        fault_domain="process",
        fault_family="power_coupling",
        fault_profile="drift",
        fault_severity="severe",
        fault_direction="negative",
        fault_magnitude_fraction=0.20,
        fault_onset_step=16,
    )

    onset = _build_row(
        plan,
        step=16,
    )

    after = _build_row(
        plan,
        step=17,
    )

    assert onset.fault_started
    assert onset.fault_progress == 0.0
    assert onset.signed_fault_fraction == 0.0
    assert not onset.fault_effect_active
    assert onset.active_fault_family == "none"

    assert after.fault_started
    assert after.fault_progress > 0.0
    assert after.signed_fault_fraction < 0.0
    assert after.fault_effect_active
    assert (
        after.active_fault_family
        == "power_coupling"
    )


def test_monitoring_serialization_is_canonical_and_deterministic():
    row = _build_row(
        _normal_plan(),
        step=20,
    )

    first = canonical_monitoring_csv_bytes(
        [
            row,
        ]
    )

    second = canonical_monitoring_csv_bytes(
        [
            row,
        ]
    )

    assert first == second

    text = first.decode(
        "utf-8"
    )

    assert "\r\n" not in text

    assert "true" in text
    assert "false" in text
    assert "True" not in text
    assert "False" not in text

    assert monitoring_dataset_sha256(
        [
            row,
        ]
    ) == monitoring_dataset_sha256(
        [
            row,
        ]
    )


def test_monitoring_config_semantic_hash_is_deterministic():
    first = monitoring_config_sha256(
        _monitoring_config()
    )

    second = monitoring_config_sha256(
        _monitoring_config()
    )

    assert first == second
    assert len(first) == 64

    int(
        first,
        16,
    )
