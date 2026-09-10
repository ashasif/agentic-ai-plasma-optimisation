from dataclasses import replace
import re

from plasma_ai.experiments.base_dataset import (
    generate_base_design,
    simulate_base_design_point,
)
from plasma_ai.experiments.dataset_artifacts import (
    APPROVED_BASE_FEATURES,
    APPROVED_PRIMARY_TARGETS,
    base_dataset_sha256,
    canonical_base_csv_bytes,
    config_sha256,
    load_feature_manifest,
    physics_source_sha256,
)
from plasma_ai.experiments.dataset_config import (
    SplitConfig,
    load_base_dataset_config,
)


CONFIG_PATH = "configs/phase3/base_dataset.json"
FEATURE_MANIFEST_PATH = (
    "configs/phase3/base_feature_manifest.json"
)
PHYSICS_ROOT = "src/plasma_ai/physics"


def _config():
    return load_base_dataset_config(
        CONFIG_PATH
    )


def test_feature_manifest_freezes_approved_features_and_targets():
    manifest = load_feature_manifest(
        FEATURE_MANIFEST_PATH
    )

    assert tuple(
        manifest["features"]
    ) == APPROVED_BASE_FEATURES

    assert tuple(
        manifest["primary_targets"]
    ) == APPROVED_PRIMARY_TARGETS

    assert set(
        manifest["features"]
    ).isdisjoint(
        manifest["primary_targets"]
    )


def test_config_hash_is_deterministic_and_semantic():
    config = _config()

    first = config_sha256(config)
    second = config_sha256(config)

    assert first == second
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        first,
    )


def test_physics_source_hash_is_deterministic():
    first = physics_source_sha256(
        PHYSICS_ROOT
    )

    second = physics_source_sha256(
        PHYSICS_ROOT
    )

    assert first == second

    assert re.fullmatch(
        r"[0-9a-f]{64}",
        first,
    )


def test_canonical_csv_and_dataset_hash_are_deterministic():
    config = _config()

    one_row_config = replace(
        config,
        splits={
            "train": SplitConfig(
                rows=1,
                seed=20260910,
            ),
            "validation": SplitConfig(
                rows=1,
                seed=20260911,
            ),
            "test": SplitConfig(
                rows=1,
                seed=20260912,
            ),
        },
    )

    point = generate_base_design(
        one_row_config
    )[0]

    row = simulate_base_design_point(
        point,
        one_row_config,
    )

    assert row.qualification_valid
    assert row.is_ml_eligible

    first_bytes = canonical_base_csv_bytes(
        [row]
    )

    second_bytes = canonical_base_csv_bytes(
        [row]
    )

    assert first_bytes == second_bytes

    header = first_bytes.decode(
        "utf-8"
    ).splitlines()[0]

    assert header.startswith(
        "simulation_id,split,split_index,design_seed,"
    )

    first_hash = base_dataset_sha256(
        [row]
    )

    second_hash = base_dataset_sha256(
        [row]
    )

    assert first_hash == second_hash

    assert re.fullmatch(
        r"[0-9a-f]{64}",
        first_hash,
    )


def test_base_feature_manifest_contains_no_diagnostics_as_features():
    manifest = load_feature_manifest(
        FEATURE_MANIFEST_PATH
    )

    features = set(
        manifest["features"]
    )

    forbidden = {
        "simulation_id",
        "split",
        "design_seed",
        "converged",
        "qualification_valid",
        "is_ml_eligible",
        "max_relative_rate_s",
        "max_balance_residual",
        "error_message",
        "true_electron_density_m3",
        "true_electron_temperature_eV",
    }

    assert features.isdisjoint(
        forbidden
    )
