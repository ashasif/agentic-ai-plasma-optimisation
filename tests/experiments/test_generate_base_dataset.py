from dataclasses import replace
import json

from plasma_ai.experiments.base_dataset import (
    generate_base_design,
    simulate_base_design_point,
)
from plasma_ai.experiments.dataset_artifacts import (
    file_sha256,
)
from plasma_ai.experiments.dataset_config import (
    SplitConfig,
    load_base_dataset_config,
)
from plasma_ai.experiments.generate_base_dataset import (
    build_base_manifest,
    build_base_summary,
    write_base_artifacts,
)


CONFIG_PATH = "configs/phase3/base_dataset.json"
FEATURE_MANIFEST_PATH = (
    "configs/phase3/base_feature_manifest.json"
)
PHYSICS_ROOT = "src/plasma_ai/physics"


def _one_row():
    config = load_base_dataset_config(
        CONFIG_PATH
    )

    tiny = replace(
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
        tiny
    )[0]

    row = simulate_base_design_point(
        point,
        tiny,
    )

    return tiny, row


def test_base_summary_reports_valid_single_row():
    _, row = _one_row()

    summary = build_base_summary(
        [row]
    )

    assert summary["rows_total"] == 1
    assert summary[
        "rows_qualification_valid"
    ] == 1
    assert summary[
        "rows_ml_eligible"
    ] == 1
    assert summary["rows_invalid"] == 0


def test_base_manifest_contains_scientific_scope_and_hashes():
    config, row = _one_row()

    manifest = build_base_manifest(
        config=config,
        rows=[row],
        config_path=CONFIG_PATH,
        feature_manifest_path=(
            FEATURE_MANIFEST_PATH
        ),
        physics_root=PHYSICS_ROOT,
    )

    assert (
        manifest["scientific_scope"][
            "experimental_data"
        ]
        is False
    )

    assert (
        manifest["scientific_scope"][
            "industrial_validation"
        ]
        is False
    )

    assert len(
        manifest["provenance"][
            "dataset_sha256"
        ]
    ) == 64

    assert len(
        manifest["provenance"][
            "physics_source_sha256"
        ]
    ) == 64


def test_write_base_artifacts_matches_written_dataset_hash(
    tmp_path,
):
    config, row = _one_row()

    dataset_path = (
        tmp_path
        / "base.csv"
    )

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    summary_path = (
        tmp_path
        / "summary.json"
    )

    result = write_base_artifacts(
        rows=[row],
        config=config,
        config_path=CONFIG_PATH,
        feature_manifest_path=(
            FEATURE_MANIFEST_PATH
        ),
        physics_root=PHYSICS_ROOT,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )

    assert dataset_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    assert (
        file_sha256(
            dataset_path
        )
        == result["dataset_sha256"]
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["provenance"][
            "dataset_sha256"
        ]
        == result["dataset_sha256"]
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary[
        "rows_total"
    ] == 1
