import hashlib
import json
from pathlib import Path


PROTOCOL_PATH = (
    "configs/phase4/surrogate_protocol.json"
)

PROTOCOL_DOC_PATH = (
    "docs/phase4/modelling_protocol.md"
)

BASE_DATASET_PATH = (
    "data/synthetic/phase3/base_steady_state.csv"
)

MONITORING_DATASET_PATH = (
    "data/synthetic/phase3/monitoring_episodes.csv"
)

EXPECTED_BASE_SHA256 = (
    "dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc"
    "546523e9f9dd423"
)

EXPECTED_MONITORING_SHA256 = (
    "ee20fe9e6ffac4875911cba23247abeeeba236954349c1696c"
    "c1cb1ee59c7a3a"
)


def _sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _protocol():
    return json.loads(
        Path(PROTOCOL_PATH).read_text(
            encoding="utf-8"
        )
    )


def test_phase4_protocol_freezes_surrogate_experiment_contract():
    protocol = _protocol()

    assert protocol["phase"] == "4A"
    assert protocol["status"] == "frozen"

    assert protocol[
        "source_phase3_git_head"
    ] == "247374c"

    base = protocol["datasets"]["base"]

    assert base["rows"] == 8192
    assert base["splits"] == {
        "train": 4096,
        "validation": 2048,
        "test": 2048,
    }
    assert base["allow_resplit"] is False

    assert _sha256(
        BASE_DATASET_PATH
    ) == EXPECTED_BASE_SHA256

    assert base[
        "sha256"
    ] == EXPECTED_BASE_SHA256

    monitoring = protocol[
        "datasets"
    ]["monitoring"]

    assert monitoring[
        "allowed_for_surrogate_training"
    ] is False

    assert monitoring[
        "allowed_for_model_selection"
    ] is False

    assert _sha256(
        MONITORING_DATASET_PATH
    ) == EXPECTED_MONITORING_SHA256

    assert monitoring[
        "sha256"
    ] == EXPECTED_MONITORING_SHA256

    assert protocol[
        "feature_contract"
    ]["features"] == [
        "nominal_absorbed_power_W",
        "target_pressure_mTorr",
    ]

    assert protocol[
        "target_contract"
    ]["primary_targets"] == [
        "true_electron_density_m3",
        "true_electron_temperature_eV",
    ]

    data_policy = protocol[
        "data_use_policy"
    ]

    assert data_policy[
        "development_fit_split"
    ] == "train"

    assert data_policy[
        "selection_split"
    ] == "validation"

    assert data_policy[
        "test_locked_during_development"
    ] is True

    assert data_policy[
        "test_may_affect_model_selection"
    ] is False

    assert data_policy[
        "final_refit_splits"
    ] == [
        "train",
        "validation",
    ]

    assert data_policy[
        "final_refit_rows"
    ] == 6144

    assert data_policy[
        "final_test_rows"
    ] == 2048

    assert data_policy[
        "retuning_after_test"
    ] is False

    physics = protocol[
        "physics_validation"
    ]

    assert physics[
        "probe_grid"
    ]["total_points"] == 1681

    assert physics[
        "source_simulator_evaluated_first"
    ] is True

    assert physics[
        "probe_points_may_enter_training"
    ] is False

    assert protocol[
        "candidate_models"
    ]["mlp"]["enabled"] is False

    doc = Path(
        PROTOCOL_DOC_PATH
    ).read_text(
        encoding="utf-8"
    )

    frozen_status = (
        "**Status:** FROZEN "
        "\u2014 PHASE 4A PROTOCOL"
    )

    assert frozen_status in doc

    assert "?" not in doc
