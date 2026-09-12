"""Phase 4F one-time locked TEST evaluation.

The real evaluation path in this module is the only Phase 4F path permitted
to expose TEST targets.

Implementation and unit testing must be completed before this function is
executed against the frozen real dataset.

No TEST result may trigger model reselection, retuning, transformation
changes, feature changes, or acceptance-rule changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.data import (
    file_sha256,
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
    regression_metrics,
)
from plasma_ai.surrogate.phase4er_final_refit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_FINAL_REFIT_PATH,
    run_phase4er_final_refit,
)
from plasma_ai.surrogate.phase4f_analysis import (
    error_space_summary,
    prediction_integrity,
    worst_case_observations,
)
from plasma_ai.surrogate.phase4f_structural import (
    evaluate_phase4f_structural_diagnostics,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/locked_test_evaluation.json"
)

EXPECTED_FINAL_REFIT_SHA256 = (
    "4132ff5221c14cf6753e1a19bed5ecaa17577462fcecce1699b57c719a81aa12"
)

EXPECTED_FEATURE_NAMES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
)

EXPECTED_TARGET_NAMES = (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)

EXPECTED_FINAL_REFIT_ROWS = 6144
EXPECTED_TEST_ROWS = 2048


def validate_phase4f_preflight(
    final_refit_path: str | Path = DEFAULT_FINAL_REFIT_PATH,
) -> dict:
    """Validate frozen pre-TEST evidence before TEST-target access."""

    path = Path(
        final_refit_path
    )

    observed_hash = file_sha256(
        path
    )

    if observed_hash != EXPECTED_FINAL_REFIT_SHA256:
        raise ValueError(
            "Phase 4E-R final-refit artifact SHA-256 does not "
            "match the frozen accepted result."
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("phase") != "4E-R":
        raise ValueError(
            "Phase 4F requires the frozen Phase 4E-R final-refit artifact."
        )

    if payload.get("stage") != "final_train_validation_refit":
        raise ValueError(
            "Unexpected Phase 4E-R final-refit stage."
        )

    data_usage = payload[
        "data_usage"
    ]

    if int(
        data_usage.get(
            "final_refit_rows",
            -1,
        )
    ) != EXPECTED_FINAL_REFIT_ROWS:
        raise ValueError(
            "Frozen final refit must contain exactly 6,144 rows."
        )

    if data_usage.get(
        "test_targets_accessed"
    ) is not False:
        raise ValueError(
            "Phase 4E-R final refit must precede TEST-target access."
        )

    if data_usage.get(
        "test_evaluation_performed"
    ) is not False:
        raise ValueError(
            "Frozen final-refit artifact unexpectedly records TEST evaluation."
        )

    gates = payload[
        "gates"
    ]

    if gates.get(
        "physics_acceptance_passed"
    ) is not True:
        raise ValueError(
            "Phase 4F requires passed pre-test physics acceptance."
        )

    if gates.get(
        "final_train_validation_refit_performed"
    ) is not True:
        raise ValueError(
            "Phase 4F requires completed TRAIN+VALIDATION refit."
        )

    if gates.get(
        "final_configurations_frozen"
    ) is not True:
        raise ValueError(
            "Phase 4F requires frozen final configurations."
        )

    if gates.get(
        "locked_test_evaluation_performed"
    ) is not False:
        raise ValueError(
            "Frozen Phase 4E-R artifact indicates TEST was already evaluated."
        )

    return payload


def run_phase4f_locked_test_evaluation(
    *,
    final_refit_path: str | Path = DEFAULT_FINAL_REFIT_PATH,
) -> dict:
    """Perform the one-time locked TEST evaluation in memory.

    Calling this function against the real frozen dataset exposes TEST targets.
    """

    preflight = validate_phase4f_preflight(
        final_refit_path
    )

    final_refit = run_phase4er_final_refit()

    if final_refit.fit_rows != EXPECTED_FINAL_REFIT_ROWS:
        raise RuntimeError(
            "Final surrogate must be fitted on exactly 6,144 rows."
        )

    if final_refit.test_targets_accessed:
        raise RuntimeError(
            "Final-refit object unexpectedly records TEST-target access."
        )

    # This is the explicit and intentional one-time Phase 4F TEST unlock.
    dataset = load_phase4_dataset(
        include_test_targets=True
    )

    if tuple(
        dataset.feature_names
    ) != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Unexpected Phase 4F feature contract."
        )

    if tuple(
        dataset.target_names
    ) != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Unexpected Phase 4F target contract."
        )

    test = dataset.split(
        "test"
    )

    if test.y is None:
        raise RuntimeError(
            "Phase 4F TEST targets were not made available."
        )

    if test.X.shape != (
        EXPECTED_TEST_ROWS,
        2,
    ):
        raise RuntimeError(
            "Phase 4F TEST feature matrix must contain exactly "
            "2,048 rows and two frozen features."
        )

    if test.y.shape != (
        EXPECTED_TEST_ROWS,
        2,
    ):
        raise RuntimeError(
            "Phase 4F TEST target matrix must contain exactly "
            "2,048 rows and two frozen targets."
        )

    if not np.isfinite(
        test.X
    ).all():
        raise ValueError(
            "Phase 4F TEST features contain non-finite values."
        )

    if not np.isfinite(
        test.y
    ).all():
        raise ValueError(
            "Phase 4F TEST targets contain non-finite values."
        )

    density_truth = np.asarray(
        test.y[
            :,
            0,
        ],
        dtype=np.float64,
    )

    temperature_truth = np.asarray(
        test.y[
            :,
            1,
        ],
        dtype=np.float64,
    )

    if np.any(
        density_truth <= 0.0
    ):
        raise ValueError(
            "Phase 4F density TEST targets must be strictly positive."
        )

    if np.any(
        temperature_truth <= 0.0
    ):
        raise ValueError(
            "Phase 4F temperature TEST targets must be strictly positive."
        )

    density_prediction = (
        final_refit.density.predict_physical(
            test.X
        )
    )

    temperature_prediction = (
        final_refit.temperature.predict_physical(
            test.X
        )
    )

    expected_prediction_shape = (
        EXPECTED_TEST_ROWS,
    )

    if density_prediction.shape != expected_prediction_shape:
        raise RuntimeError(
            "Density surrogate returned unexpected TEST prediction shape."
        )

    if temperature_prediction.shape != expected_prediction_shape:
        raise RuntimeError(
            "Temperature surrogate returned unexpected TEST prediction shape."
        )

    density_integrity = prediction_integrity(
        density_prediction
    )

    temperature_integrity = prediction_integrity(
        temperature_prediction
    )

    density_metrics = regression_metrics(
        DENSITY_TARGET,
        density_truth,
        density_prediction,
    )

    temperature_metrics = regression_metrics(
        TEMPERATURE_TARGET,
        temperature_truth,
        temperature_prediction,
    )

    density_error_space = error_space_summary(
        DENSITY_TARGET,
        test.X,
        density_truth,
        density_prediction,
    )

    temperature_error_space = error_space_summary(
        TEMPERATURE_TARGET,
        test.X,
        temperature_truth,
        temperature_prediction,
    )

    density_worst = worst_case_observations(
        DENSITY_TARGET,
        test.X,
        density_truth,
        density_prediction,
    )

    temperature_worst = worst_case_observations(
        TEMPERATURE_TARGET,
        test.X,
        temperature_truth,
        temperature_prediction,
    )

    structural = evaluate_phase4f_structural_diagnostics(
        final_refit
    )

    return {
        "phase": "4F",
        "stage": "one_time_locked_test_evaluation",
        "evaluation_status": "completed",
        "irreversible_test_evaluation": True,
        "input_artifacts": {
            "phase4er_final_refit": {
                "path": str(
                    final_refit_path
                ),
                "sha256": EXPECTED_FINAL_REFIT_SHA256,
            },
        },
        "data_usage": {
            "final_fit_splits": [
                "train",
                "validation",
            ],
            "final_fit_rows": int(
                final_refit.fit_rows
            ),
            "test_rows": int(
                test.X.shape[0]
            ),
            "test_targets_accessed": True,
            "test_evaluation_performed": True,
            "test_may_affect_model_selection": False,
            "retuning_after_test_permitted": False,
        },
        "feature_contract": {
            "features": list(
                dataset.feature_names
            ),
            "feature_order_frozen": True,
        },
        "density": {
            "target": DENSITY_TARGET,
            "candidate_id": (
                final_refit
                .density
                .configuration
                .candidate_spec
                .candidate_id
            ),
            "model": (
                final_refit
                .density
                .configuration
                .candidate_spec
                .model_name
            ),
            "transform": (
                final_refit
                .density
                .configuration
                .transform_name
            ),
            "fit_rows": int(
                final_refit.density.fit_rows
            ),
            "metrics": density_metrics,
            "prediction_integrity": density_integrity,
            "error_space": density_error_space,
            "worst_case_observations": density_worst,
        },
        "temperature": {
            "target": TEMPERATURE_TARGET,
            "candidate_id": (
                final_refit
                .temperature
                .configuration
                .candidate_spec
                .candidate_id
            ),
            "model": (
                final_refit
                .temperature
                .configuration
                .candidate_spec
                .model_name
            ),
            "transform": (
                final_refit
                .temperature
                .configuration
                .transform_name
            ),
            "fit_rows": int(
                final_refit.temperature.fit_rows
            ),
            "metrics": temperature_metrics,
            "prediction_integrity": temperature_integrity,
            "error_space": temperature_error_space,
            "worst_case_observations": temperature_worst,
        },
        "structural_diagnostics": structural,
        "interpretation": {
            "performance_threshold_introduced": False,
            "test_is_final_generalisation_evidence": True,
            "test_is_development_feedback": False,
            "model_changes_permitted_from_test": False,
        },
        "preflight": {
            "phase4er_final_refit_phase": preflight[
                "phase"
            ],
            "phase4er_final_refit_stage": preflight[
                "stage"
            ],
            "pre_test_physics_acceptance_passed": True,
            "final_configuration_frozen_before_test": True,
        },
        "scientific_scope": {
            "synthetic_data": True,
            "reduced_order_argon_plasma_model": True,
            "numerically_qualified_model_envelope": True,
            "surrogate_of_reduced_order_simulator": True,
            "experimental_validation": False,
            "industrial_validation": False,
            "oipt_operating_range_claim": False,
            "absorbed_power_is_generator_rf_power": False,
            "reactive_etch_or_deposition_prediction": False,
            "wafer_scale_spatial_modelling": False,
        },
    }


def write_phase4f_locked_test_evaluation(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    final_refit_path: str | Path = DEFAULT_FINAL_REFIT_PATH,
) -> Path:
    """Perform and persist the one-time locked TEST evaluation.

    The writer deliberately refuses to overwrite an existing result artifact.
    """

    destination = Path(
        output_path
    )

    if destination.exists():
        raise FileExistsError(
            "Phase 4F locked TEST evaluation artifact already exists. "
            "Refusing to overwrite consumed TEST evidence."
        )

    payload = run_phase4f_locked_test_evaluation(
        final_refit_path=final_refit_path,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination
