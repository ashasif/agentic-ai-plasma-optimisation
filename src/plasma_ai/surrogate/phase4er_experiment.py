"""Phase 4E-R constrained density TRAIN/VALIDATION orchestration.

This module executes the separately predeclared Phase 4E-R density
redevelopment search using TRAIN for fitting and VALIDATION for selection.

TEST targets must remain withheld throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)
from plasma_ai.surrogate.data import (
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    PHASE4ER_EARLY_STOPPING,
    PHASE4ER_MONOTONIC_CST,
    enumerate_phase4er_density_specs,
    fit_phase4er_density_candidate,
)
from plasma_ai.surrogate.phase4er_selection import (
    PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION,
    select_phase4er_density_candidate,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/phase4er_validation_selection.json"
)

EXPECTED_FEATURE_NAMES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
)

EXPECTED_TARGET_NAMES = (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)

EXPECTED_TRAIN_ROWS = 4096
EXPECTED_VALIDATION_ROWS = 2048
EXPECTED_TEST_ROWS = 2048
EXPECTED_CANDIDATE_COUNT = 16


def _result_to_payload(
    result: ClassicalValidationResult,
) -> dict:
    return {
        "target": result.target_name,
        "transform": result.transform_name,
        "candidate_id": result.candidate_id,
        "model": result.model_name,
        "parameters": dict(
            result.parameters
        ),
        "metrics": dict(
            result.metrics
        ),
        "prediction_sanity": dict(
            result.prediction_sanity
        ),
    }


def _decision_to_payload(
    decision,
) -> dict:
    selected = decision.selected_result

    return {
        "target": selected.target_name,
        "selected_transform": selected.transform_name,
        "selected_candidate_id": selected.candidate_id,
        "selected_model": selected.model_name,
        "selected_parameters": dict(
            selected.parameters
        ),
        "selected_metrics": dict(
            selected.metrics
        ),
        "selected_prediction_sanity": dict(
            selected.prediction_sanity
        ),
        "selection_stage": decision.selection_stage,
        "primary_metric": decision.primary_metric,
        "secondary_metrics": list(
            decision.secondary_metrics
        ),
        "primary_equivalent_candidate_ids": list(
            decision.primary_equivalent_candidate_ids
        ),
        "primary_equivalent_candidates": list(
            decision.primary_equivalent_candidates
        ),
        "best_primary_value": (
            decision.best_primary_value
        ),
        "practical_equivalence_relative_fraction": (
            decision.practical_equivalence_relative_fraction
        ),
        "primary_strict_upper_boundary": (
            decision.primary_strict_upper_boundary
        ),
        "validation_selected_only": True,
        "physics_acceptance_pending": True,
        "locked_test_evaluation_performed": False,
    }


def run_phase4er_validation_selection() -> dict:
    """Run the frozen Phase 4E-R density TRAIN/VALIDATION search."""

    dataset = load_phase4_dataset()

    train = dataset.split(
        "train"
    )
    validation = dataset.split(
        "validation"
    )
    test = dataset.split(
        "test"
    )

    if train.y is None:
        raise RuntimeError(
            "TRAIN targets are unavailable."
        )

    if validation.y is None:
        raise RuntimeError(
            "VALIDATION targets are unavailable."
        )

    if test.y is not None:
        raise RuntimeError(
            "TEST targets must remain locked during Phase 4E-R."
        )

    if tuple(
        dataset.feature_names
    ) != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Unexpected Phase 4E-R feature contract."
        )

    if tuple(
        dataset.target_names
    ) != EXPECTED_TARGET_NAMES:
        raise RuntimeError(
            "Unexpected Phase 4E-R target contract."
        )

    observed_rows = {
        "train": int(
            train.X.shape[0]
        ),
        "validation": int(
            validation.X.shape[0]
        ),
        "test": int(
            test.X.shape[0]
        ),
    }

    expected_rows = {
        "train": EXPECTED_TRAIN_ROWS,
        "validation": EXPECTED_VALIDATION_ROWS,
        "test": EXPECTED_TEST_ROWS,
    }

    if observed_rows != expected_rows:
        raise RuntimeError(
            "Unexpected frozen Phase 4 split sizes: "
            f"expected {expected_rows}, "
            f"observed {observed_rows}."
        )

    density_index = (
        dataset.target_names.index(
            DENSITY_TARGET
        )
    )

    specs = (
        enumerate_phase4er_density_specs()
    )

    if len(specs) != EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            "Phase 4E-R must evaluate exactly "
            f"{EXPECTED_CANDIDATE_COUNT} density candidates."
        )

    validation_results = tuple(
        fit_phase4er_density_candidate(
            train_X=train.X,
            train_y=train.y[
                :,
                density_index,
            ],
            validation_X=validation.X,
            validation_y=validation.y[
                :,
                density_index,
            ],
            spec=spec,
        )
        for spec in specs
    )

    if (
        len(validation_results)
        != EXPECTED_CANDIDATE_COUNT
    ):
        raise RuntimeError(
            "Phase 4E-R density search produced "
            f"{len(validation_results)} results; "
            f"expected {EXPECTED_CANDIDATE_COUNT}."
        )

    density_decision = (
        select_phase4er_density_candidate(
            validation_results
        )
    )

    return {
        "phase": "4E-R",
        "stage": (
            "pre_test_physics_constrained_"
            "density_redevelopment"
        ),
        "protocol_status": "predeclared_and_frozen",
        "phase4f_status": "locked",
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
        "data_usage": {
            "fit_split": "train",
            "evaluation_split": "validation",
            "train_rows": observed_rows[
                "train"
            ],
            "validation_rows": observed_rows[
                "validation"
            ],
            "test_rows": observed_rows[
                "test"
            ],
            "test_targets_accessed": False,
            "train_validation_refit_performed": False,
        },
        "density_redevelopment": {
            "target": DENSITY_TARGET,
            "transform": PHASE4ER_DENSITY_TRANSFORM,
            "candidate_count": len(
                validation_results
            ),
            "monotonic_cst": list(
                PHASE4ER_MONOTONIC_CST
            ),
            "early_stopping": (
                PHASE4ER_EARLY_STOPPING
            ),
            "practical_equivalence_relative_fraction": (
                PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
            ),
            "selection": _decision_to_payload(
                density_decision
            ),
            "results": [
                _result_to_payload(
                    result
                )
                for result in validation_results
            ],
        },
        "temperature": {
            "redeveloped": False,
            "configuration_source": (
                "frozen Phase 4D selection"
            ),
            "physics_acceptance_required": True,
        },
        "gates": {
            "physics_acceptance_performed": False,
            "final_train_validation_refit_allowed": False,
            "final_train_validation_refit_performed": False,
            "locked_test_evaluation_performed": False,
            "production_model_persisted": False,
        },
    }


def write_phase4er_validation_selection(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Run Phase 4E-R and write its structured validation artifact."""

    destination = Path(
        output_path
    )

    payload = (
        run_phase4er_validation_selection()
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
