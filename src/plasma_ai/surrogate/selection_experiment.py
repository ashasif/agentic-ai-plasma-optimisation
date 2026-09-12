"""Controlled Phase 4D validation-based surrogate selection.

Phase 4D evaluates the complete frozen classical candidate grid using
TRAIN for fitting and VALIDATION for model/transform selection.

TEST targets remain withheld throughout this phase.

The selected candidates remain validation-selected candidates only.
Physics-aware pre-test acceptance belongs to Phase 4E, locked TEST
evaluation belongs to Phase 4F, and production persistence belongs to
Phase 4G.
"""

from __future__ import annotations

import json
from pathlib import Path

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
    fit_classical_candidate,
)
from plasma_ai.surrogate.data import (
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.selection import (
    ValidationSelectionDecision,
    enumerate_phase4d_classical_specs,
    select_validation_candidate,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/validation_selection.json"
)

PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION = 0.02

PHASE4D_TARGET_TRANSFORMS = (
    (
        DENSITY_TARGET,
        "identity",
    ),
    (
        DENSITY_TARGET,
        "log10",
    ),
    (
        TEMPERATURE_TARGET,
        "identity",
    ),
)


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
        "metrics": {
            name: float(value)
            for name, value
            in result.metrics.items()
        },
        "prediction_sanity": dict(
            result.prediction_sanity
        ),
    }


def _decision_to_payload(
    decision: ValidationSelectionDecision,
    *,
    candidate_pool_size: int,
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
        "selected_metrics": {
            name: float(value)
            for name, value
            in selected.metrics.items()
        },
        "selection_stage": decision.selection_stage,
        "primary_metric": decision.primary_metric,
        "secondary_metrics": list(
            decision.secondary_metrics
        ),
        "primary_equivalent_candidate_ids": list(
            decision.primary_equivalent_candidate_ids
        ),
        "primary_equivalent_candidates": [
            dict(candidate)
            for candidate
            in decision.primary_equivalent_candidates
        ],
        "best_primary_value": float(
            decision.best_primary_value
        ),
        "practical_equivalence_relative_fraction": float(
            decision.practical_equivalence_relative_fraction
        ),
        "primary_strict_upper_boundary": float(
            decision.primary_strict_upper_boundary
        ),
        "candidate_pool_size": int(
            candidate_pool_size
        ),
        "practical_equivalence_relative_fraction": (
            PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
        ),
        "validation_selected_only": True,
        "phase4e_physics_acceptance_pending": True,
        "locked_test_evaluation_performed": False,
    }


def run_validation_selection() -> dict:
    """Run the frozen Phase 4D TRAIN/VALIDATION search and selection."""
    dataset = load_phase4_dataset()

    train = dataset.split("train")
    validation = dataset.split("validation")
    test = dataset.split("test")

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
            "TEST targets must remain locked in Phase 4D."
        )

    target_index = {
        DENSITY_TARGET: 0,
        TEMPERATURE_TARGET: 1,
    }

    specs = enumerate_phase4d_classical_specs()

    if len(specs) != 36:
        raise RuntimeError(
            "Phase 4D requires exactly 36 frozen "
            "classical configurations."
        )

    validation_results: list[
        ClassicalValidationResult
    ] = []

    for (
        target_name,
        transform_name,
    ) in PHASE4D_TARGET_TRANSFORMS:
        index = target_index[
            target_name
        ]

        for spec in specs:
            result = fit_classical_candidate(
                train_X=train.X,
                train_y=train.y[:, index],
                validation_X=validation.X,
                validation_y=validation.y[:, index],
                target_name=target_name,
                transform_name=transform_name,
                spec=spec,
            )

            validation_results.append(
                result
            )

    expected_fits = (
        len(specs)
        * len(PHASE4D_TARGET_TRANSFORMS)
    )

    if len(validation_results) != expected_fits:
        raise RuntimeError(
            "Phase 4D controlled search produced "
            f"{len(validation_results)} results; "
            f"expected {expected_fits}."
        )

    density_results = tuple(
        result
        for result in validation_results
        if result.target_name == DENSITY_TARGET
    )

    temperature_results = tuple(
        result
        for result in validation_results
        if result.target_name == TEMPERATURE_TARGET
    )

    if len(density_results) != 72:
        raise RuntimeError(
            "Phase 4D density selection pool must "
            f"contain 72 candidates; observed "
            f"{len(density_results)}."
        )

    if len(temperature_results) != 36:
        raise RuntimeError(
            "Phase 4D temperature selection pool must "
            f"contain 36 candidates; observed "
            f"{len(temperature_results)}."
        )

    density_decision = select_validation_candidate(
        density_results,
        target_name=DENSITY_TARGET,
        practical_equivalence_relative_fraction=(
            PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
        ),
    )

    temperature_decision = select_validation_candidate(
        temperature_results,
        target_name=TEMPERATURE_TARGET,
        practical_equivalence_relative_fraction=(
            PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
        ),
    )

    return {
        "phase": "4D",
        "experiment": (
            "controlled_validation_selection"
        ),
        "scientific_scope": {
            "data_type": "synthetic data",
            "source_model": (
                "reduced-order argon plasma model"
            ),
            "domain": (
                "numerically qualified model envelope"
            ),
            "surrogate_role": (
                "surrogate of the reduced-order simulator"
            ),
            "experimental_validation": False,
            "industrial_validation": False,
            "oipt_operating_range_claim": False,
            "reactive_etch_or_deposition_prediction": False,
            "wafer_scale_spatial_modelling": False,
            "absorbed_power_is_generator_rf_power": False,
        },
        "data_usage": {
            "fit_split": "train",
            "evaluation_split": "validation",
            "test_targets_accessed": False,
            "train_rows": int(
                train.X.shape[0]
            ),
            "validation_rows": int(
                validation.X.shape[0]
            ),
            "test_rows": int(
                test.X.shape[0]
            ),
        },
        "selection_design": {
            "classical_configurations_per_target_transform": int(
                len(specs)
            ),
            "target_transform_combinations": int(
                len(PHASE4D_TARGET_TRANSFORMS)
            ),
            "total_fits": int(
                len(validation_results)
            ),
            "practical_equivalence_relative_fraction": (
                PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
            ),
            "validation_selection_performed": True,
            "physics_acceptance_performed": False,
            "locked_test_evaluation_performed": False,
            "train_validation_refit_performed": False,
            "production_model_persisted": False,
        },
        "selection": {
            "density": _decision_to_payload(
                density_decision,
                candidate_pool_size=len(
                    density_results
                ),
            ),
            "temperature": _decision_to_payload(
                temperature_decision,
                candidate_pool_size=len(
                    temperature_results
                ),
            ),
        },
        "results": [
            _result_to_payload(result)
            for result in validation_results
        ],
    }


def write_validation_selection(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Run Phase 4D and write its structured selection artifact."""
    destination = Path(
        output_path
    )

    payload = run_validation_selection()

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
