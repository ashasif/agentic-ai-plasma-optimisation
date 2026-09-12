"""Phase 4E-R physics-aware pre-test acceptance experiment.

The experiment evaluates:

- the frozen Phase 4E-R validation-selected monotonic density surrogate;
- the unchanged frozen Phase 4D temperature surrogate.

Both candidates are fitted on TRAIN only and evaluated against the exact
frozen Phase 4E source-reference probe grid using the unchanged Phase 4E
acceptance rules.

No TEST targets are accessed. No TRAIN+VALIDATION refit or Phase 4F
evaluation occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4er_candidates import (
    DEFAULT_PHASE4ER_SELECTION_PATH,
    EXPECTED_PHASE4ER_SELECTION_SHA256,
    fit_phase4er_acceptance_candidates,
)
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_EARLY_STOPPING,
    PHASE4ER_MONOTONIC_CST,
)
from plasma_ai.surrogate.physics_acceptance import (
    check_strict_positivity,
    compare_source_referenced_oscillation,
    compare_source_referenced_trend,
)
from plasma_ai.surrogate.physics_acceptance_experiment import (
    DEFAULT_SOURCE_REFERENCE_PATH,
    EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256,
    _load_frozen_source_reference,
    _oscillation_payload,
    _positivity_payload,
    _trend_payload,
)
from plasma_ai.surrogate.probe_grid import (
    build_phase4e_probe_grid,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/phase4er_physics_acceptance.json"
)


def run_phase4er_physics_acceptance(
    *,
    source_reference_path: str | Path = DEFAULT_SOURCE_REFERENCE_PATH,
    phase4er_selection_path: str | Path = DEFAULT_PHASE4ER_SELECTION_PATH,
) -> dict:
    """Run frozen Phase 4E-R physics-aware acceptance."""

    grid = build_phase4e_probe_grid()

    (
        source_features,
        source_density,
        source_temperature,
    ) = _load_frozen_source_reference(
        source_reference_path
    )

    if not np.array_equal(
        source_features,
        grid.features,
    ):
        raise ValueError(
            "Frozen source-reference feature ordering does not "
            "match the Phase 4E probe grid."
        )

    (
        density_model,
        temperature_model,
    ) = fit_phase4er_acceptance_candidates(
        phase4er_selection_path=phase4er_selection_path,
    )

    density_prediction = (
        density_model.predict_physical(
            grid.features
        )
    )

    temperature_prediction = (
        temperature_model.predict_physical(
            grid.features
        )
    )

    expected_shape = (
        grid.total_points,
    )

    if density_prediction.shape != expected_shape:
        raise RuntimeError(
            "Density candidate returned unexpected probe-grid shape."
        )

    if temperature_prediction.shape != expected_shape:
        raise RuntimeError(
            "Temperature candidate returned unexpected probe-grid shape."
        )

    shape = (
        grid.absorbed_power_W.size,
        grid.target_pressure_mTorr.size,
    )

    source_density_grid = (
        source_density.reshape(
            shape
        )
    )

    source_temperature_grid = (
        source_temperature.reshape(
            shape
        )
    )

    density_prediction_grid = (
        density_prediction.reshape(
            shape
        )
    )

    temperature_prediction_grid = (
        temperature_prediction.reshape(
            shape
        )
    )

    density_positivity = (
        check_strict_positivity(
            density_prediction
        )
    )

    temperature_positivity = (
        check_strict_positivity(
            temperature_prediction
        )
    )

    density_power_trend = (
        compare_source_referenced_trend(
            source_density_grid,
            density_prediction_grid,
            axis=0,
        )
    )

    density_pressure_trend = (
        compare_source_referenced_trend(
            source_density_grid,
            density_prediction_grid,
            axis=1,
        )
    )

    temperature_pressure_trend = (
        compare_source_referenced_trend(
            source_temperature_grid,
            temperature_prediction_grid,
            axis=1,
        )
    )

    density_power_oscillation = (
        compare_source_referenced_oscillation(
            source_density_grid,
            density_prediction_grid,
            axis=0,
        )
    )

    density_pressure_oscillation = (
        compare_source_referenced_oscillation(
            source_density_grid,
            density_prediction_grid,
            axis=1,
        )
    )

    temperature_pressure_oscillation = (
        compare_source_referenced_oscillation(
            source_temperature_grid,
            temperature_prediction_grid,
            axis=1,
        )
    )

    density_pass = all(
        (
            density_positivity.passed,
            density_power_trend.passed,
            density_pressure_trend.passed,
            density_power_oscillation.passed,
            density_pressure_oscillation.passed,
        )
    )

    temperature_pass = all(
        (
            temperature_positivity.passed,
            temperature_pressure_trend.passed,
            temperature_pressure_oscillation.passed,
        )
    )

    overall_pass = (
        density_pass
        and temperature_pass
    )

    return {
        "phase": "4E-R",
        "stage": "physics_aware_pre_test_acceptance",
        "phase4f_status": "locked",
        "validation_selection": {
            "path": str(
                phase4er_selection_path
            ),
            "sha256": (
                EXPECTED_PHASE4ER_SELECTION_SHA256
            ),
            "frozen_before_acceptance": True,
        },
        "source_reference": {
            "path": str(
                source_reference_path
            ),
            "array_sha256": (
                EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256
            ),
            "total_points": int(
                grid.total_points
            ),
            "evaluated_before_surrogate": True,
        },
        "data_usage": {
            "fit_split": "train",
            "fit_rows": int(
                density_model.fit_rows
            ),
            "test_targets_accessed": False,
            "validation_used_for_fitting": False,
            "validation_used_for_acceptance": False,
            "train_validation_refit_performed": False,
        },
        "density": {
            "target": DENSITY_TARGET,
            "transform": (
                density_model.configuration.transform_name
            ),
            "candidate_id": (
                density_model
                .configuration
                .candidate_spec
                .candidate_id
            ),
            "model": (
                density_model
                .configuration
                .candidate_spec
                .model_name
            ),
            "parameters": dict(
                density_model
                .configuration
                .candidate_spec
                .parameters
            ),
            "fixed_configuration": {
                "monotonic_cst": list(
                    PHASE4ER_MONOTONIC_CST
                ),
                "early_stopping": (
                    PHASE4ER_EARLY_STOPPING
                ),
            },
            "prediction_minimum": float(
                np.min(
                    density_prediction
                )
            ),
            "prediction_maximum": float(
                np.max(
                    density_prediction
                )
            ),
            "positivity": _positivity_payload(
                density_positivity
            ),
            "trends": {
                "vs_power_at_fixed_pressure": (
                    _trend_payload(
                        density_power_trend
                    )
                ),
                "vs_pressure_at_fixed_power": (
                    _trend_payload(
                        density_pressure_trend
                    )
                ),
            },
            "oscillation": {
                "vs_power_at_fixed_pressure": (
                    _oscillation_payload(
                        density_power_oscillation
                    )
                ),
                "vs_pressure_at_fixed_power": (
                    _oscillation_payload(
                        density_pressure_oscillation
                    )
                ),
            },
            "passed": bool(
                density_pass
            ),
        },
        "temperature": {
            "target": TEMPERATURE_TARGET,
            "configuration_source": (
                "frozen Phase 4D selection"
            ),
            "transform": (
                temperature_model.configuration.transform_name
            ),
            "candidate_id": (
                temperature_model
                .configuration
                .candidate_spec
                .candidate_id
            ),
            "model": (
                temperature_model
                .configuration
                .candidate_spec
                .model_name
            ),
            "parameters": dict(
                temperature_model
                .configuration
                .candidate_spec
                .parameters
            ),
            "prediction_minimum": float(
                np.min(
                    temperature_prediction
                )
            ),
            "prediction_maximum": float(
                np.max(
                    temperature_prediction
                )
            ),
            "positivity": _positivity_payload(
                temperature_positivity
            ),
            "trends": {
                "vs_pressure_at_fixed_power": (
                    _trend_payload(
                        temperature_pressure_trend
                    )
                ),
            },
            "oscillation": {
                "vs_pressure_at_fixed_power": (
                    _oscillation_payload(
                        temperature_pressure_oscillation
                    )
                ),
            },
            "passed": bool(
                temperature_pass
            ),
        },
        "acceptance": {
            "allowed_unsupported_reversals": 0,
            "allowed_spurious_turning_points": 0,
            "density_passed": bool(
                density_pass
            ),
            "temperature_passed": bool(
                temperature_pass
            ),
            "overall_passed": bool(
                overall_pass
            ),
            "final_train_validation_refit_allowed": bool(
                overall_pass
            ),
        },
        "gates": {
            "physics_acceptance_performed": True,
            "final_train_validation_refit_allowed": bool(
                overall_pass
            ),
            "final_train_validation_refit_performed": False,
            "locked_test_evaluation_performed": False,
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


def write_phase4er_physics_acceptance(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    source_reference_path: str | Path = DEFAULT_SOURCE_REFERENCE_PATH,
    phase4er_selection_path: str | Path = DEFAULT_PHASE4ER_SELECTION_PATH,
) -> Path:
    """Run Phase 4E-R acceptance and write its structured artifact."""

    destination = Path(
        output_path
    )

    payload = run_phase4er_physics_acceptance(
        source_reference_path=source_reference_path,
        phase4er_selection_path=phase4er_selection_path,
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
