"""Controlled Phase 4E physics-aware pre-test acceptance experiment.

The experiment evaluates the two fixed Phase 4D validation-selected
candidates against the frozen reduced-order source-simulator probe grid.

Acceptance is based only on the pre-approved Phase 4E rules:

- finite and strictly positive physical-scale predictions;
- zero source-unsupported adjacent trend reversals;
- zero source-unsupported strict local turning points.

No TEST targets are accessed. No model selection, retuning, threshold
alteration, final TRAIN+VALIDATION refit, or production persistence occurs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4e_candidates import (
    fit_phase4e_selected_candidates,
)
from plasma_ai.surrogate.physics_acceptance import (
    check_strict_positivity,
    compare_source_referenced_oscillation,
    compare_source_referenced_trend,
)
from plasma_ai.surrogate.probe_grid import (
    build_phase4e_probe_grid,
)


DEFAULT_SOURCE_REFERENCE_PATH = Path(
    "results/phase4/source_reference_grid.json"
)

EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256 = (
    "4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a"
)


def _load_frozen_source_reference(
    source_reference_path: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and verify the frozen Phase 4E source-reference artifact."""

    path = Path(
        source_reference_path
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("phase") != "4E":
        raise ValueError(
            "Source-reference artifact must belong to Phase 4E."
        )

    if payload.get("artifact") != "source_reference_grid":
        raise ValueError(
            "Unexpected Phase 4E source-reference artifact type."
        )

    gate = payload[
        "source_reference_gate"
    ]

    if not gate.get(
        "all_targets_finite",
        False,
    ):
        raise ValueError(
            "Frozen source reference did not pass finite-target gate."
        )

    if not gate.get(
        "all_qualification_valid",
        False,
    ):
        raise ValueError(
            "Frozen source reference did not pass qualification gate."
        )

    rows = payload[
        "rows"
    ]

    if len(rows) != 1681:
        raise ValueError(
            "Phase 4E source-reference artifact must contain "
            "exactly 1681 rows."
        )

    features = np.asarray(
        [
            [
                row["nominal_absorbed_power_W"],
                row["target_pressure_mTorr"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    density = np.asarray(
        [
            row["true_electron_density_m3"]
            for row in rows
        ],
        dtype=np.float64,
    )

    temperature = np.asarray(
        [
            row["true_electron_temperature_eV"]
            for row in rows
        ],
        dtype=np.float64,
    )

    if not (
        np.isfinite(features).all()
        and np.isfinite(density).all()
        and np.isfinite(temperature).all()
    ):
        raise ValueError(
            "Frozen source-reference artifact contains "
            "non-finite numerical values."
        )

    reference_matrix = np.column_stack(
        (
            features,
            density,
            temperature,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    observed_hash = hashlib.sha256(
        reference_matrix.tobytes(
            order="C"
        )
    ).hexdigest()

    stored_hash = payload[
        "reference_array_contract"
    ][
        "sha256"
    ]

    if observed_hash != stored_hash:
        raise ValueError(
            "Source-reference numerical-array hash does not "
            "match its stored artifact contract."
        )

    if (
        observed_hash
        != EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256
    ):
        raise ValueError(
            "Source-reference numerical-array hash does not "
            "match the frozen Phase 4E reference."
        )

    return (
        features,
        density,
        temperature,
    )


def _trend_payload(
    result,
) -> dict:
    return {
        "axis": int(
            result.axis
        ),
        "total_intervals": int(
            result.total_intervals
        ),
        "unsupported_reversal_count": int(
            result.unsupported_reversal_count
        ),
        "unsupported_reversal_locations": [
            [
                int(i),
                int(j),
            ]
            for i, j
            in result.unsupported_reversal_locations
        ],
        "passed": bool(
            result.passed
        ),
    }


def _oscillation_payload(
    result,
) -> dict:
    return {
        "axis": int(
            result.axis
        ),
        "source_turning_point_count": int(
            result.source_turning_point_count
        ),
        "surrogate_turning_point_count": int(
            result.surrogate_turning_point_count
        ),
        "spurious_turning_point_count": int(
            result.spurious_turning_point_count
        ),
        "spurious_turning_point_locations": [
            [
                int(i),
                int(j),
            ]
            for i, j
            in result.spurious_turning_point_locations
        ],
        "passed": bool(
            result.passed
        ),
    }


def _positivity_payload(
    result,
) -> dict:
    return {
        "total_points": int(
            result.total_points
        ),
        "nonfinite_count": int(
            result.nonfinite_count
        ),
        "nonpositive_count": int(
            result.nonpositive_count
        ),
        "passed": bool(
            result.passed
        ),
    }


def run_phase4e_physics_acceptance(
    *,
    source_reference_path: str | Path = DEFAULT_SOURCE_REFERENCE_PATH,
) -> dict:
    """Run the frozen Phase 4E physics-aware acceptance experiment."""

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
    ) = fit_phase4e_selected_candidates()

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
        "phase": "4E",
        "stage": "physics_aware_pre_test_acceptance",
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
