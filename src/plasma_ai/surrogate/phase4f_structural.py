"""Phase 4F structural diagnostics for the locked final surrogate.

This module evaluates the final TRAIN+VALIDATION-refitted surrogate on the
already-frozen Phase 4 probe grid.

It reuses the Phase 4E-R source-reference trend, oscillation, and positivity
rules exactly. It does not load or access TEST targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4er_final_refit import (
    Phase4ERFinalRefit,
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


def evaluate_phase4f_structural_diagnostics(
    final_refit: Phase4ERFinalRefit,
    *,
    source_reference_path: str | Path = DEFAULT_SOURCE_REFERENCE_PATH,
) -> dict:
    """Evaluate frozen structural diagnostics after final 6,144-row refit."""

    if final_refit.fit_rows != 6144:
        raise ValueError(
            "Phase 4F structural diagnostics require the "
            "frozen 6,144-row final refit."
        )

    if final_refit.test_targets_accessed:
        raise ValueError(
            "Final-refit metadata unexpectedly records TEST-target access."
        )

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
            "match the Phase 4 probe grid."
        )

    density_prediction = (
        final_refit.density.predict_physical(
            grid.features
        )
    )

    temperature_prediction = (
        final_refit.temperature.predict_physical(
            grid.features
        )
    )

    expected_shape = (
        grid.total_points,
    )

    if density_prediction.shape != expected_shape:
        raise RuntimeError(
            "Final density surrogate returned an unexpected "
            "probe-grid prediction shape."
        )

    if temperature_prediction.shape != expected_shape:
        raise RuntimeError(
            "Final temperature surrogate returned an unexpected "
            "probe-grid prediction shape."
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

    return {
        "probe_grid": {
            "total_points": int(
                grid.total_points
            ),
            "absorbed_power_points": int(
                grid.absorbed_power_W.size
            ),
            "pressure_points": int(
                grid.target_pressure_mTorr.size
            ),
            "source_reference_path": str(
                source_reference_path
            ),
            "source_reference_array_sha256": (
                EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256
            ),
        },
        "density": {
            "target": DENSITY_TARGET,
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
        "overall_passed": bool(
            density_pass
            and temperature_pass
        ),
        "interpretation": {
            "selection_gate": False,
            "post_refit_reporting_diagnostic": True,
            "test_targets_used": False,
            "retuning_permitted_from_result": False,
        },
    }
