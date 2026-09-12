"""Auditable Phase 4E source-reference artifact construction.

The artifact records the canonical reduced-order source-simulator response on
the frozen Phase 4E physics-validation probe grid.

It contains source results only. It does not contain surrogate predictions,
TEST targets, TEST metrics, or production models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.probe_grid import (
    Phase4EProbeGrid,
)
from plasma_ai.surrogate.source_reference import (
    Phase4ESourceReference,
    validate_phase4e_source_reference,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/source_reference_grid.json"
)


def build_source_reference_payload(
    grid: Phase4EProbeGrid,
    reference: Phase4ESourceReference,
) -> dict:
    """Build the deterministic Phase 4E source-reference payload."""

    if grid.total_points != reference.total_points:
        raise ValueError(
            "Probe-grid and source-reference point counts differ."
        )

    validate_phase4e_source_reference(
        reference,
        expected_points=grid.total_points,
    )

    source_features = np.asarray(
        [
            [
                row.nominal_absorbed_power_W,
                row.target_pressure_mTorr,
            ]
            for row in reference.rows
        ],
        dtype=np.float64,
    )

    if not np.array_equal(
        source_features,
        grid.features,
    ):
        raise ValueError(
            "Source-reference row ordering does not match "
            "the frozen probe grid."
        )

    reference_matrix = np.column_stack(
        (
            grid.features,
            reference.electron_density_m3,
            reference.electron_temperature_eV,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    array_sha256 = hashlib.sha256(
        reference_matrix.tobytes(
            order="C",
        )
    ).hexdigest()

    rows = []

    for row in reference.rows:
        rows.append(
            {
                "simulation_id": row.simulation_id,
                "nominal_absorbed_power_W": float(
                    row.nominal_absorbed_power_W
                ),
                "target_pressure_mTorr": float(
                    row.target_pressure_mTorr
                ),
                "true_electron_density_m3": float(
                    row.true_electron_density_m3
                ),
                "true_electron_temperature_eV": float(
                    row.true_electron_temperature_eV
                ),
                "integration_success": bool(
                    row.integration_success
                ),
                "converged": bool(
                    row.converged
                ),
                "qualification_valid": bool(
                    row.qualification_valid
                ),
                "domain_status": row.domain_status,
                "model_validity_status": (
                    row.model_validity_status
                ),
            }
        )

    return {
        "phase": "4E",
        "artifact": "source_reference_grid",
        "reference_role": (
            "source_reduced_order_simulator"
        ),
        "source_evaluated_before_surrogate": True,
        "probe_design": {
            "type": "cartesian_regular",
            "absorbed_power_points": int(
                grid.absorbed_power_W.size
            ),
            "pressure_points": int(
                grid.target_pressure_mTorr.size
            ),
            "total_points": int(
                grid.total_points
            ),
            "feature_order": [
                "nominal_absorbed_power_W",
                "target_pressure_mTorr",
            ],
        },
        "source_reference_gate": {
            "all_targets_finite": bool(
                reference.all_targets_finite
            ),
            "all_qualification_valid": bool(
                reference.all_qualification_valid
            ),
            "qualification_valid_count": int(
                sum(
                    row.qualification_valid
                    for row in reference.rows
                )
            ),
            "integration_success_count": int(
                sum(
                    row.integration_success
                    for row in reference.rows
                )
            ),
            "converged_count": int(
                sum(
                    row.converged
                    for row in reference.rows
                )
            ),
        },
        "reference_array_contract": {
            "column_order": [
                "nominal_absorbed_power_W",
                "target_pressure_mTorr",
                "true_electron_density_m3",
                "true_electron_temperature_eV",
            ],
            "dtype": "float64",
            "byte_order": "C",
            "sha256": array_sha256,
        },
        "scientific_scope": {
            "synthetic_data": True,
            "reduced_order_argon_plasma_model": True,
            "numerically_qualified_model_envelope": True,
            "experimental_validation": False,
            "industrial_validation": False,
            "oipt_operating_range_claim": False,
            "absorbed_power_is_generator_rf_power": False,
            "reactive_etch_or_deposition_prediction": False,
            "wafer_scale_spatial_modelling": False,
        },
        "rows": rows,
    }


def write_source_reference_artifact(
    grid: Phase4EProbeGrid,
    reference: Phase4ESourceReference,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write the deterministic Phase 4E source-reference JSON artifact."""

    destination = Path(
        output_path
    )

    payload = build_source_reference_payload(
        grid,
        reference,
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
