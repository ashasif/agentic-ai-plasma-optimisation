"""Frozen Phase 4E physics-validation probe-grid construction.

The Phase 4A surrogate protocol is authoritative for the physics-validation
probe design and qualified numerical envelope.

The Phase 3 base-dataset configuration is cross-checked so that Phase 4E
cannot silently evaluate a different source-model envelope.

This module constructs coordinates only. It does not execute the source
simulator, fit surrogate models, or access TEST targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


DEFAULT_PHASE4_PROTOCOL_PATH = Path(
    "configs/phase4/surrogate_protocol.json"
)

DEFAULT_PHASE3_CONFIG_PATH = Path(
    "configs/phase3/base_dataset.json"
)


@dataclass(frozen=True)
class Phase4EProbeGrid:
    """Deterministic Cartesian physics-validation probe grid."""

    absorbed_power_W: NDArray[np.float64]
    target_pressure_mTorr: NDArray[np.float64]
    features: NDArray[np.float64]

    @property
    def total_points(self) -> int:
        """Return the number of Cartesian probe points."""

        return int(self.features.shape[0])


def _load_json(path: str | Path) -> dict:
    """Load one JSON configuration file."""

    source = Path(path)

    return json.loads(
        source.read_text(
            encoding="utf-8",
        )
    )


def build_phase4e_probe_grid(
    phase4_protocol_path: str | Path = (
        DEFAULT_PHASE4_PROTOCOL_PATH
    ),
    phase3_config_path: str | Path = (
        DEFAULT_PHASE3_CONFIG_PATH
    ),
) -> Phase4EProbeGrid:
    """Build the exact frozen Phase 4E 41 x 41 Cartesian grid."""

    phase4 = _load_json(
        phase4_protocol_path
    )
    phase3 = _load_json(
        phase3_config_path
    )

    probe_spec = phase4[
        "physics_validation"
    ]["probe_grid"]

    qualified_domain = phase4[
        "qualified_domain"
    ]

    if probe_spec["type"] != "cartesian_regular":
        raise ValueError(
            "Phase 4E requires the frozen "
            "cartesian_regular probe-grid type."
        )

    if not probe_spec[
        "include_domain_endpoints"
    ]:
        raise ValueError(
            "Phase 4E requires inclusion of "
            "qualified-domain endpoints."
        )

    power_min = float(
        qualified_domain[
            "nominal_absorbed_power_W"
        ]["min"]
    )
    power_max = float(
        qualified_domain[
            "nominal_absorbed_power_W"
        ]["max"]
    )

    pressure_min = float(
        qualified_domain[
            "target_pressure_mTorr"
        ]["min"]
    )
    pressure_max = float(
        qualified_domain[
            "target_pressure_mTorr"
        ]["max"]
    )

    phase3_envelope = {
        "power_min": float(
            phase3["absorbed_power_min_W"]
        ),
        "power_max": float(
            phase3["absorbed_power_max_W"]
        ),
        "pressure_min": float(
            phase3["target_pressure_min_mTorr"]
        ),
        "pressure_max": float(
            phase3["target_pressure_max_mTorr"]
        ),
    }

    phase4_envelope = {
        "power_min": power_min,
        "power_max": power_max,
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
    }

    if phase3_envelope != phase4_envelope:
        raise ValueError(
            "Phase 4A qualified domain does not "
            "match the Phase 3 base-model envelope."
        )

    power_points = int(
        probe_spec[
            "absorbed_power_points"
        ]
    )
    pressure_points = int(
        probe_spec[
            "pressure_points"
        ]
    )
    expected_total = int(
        probe_spec[
            "total_points"
        ]
    )

    if (
        power_points
        * pressure_points
        != expected_total
    ):
        raise ValueError(
            "Frozen Phase 4E probe-grid dimensions "
            "do not match total_points."
        )

    power_axis = np.linspace(
        power_min,
        power_max,
        power_points,
        endpoint=True,
        dtype=float,
    )

    pressure_axis = np.linspace(
        pressure_min,
        pressure_max,
        pressure_points,
        endpoint=True,
        dtype=float,
    )

    power_mesh, pressure_mesh = np.meshgrid(
        power_axis,
        pressure_axis,
        indexing="ij",
    )

    features = np.column_stack(
        (
            power_mesh.ravel(),
            pressure_mesh.ravel(),
        )
    ).astype(
        np.float64,
        copy=False,
    )

    if features.shape != (
        expected_total,
        2,
    ):
        raise RuntimeError(
            "Constructed Phase 4E probe grid has "
            "an unexpected shape."
        )

    unique_points = np.unique(
        features,
        axis=0,
    )

    if unique_points.shape[0] != expected_total:
        raise RuntimeError(
            "Constructed Phase 4E probe grid "
            "contains duplicate points."
        )

    return Phase4EProbeGrid(
        absorbed_power_W=power_axis,
        target_pressure_mTorr=pressure_axis,
        features=features,
    )
