"""Canonical Phase 4E source-simulator reference evaluation.

Phase 4E reuses the exact Phase 3 base steady-state simulation path via
``simulate_base_design_point``. No second plasma simulator is implemented.

Probe points are labelled separately from TRAIN, VALIDATION and TEST and are
never appended to those modelling splits.

This module does not fit surrogate models and does not access TEST targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from plasma_ai.experiments.base_dataset import (
    BaseDesignPoint,
    BaseSimulationRow,
    simulate_base_design_point,
)
from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
    load_base_dataset_config,
)
from plasma_ai.surrogate.probe_grid import (
    Phase4EProbeGrid,
)


DEFAULT_PHASE3_CONFIG_PATH = Path(
    "configs/phase3/base_dataset.json"
)

PHASE4E_PROBE_SPLIT = "phase4e_probe"


@dataclass(frozen=True)
class Phase4ESourceReference:
    """Ordered source-simulator results on the frozen Phase 4E grid."""

    rows: tuple[BaseSimulationRow, ...]
    electron_density_m3: NDArray[np.float64]
    electron_temperature_eV: NDArray[np.float64]

    @property
    def total_points(self) -> int:
        """Return the number of evaluated source-reference points."""

        return len(self.rows)

    @property
    def all_qualification_valid(self) -> bool:
        """Return whether every source row passes Phase 3 qualification."""

        return all(
            row.qualification_valid
            for row in self.rows
        )

    @property
    def all_targets_finite(self) -> bool:
        """Return whether both source target arrays are fully finite."""

        return bool(
            np.all(
                np.isfinite(
                    self.electron_density_m3
                )
            )
            and np.all(
                np.isfinite(
                    self.electron_temperature_eV
                )
            )
        )


def _probe_design_point(
    *,
    power_W: float,
    pressure_mTorr: float,
    index: int,
    config: BaseDatasetConfig,
) -> BaseDesignPoint:
    """Create one canonical Phase 3 design point for Phase 4E probing."""

    return BaseDesignPoint(
        simulation_id=(
            f"phase4e_probe_{index:04d}"
        ),
        split=PHASE4E_PROBE_SPLIT,
        split_index=index,
        design_seed=0,
        nominal_absorbed_power_W=float(
            power_W
        ),
        target_pressure_mTorr=float(
            pressure_mTorr
        ),
        nominal_flow_sccm=float(
            config.fixed_flow_sccm
        ),
        gas_temperature_K=float(
            config.gas_temperature_K
        ),
        ion_neutral_cross_section_m2=float(
            config.ion_neutral_cross_section_m2
        ),
        radius_m=float(
            config.radius_m
        ),
        length_m=float(
            config.length_m
        ),
    )


def evaluate_phase4e_source_reference(
    grid: Phase4EProbeGrid,
    *,
    phase3_config_path: str | Path = (
        DEFAULT_PHASE3_CONFIG_PATH
    ),
) -> Phase4ESourceReference:
    """Evaluate the canonical Phase 3 simulator on the ordered probe grid.

    The function intentionally performs source evaluation only. Surrogate
    prediction and acceptance logic are separate Phase 4E stages.
    """

    config = load_base_dataset_config(
        phase3_config_path
    )

    rows: list[BaseSimulationRow] = []

    for index, feature_row in enumerate(
        grid.features
    ):
        power_W = float(
            feature_row[0]
        )
        pressure_mTorr = float(
            feature_row[1]
        )

        point = _probe_design_point(
            power_W=power_W,
            pressure_mTorr=pressure_mTorr,
            index=index,
            config=config,
        )

        row = simulate_base_design_point(
            point,
            config,
        )

        rows.append(row)

    ordered_rows = tuple(rows)

    density = np.asarray(
        [
            row.true_electron_density_m3
            for row in ordered_rows
        ],
        dtype=np.float64,
    )

    temperature = np.asarray(
        [
            row.true_electron_temperature_eV
            for row in ordered_rows
        ],
        dtype=np.float64,
    )

    return Phase4ESourceReference(
        rows=ordered_rows,
        electron_density_m3=density,
        electron_temperature_eV=temperature,
    )


def validate_phase4e_source_reference(
    reference: Phase4ESourceReference,
    *,
    expected_points: int = 1681,
) -> None:
    """Enforce the approved Phase 4E source-reference gate."""

    if reference.total_points != expected_points:
        raise ValueError(
            "Phase 4E source-reference point count "
            "does not match the frozen probe design."
        )

    if not reference.all_targets_finite:
        raise ValueError(
            "Phase 4E source-reference targets "
            "contain non-finite values."
        )

    if not reference.all_qualification_valid:
        invalid_ids = [
            row.simulation_id
            for row in reference.rows
            if not row.qualification_valid
        ]

        raise ValueError(
            "Phase 4E source-reference grid contains "
            "qualification-invalid source rows: "
            + ", ".join(invalid_ids)
        )
