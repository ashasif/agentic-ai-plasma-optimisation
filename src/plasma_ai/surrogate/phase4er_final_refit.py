"""Phase 4E-R final TRAIN+VALIDATION refit.

This module performs the pre-test final refit only after the frozen
Phase 4E-R physics-acceptance artifact authorizes it.

Both frozen estimators are rebuilt from their already-selected
configurations and fitted on exactly TRAIN + VALIDATION = 6,144 rows.

No model selection, retuning, physics re-evaluation, or TEST-target
access occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.classical import (
    build_classical_candidate,
)
from plasma_ai.surrogate.data import (
    file_sha256,
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4e_candidates import (
    FittedSelectedCandidate,
    load_phase4d_selected_configurations,
)
from plasma_ai.surrogate.phase4er_candidates import (
    DEFAULT_PHASE4D_SELECTION_PATH,
    DEFAULT_PHASE4ER_SELECTION_PATH,
    load_phase4er_density_configuration,
)
from plasma_ai.surrogate.phase4er_density import (
    build_phase4er_density_candidate,
)
from plasma_ai.surrogate.transforms import (
    TargetTransform,
)


DEFAULT_ACCEPTANCE_PATH = Path(
    "results/phase4/phase4er_physics_acceptance.json"
)

EXPECTED_ACCEPTANCE_SHA256 = (
    "618cabc5fc28be08d24b73c747262b40ed2a6134820098ae0eb33b2d1fc71875"
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
EXPECTED_FINAL_REFIT_ROWS = 6144


@dataclass
class Phase4ERFinalRefit:
    """Locked pre-test final surrogate candidates."""

    density: FittedSelectedCandidate
    temperature: FittedSelectedCandidate
    fit_rows: int
    test_targets_accessed: bool = False


def validate_phase4er_refit_gate(
    acceptance_path: str | Path = DEFAULT_ACCEPTANCE_PATH,
) -> dict:
    """Verify that frozen physics acceptance permits the final refit."""

    path = Path(
        acceptance_path
    )

    observed_hash = file_sha256(
        path
    )

    if observed_hash != EXPECTED_ACCEPTANCE_SHA256:
        raise ValueError(
            "Phase 4E-R physics-acceptance artifact SHA-256 "
            "does not match the frozen accepted result."
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("phase") != "4E-R":
        raise ValueError(
            "Final refit requires the Phase 4E-R "
            "physics-acceptance artifact."
        )

    if payload.get("phase4f_status") != "locked":
        raise ValueError(
            "Phase 4F must remain locked before final refit."
        )

    data_usage = payload[
        "data_usage"
    ]

    if data_usage.get(
        "test_targets_accessed"
    ) is not False:
        raise ValueError(
            "TEST targets must remain unaccessed before final refit."
        )

    if data_usage.get(
        "train_validation_refit_performed"
    ) is not False:
        raise ValueError(
            "Frozen acceptance artifact must precede "
            "the final TRAIN+VALIDATION refit."
        )

    acceptance = payload[
        "acceptance"
    ]

    if acceptance.get(
        "density_passed"
    ) is not True:
        raise ValueError(
            "Density physics acceptance did not pass."
        )

    if acceptance.get(
        "temperature_passed"
    ) is not True:
        raise ValueError(
            "Temperature physics acceptance did not pass."
        )

    if acceptance.get(
        "overall_passed"
    ) is not True:
        raise ValueError(
            "Overall Phase 4E-R physics acceptance did not pass."
        )

    if acceptance.get(
        "final_train_validation_refit_allowed"
    ) is not True:
        raise ValueError(
            "Frozen Phase 4E-R acceptance does not authorize "
            "the final TRAIN+VALIDATION refit."
        )

    gates = payload[
        "gates"
    ]

    if gates.get(
        "locked_test_evaluation_performed"
    ) is not False:
        raise ValueError(
            "Locked TEST evaluation must not precede final refit."
        )

    return payload


def run_phase4er_final_refit(
    *,
    acceptance_path: str | Path = DEFAULT_ACCEPTANCE_PATH,
    phase4er_selection_path: str | Path = DEFAULT_PHASE4ER_SELECTION_PATH,
    phase4d_selection_path: str | Path = DEFAULT_PHASE4D_SELECTION_PATH,
) -> Phase4ERFinalRefit:
    """Refit both frozen candidates on exactly TRAIN + VALIDATION."""

    validate_phase4er_refit_gate(
        acceptance_path
    )

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
            "TEST targets must remain locked during "
            "the Phase 4E-R final refit."
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

    if int(
        train.X.shape[0]
    ) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Unexpected TRAIN row count."
        )

    if int(
        validation.X.shape[0]
    ) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            "Unexpected VALIDATION row count."
        )

    combined_X = np.concatenate(
        (
            train.X,
            validation.X,
        ),
        axis=0,
    )

    combined_y = np.concatenate(
        (
            train.y,
            validation.y,
        ),
        axis=0,
    )

    if combined_X.shape != (
        EXPECTED_FINAL_REFIT_ROWS,
        2,
    ):
        raise RuntimeError(
            "Final refit feature matrix must contain "
            f"exactly {EXPECTED_FINAL_REFIT_ROWS} rows "
            "and two frozen features."
        )

    if combined_y.shape != (
        EXPECTED_FINAL_REFIT_ROWS,
        2,
    ):
        raise RuntimeError(
            "Final refit target matrix must contain "
            f"exactly {EXPECTED_FINAL_REFIT_ROWS} rows "
            "and two frozen targets."
        )

    density_configuration = (
        load_phase4er_density_configuration(
            phase4er_selection_path
        )
    )

    (
        _,
        temperature_configuration,
    ) = load_phase4d_selected_configurations(
        phase4d_selection_path
    )

    if (
        density_configuration.target_name
        != DENSITY_TARGET
    ):
        raise RuntimeError(
            "Unexpected frozen density configuration."
        )

    if (
        temperature_configuration.target_name
        != TEMPERATURE_TARGET
    ):
        raise RuntimeError(
            "Unexpected frozen temperature configuration."
        )

    density_transform = TargetTransform(
        density_configuration.transform_name
    )

    temperature_transform = TargetTransform(
        temperature_configuration.transform_name
    )

    density_y = np.asarray(
        combined_y[
            :,
            0,
        ],
        dtype=np.float64,
    )

    temperature_y = np.asarray(
        combined_y[
            :,
            1,
        ],
        dtype=np.float64,
    )

    if not (
        np.isfinite(density_y).all()
        and np.isfinite(temperature_y).all()
    ):
        raise ValueError(
            "Final refit targets must contain only finite values."
        )

    if np.any(
        density_y <= 0.0
    ):
        raise ValueError(
            "Final refit density values must be strictly positive."
        )

    if np.any(
        temperature_y <= 0.0
    ):
        raise ValueError(
            "Final refit temperature values must be strictly positive."
        )

    density_model = (
        build_phase4er_density_candidate(
            density_configuration.candidate_spec
        )
    )

    temperature_model = (
        build_classical_candidate(
            temperature_configuration.candidate_spec
        )
    )

    density_model.fit(
        combined_X,
        density_transform.forward(
            density_y
        ),
    )

    temperature_model.fit(
        combined_X,
        temperature_transform.forward(
            temperature_y
        ),
    )

    density_fitted = FittedSelectedCandidate(
        configuration=density_configuration,
        model=density_model,
        transform=density_transform,
        fit_rows=EXPECTED_FINAL_REFIT_ROWS,
    )

    temperature_fitted = FittedSelectedCandidate(
        configuration=temperature_configuration,
        model=temperature_model,
        transform=temperature_transform,
        fit_rows=EXPECTED_FINAL_REFIT_ROWS,
    )

    return Phase4ERFinalRefit(
        density=density_fitted,
        temperature=temperature_fitted,
        fit_rows=EXPECTED_FINAL_REFIT_ROWS,
        test_targets_accessed=False,
    )
