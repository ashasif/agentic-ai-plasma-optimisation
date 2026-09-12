"""Phase 4E-R reconstruction and TRAIN-only fitting for acceptance.

The density configuration is reconstructed from the frozen Phase 4E-R
validation-selection artifact.

The temperature configuration remains the exact Phase 4D-selected
candidate.

No model selection, retuning, VALIDATION fitting, TRAIN+VALIDATION refit,
or TEST-target access occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plasma_ai.surrogate.classical import (
    ClassicalCandidateSpec,
)
from plasma_ai.surrogate.data import (
    SurrogateDataset,
    file_sha256,
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4e_candidates import (
    FittedSelectedCandidate,
    SelectedCandidateConfiguration,
    fit_selected_candidate_on_train,
    load_phase4d_selected_configurations,
)
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    build_phase4er_density_candidate,
    enumerate_phase4er_density_specs,
)
from plasma_ai.surrogate.transforms import (
    TargetTransform,
)


DEFAULT_PHASE4ER_SELECTION_PATH = Path(
    "results/phase4/phase4er_validation_selection.json"
)

DEFAULT_PHASE4D_SELECTION_PATH = Path(
    "results/phase4/validation_selection.json"
)

EXPECTED_PHASE4ER_SELECTION_SHA256 = (
    "453a30cfc1dcd9aa8a9ec561f6de42ba07da4278d242d145863e16489002f388"
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


def load_phase4er_density_configuration(
    selection_path: str | Path = DEFAULT_PHASE4ER_SELECTION_PATH,
) -> SelectedCandidateConfiguration:
    """Reconstruct the exact frozen Phase 4E-R density winner."""

    path = Path(
        selection_path
    )

    observed_hash = file_sha256(
        path
    )

    if (
        observed_hash
        != EXPECTED_PHASE4ER_SELECTION_SHA256
    ):
        raise ValueError(
            "Phase 4E-R validation-selection artifact SHA-256 "
            "does not match the frozen result."
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if payload.get(
        "phase"
    ) != "4E-R":
        raise ValueError(
            "Density configuration must come from "
            "the Phase 4E-R artifact."
        )

    if payload.get(
        "phase4f_status"
    ) != "locked":
        raise ValueError(
            "Phase 4F must remain locked during Phase 4E-R."
        )

    data_usage = payload[
        "data_usage"
    ]

    if data_usage.get(
        "test_targets_accessed"
    ) is not False:
        raise ValueError(
            "Phase 4E-R selection artifact must record "
            "TEST targets as unaccessed."
        )

    if data_usage.get(
        "train_validation_refit_performed"
    ) is not False:
        raise ValueError(
            "Phase 4E-R selection artifact must precede "
            "TRAIN+VALIDATION refit."
        )

    selected = payload[
        "density_redevelopment"
    ][
        "selection"
    ]

    if selected.get(
        "target"
    ) != DENSITY_TARGET:
        raise ValueError(
            "Unexpected Phase 4E-R selected target."
        )

    if selected.get(
        "selected_transform"
    ) != PHASE4ER_DENSITY_TRANSFORM:
        raise ValueError(
            "Unexpected Phase 4E-R density transform."
        )

    if selected.get(
        "validation_selected_only"
    ) is not True:
        raise ValueError(
            "Phase 4E-R acceptance requires a "
            "VALIDATION-selected density candidate."
        )

    selected_id = selected[
        "selected_candidate_id"
    ]

    selected_model = selected[
        "selected_model"
    ]

    selected_parameters = dict(
        selected[
            "selected_parameters"
        ]
    )

    matching_specs = [
        spec
        for spec in enumerate_phase4er_density_specs()
        if spec.candidate_id == selected_id
    ]

    if len(
        matching_specs
    ) != 1:
        raise ValueError(
            "Frozen Phase 4E-R selected candidate ID "
            "is not in the predeclared candidate grid."
        )

    frozen_spec = matching_specs[
        0
    ]

    if (
        frozen_spec.model_name
        != selected_model
    ):
        raise ValueError(
            "Phase 4E-R selected model does not match "
            "the frozen candidate specification."
        )

    if (
        dict(
            frozen_spec.parameters
        )
        != selected_parameters
    ):
        raise ValueError(
            "Phase 4E-R selected parameters do not match "
            "the frozen candidate specification."
        )

    return SelectedCandidateConfiguration(
        target_name=DENSITY_TARGET,
        transform_name=PHASE4ER_DENSITY_TRANSFORM,
        candidate_spec=ClassicalCandidateSpec(
            candidate_id=selected_id,
            model_name=selected_model,
            parameters=selected_parameters,
        ),
    )


def fit_phase4er_density_on_train(
    dataset: SurrogateDataset,
    configuration: SelectedCandidateConfiguration,
) -> FittedSelectedCandidate:
    """Fit the fixed Phase 4E-R density winner on TRAIN only."""

    train = dataset.split(
        "train"
    )

    test = dataset.split(
        "test"
    )

    if train.y is None:
        raise RuntimeError(
            "TRAIN targets are unavailable."
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

    if int(
        train.X.shape[0]
    ) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Phase 4E-R acceptance density candidate must "
            f"fit exactly {EXPECTED_TRAIN_ROWS} TRAIN rows."
        )

    if (
        configuration.target_name
        != DENSITY_TARGET
    ):
        raise ValueError(
            "Phase 4E-R constrained fit is density-only."
        )

    if (
        configuration.transform_name
        != PHASE4ER_DENSITY_TRANSFORM
    ):
        raise ValueError(
            "Unexpected Phase 4E-R density transform."
        )

    physical_y = np.asarray(
        train.y[
            :,
            0,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        physical_y
    ).all():
        raise ValueError(
            "TRAIN density target contains non-finite values."
        )

    if np.any(
        physical_y <= 0.0
    ):
        raise ValueError(
            "TRAIN density target must be strictly positive."
        )

    transform = TargetTransform(
        configuration.transform_name
    )

    transformed_y = transform.forward(
        physical_y
    )

    model = build_phase4er_density_candidate(
        configuration.candidate_spec
    )

    model.fit(
        train.X,
        transformed_y,
    )

    return FittedSelectedCandidate(
        configuration=configuration,
        model=model,
        transform=transform,
        fit_rows=int(
            train.X.shape[0]
        ),
    )


def fit_phase4er_acceptance_candidates(
    *,
    phase4er_selection_path: str | Path = DEFAULT_PHASE4ER_SELECTION_PATH,
    phase4d_selection_path: str | Path = DEFAULT_PHASE4D_SELECTION_PATH,
) -> tuple[
    FittedSelectedCandidate,
    FittedSelectedCandidate,
]:
    """Fit fixed density and temperature candidates using TRAIN only."""

    dataset = load_phase4_dataset()

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
        temperature_configuration.target_name
        != TEMPERATURE_TARGET
    ):
        raise RuntimeError(
            "Unexpected Phase 4D temperature configuration."
        )

    density_model = (
        fit_phase4er_density_on_train(
            dataset,
            density_configuration,
        )
    )

    temperature_model = (
        fit_selected_candidate_on_train(
            dataset,
            temperature_configuration,
        )
    )

    if (
        density_model.fit_rows
        != EXPECTED_TRAIN_ROWS
        or temperature_model.fit_rows
        != EXPECTED_TRAIN_ROWS
    ):
        raise RuntimeError(
            "Both Phase 4E-R acceptance candidates must "
            "fit exactly the frozen TRAIN split."
        )

    return (
        density_model,
        temperature_model,
    )
