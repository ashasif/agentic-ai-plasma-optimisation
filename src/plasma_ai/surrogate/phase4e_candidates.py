"""Phase 4E reconstruction and TRAIN-only fitting of Phase 4D winners.

The selected specifications are read from the frozen Phase 4D selection
artifact. No model selection or retuning occurs here.

The Phase 4 classical constructor and target-transform implementation are
reused directly.

TEST targets must remain unavailable throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plasma_ai.surrogate.classical import (
    ClassicalCandidateSpec,
    build_classical_candidate,
)
from plasma_ai.surrogate.data import (
    SurrogateDataset,
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.transforms import (
    TargetTransform,
    allowed_transforms_for_target,
)


DEFAULT_SELECTION_PATH = Path(
    "results/phase4/validation_selection.json"
)

TARGET_INDEX = {
    DENSITY_TARGET: 0,
    TEMPERATURE_TARGET: 1,
}


@dataclass(frozen=True)
class SelectedCandidateConfiguration:
    """Exact Phase 4D-selected configuration for one target."""

    target_name: str
    transform_name: str
    candidate_spec: ClassicalCandidateSpec


@dataclass
class FittedSelectedCandidate:
    """TRAIN-fitted Phase 4E candidate with physical-scale prediction."""

    configuration: SelectedCandidateConfiguration
    model: Any
    transform: TargetTransform
    fit_rows: int

    def predict_physical(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        """Predict and return values on the physical target scale."""

        matrix = np.asarray(
            X,
            dtype=np.float64,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Prediction features must be a two-dimensional matrix."
            )

        if matrix.shape[1] != 2:
            raise ValueError(
                "Phase 4E predictions require exactly two frozen features."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "Prediction features must contain only finite values."
            )

        transformed = np.asarray(
            self.model.predict(
                matrix
            ),
            dtype=np.float64,
        )

        return self.transform.inverse(
            transformed
        )


def load_phase4d_selected_configurations(
    selection_path: str | Path = DEFAULT_SELECTION_PATH,
) -> tuple[
    SelectedCandidateConfiguration,
    SelectedCandidateConfiguration,
]:
    """Reconstruct the exact density and temperature Phase 4D winners."""

    payload = json.loads(
        Path(
            selection_path
        ).read_text(
            encoding="utf-8"
        )
    )

    if payload.get("phase") != "4D":
        raise ValueError(
            "Selected-candidate artifact must be the Phase 4D artifact."
        )

    selection = payload[
        "selection"
    ]

    configurations = []

    for key, expected_target in (
        (
            "density",
            DENSITY_TARGET,
        ),
        (
            "temperature",
            TEMPERATURE_TARGET,
        ),
    ):
        selected = selection[
            key
        ]

        target_name = selected[
            "target"
        ]

        if target_name != expected_target:
            raise ValueError(
                f"Unexpected selected target for {key!r}: "
                f"{target_name!r}."
            )

        transform_name = selected[
            "selected_transform"
        ]

        if transform_name not in (
            allowed_transforms_for_target(
                target_name
            )
        ):
            raise ValueError(
                "Phase 4D artifact contains a transform "
                "outside the frozen target-transform contract."
            )

        if not selected.get(
            "validation_selected_only",
            False,
        ):
            raise ValueError(
                "Phase 4E requires Phase 4D validation-selected "
                "candidate metadata."
            )

        configurations.append(
            SelectedCandidateConfiguration(
                target_name=target_name,
                transform_name=transform_name,
                candidate_spec=ClassicalCandidateSpec(
                    candidate_id=selected[
                        "selected_candidate_id"
                    ],
                    model_name=selected[
                        "selected_model"
                    ],
                    parameters=dict(
                        selected[
                            "selected_parameters"
                        ]
                    ),
                ),
            )
        )

    return (
        configurations[0],
        configurations[1],
    )


def fit_selected_candidate_on_train(
    dataset: SurrogateDataset,
    configuration: SelectedCandidateConfiguration,
) -> FittedSelectedCandidate:
    """Fit one fixed Phase 4D-selected candidate using TRAIN only."""

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
            "TEST targets must remain locked during Phase 4E."
        )

    if tuple(
        dataset.target_names
    ) != (
        DENSITY_TARGET,
        TEMPERATURE_TARGET,
    ):
        raise RuntimeError(
            "Unexpected Phase 4 target ordering."
        )

    target_index = TARGET_INDEX[
        configuration.target_name
    ]

    physical_y = np.asarray(
        train.y[
            :,
            target_index,
        ],
        dtype=np.float64,
    )

    transform = TargetTransform(
        configuration.transform_name
    )

    transformed_y = transform.forward(
        physical_y
    )

    model = build_classical_candidate(
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


def fit_phase4e_selected_candidates(
    *,
    selection_path: str | Path = DEFAULT_SELECTION_PATH,
) -> tuple[
    FittedSelectedCandidate,
    FittedSelectedCandidate,
]:
    """Load TEST-locked data and fit both selected candidates on TRAIN."""

    dataset = load_phase4_dataset()

    configurations = (
        load_phase4d_selected_configurations(
            selection_path
        )
    )

    return tuple(
        fit_selected_candidate_on_train(
            dataset,
            configuration,
        )
        for configuration in configurations
    )
