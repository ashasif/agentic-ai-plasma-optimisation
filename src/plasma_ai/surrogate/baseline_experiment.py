"""Reproducible Phase 4B reference-baseline experiment.

This experiment uses only the frozen TRAIN and VALIDATION splits.

TEST targets remain withheld.
"""

from __future__ import annotations

import json
from pathlib import Path

from plasma_ai.surrogate.baselines import (
    fit_reference_baseline,
)
from plasma_ai.surrogate.data import (
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/reference_baselines.json"
)


def run_reference_baselines() -> dict:
    """Run the frozen Phase 4B reference baseline experiment."""
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
            "TEST targets must remain locked in Phase 4B."
        )

    target_index = {
        DENSITY_TARGET: 0,
        TEMPERATURE_TARGET: 1,
    }

    experiment_plan = [
        (
            DENSITY_TARGET,
            "identity",
            "dummy_mean",
        ),
        (
            DENSITY_TARGET,
            "log10",
            "dummy_mean",
        ),
        (
            DENSITY_TARGET,
            "identity",
            "linear_regression",
        ),
        (
            DENSITY_TARGET,
            "log10",
            "linear_regression",
        ),
        (
            TEMPERATURE_TARGET,
            "identity",
            "dummy_mean",
        ),
        (
            TEMPERATURE_TARGET,
            "identity",
            "linear_regression",
        ),
    ]

    results = []

    for (
        target_name,
        transform_name,
        model_name,
    ) in experiment_plan:
        index = target_index[
            target_name
        ]

        result = fit_reference_baseline(
            train_X=train.X,
            train_y=train.y[:, index],
            validation_X=validation.X,
            validation_y=validation.y[:, index],
            target_name=target_name,
            transform_name=transform_name,
            model_name=model_name,
        )

        results.append({
            "target": result.target_name,
            "transform": result.transform_name,
            "model": result.model_name,
            "metrics": {
                name: float(value)
                for name, value
                in result.metrics.items()
            },
        })

    return {
        "phase": "4B",
        "experiment": "reference_baselines",
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
        "results": results,
    }


def write_reference_baselines(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Run and write the Phase 4B reference results."""
    destination = Path(
        output_path
    )

    payload = run_reference_baselines()

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
