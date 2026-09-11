"""Reproducible Phase 4C classical-surrogate benchmark.

Phase 4C benchmarks predeclared classical candidate configurations
using the frozen TRAIN and VALIDATION splits only.

TEST targets remain withheld.

This phase reports candidate-family behaviour but does not perform the
final validation-based model selection reserved for Phase 4D.
"""

from __future__ import annotations

import json
from pathlib import Path

from plasma_ai.surrogate.classical import (
    PHASE4C_BENCHMARK_SPECS,
    fit_classical_candidate,
)
from plasma_ai.surrogate.data import (
    load_phase4_dataset,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/classical_benchmark.json"
)

REFERENCE_BASELINE_PATH = Path(
    "results/phase4/reference_baselines.json"
)


PHASE4C_TARGET_TRANSFORMS = (
    (
        DENSITY_TARGET,
        "identity",
    ),
    (
        DENSITY_TARGET,
        "log10",
    ),
    (
        TEMPERATURE_TARGET,
        "identity",
    ),
)


def _load_reference_baselines() -> dict:
    if not REFERENCE_BASELINE_PATH.exists():
        raise FileNotFoundError(
            "Phase 4B reference-baseline artifact "
            f"not found: {REFERENCE_BASELINE_PATH}"
        )

    payload = json.loads(
        REFERENCE_BASELINE_PATH.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("phase") != "4B":
        raise ValueError(
            "Unexpected reference-baseline phase marker."
        )

    if (
        payload.get("experiment")
        != "reference_baselines"
    ):
        raise ValueError(
            "Unexpected reference-baseline "
            "experiment marker."
        )

    data_usage = payload.get(
        "data_usage",
        {},
    )

    if data_usage.get(
        "test_targets_accessed"
    ) is not False:
        raise ValueError(
            "Reference baselines do not confirm "
            "locked TEST-target handling."
        )

    return payload


def _build_linear_baseline_comparisons(
    *,
    results: list[dict],
    reference_baselines: dict,
) -> list[dict]:
    """Compare Phase 4C primary metrics with Phase 4B linear baselines.

    These comparisons are descriptive only. They do not perform
    Phase 4D model or target-transform selection.
    """
    linear_baselines = {}

    for baseline in reference_baselines["results"]:
        if baseline["model"] != "linear_regression":
            continue

        key = (
            baseline["target"],
            baseline["transform"],
        )

        linear_baselines[key] = baseline

    comparisons = []

    for result in results:
        key = (
            result["target"],
            result["transform"],
        )

        if key not in linear_baselines:
            raise ValueError(
                "Missing Phase 4B linear baseline for "
                f"target/transform combination {key!r}."
            )

        if result["target"] == DENSITY_TARGET:
            primary_metric = (
                "p95_absolute_relative_error"
            )
        elif result["target"] == TEMPERATURE_TARGET:
            primary_metric = "rmse_eV"
        else:
            raise KeyError(
                "Unknown Phase 4 target in benchmark "
                f"comparison: {result['target']!r}."
            )

        baseline_value = float(
            linear_baselines[key][
                "metrics"
            ][primary_metric]
        )

        candidate_value = float(
            result["metrics"][primary_metric]
        )

        if baseline_value <= 0.0:
            raise ValueError(
                "Phase 4B primary baseline metric "
                "must be strictly positive."
            )

        relative_improvement = (
            baseline_value - candidate_value
        ) / baseline_value

        comparisons.append({
            "target": result["target"],
            "transform": result["transform"],
            "candidate_id": result["candidate_id"],
            "primary_metric": primary_metric,
            "linear_baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "relative_improvement_fraction": float(
                relative_improvement
            ),
            "better_than_linear": bool(
                candidate_value < baseline_value
            ),
            "selection_claim": False,
        })

    return comparisons


def run_classical_benchmark() -> dict:
    """Run the controlled Phase 4C benchmark."""
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
            "TEST targets must remain locked in Phase 4C."
        )

    target_index = {
        DENSITY_TARGET: 0,
        TEMPERATURE_TARGET: 1,
    }

    results = []

    for (
        target_name,
        transform_name,
    ) in PHASE4C_TARGET_TRANSFORMS:
        index = target_index[
            target_name
        ]

        for spec in PHASE4C_BENCHMARK_SPECS:
            result = fit_classical_candidate(
                train_X=train.X,
                train_y=train.y[:, index],
                validation_X=validation.X,
                validation_y=validation.y[:, index],
                target_name=target_name,
                transform_name=transform_name,
                spec=spec,
            )

            results.append({
                "target": result.target_name,
                "transform": result.transform_name,
                "candidate_id": result.candidate_id,
                "model": result.model_name,
                "parameters": dict(
                    result.parameters
                ),
                "metrics": {
                    name: float(value)
                    for name, value
                    in result.metrics.items()
                },
                "prediction_sanity": dict(
                    result.prediction_sanity
                ),
            })

    if len(results) != 12:
        raise RuntimeError(
            "Phase 4C benchmark must produce "
            f"exactly 12 results; observed {len(results)}."
        )

    reference_baselines = (
        _load_reference_baselines()
    )

    baseline_comparison = (
        _build_linear_baseline_comparisons(
            results=results,
            reference_baselines=reference_baselines,
        )
    )

    return {
        "phase": "4C",
        "experiment": "classical_benchmark",
        "scientific_scope": {
            "data_type": "synthetic data",
            "source_model": (
                "reduced-order argon plasma model"
            ),
            "domain": (
                "numerically qualified model envelope"
            ),
            "surrogate_role": (
                "surrogate of the reduced-order simulator"
            ),
            "experimental_validation": False,
            "industrial_validation": False,
            "oipt_operating_range_claim": False,
            "reactive_etch_or_deposition_prediction": False,
            "wafer_scale_spatial_modelling": False,
            "absorbed_power_is_generator_rf_power": False,
        },
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
        "benchmark_design": {
            "candidate_configurations": int(
                len(PHASE4C_BENCHMARK_SPECS)
            ),
            "target_transform_combinations": int(
                len(PHASE4C_TARGET_TRANSFORMS)
            ),
            "total_fits": int(
                len(results)
            ),
            "full_hyperparameter_search": False,
            "final_model_selection_performed": False,
            "final_density_transform_selected": False,
            "production_model_persisted": False,
            "phase4d_reserved_for_selection": True,
        },
        "reference_baselines": reference_baselines,
        "baseline_comparison": baseline_comparison,
        "results": results,
    }


def write_classical_benchmark(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Run and write the Phase 4C benchmark artifact."""
    destination = Path(
        output_path
    )

    payload = run_classical_benchmark()

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
