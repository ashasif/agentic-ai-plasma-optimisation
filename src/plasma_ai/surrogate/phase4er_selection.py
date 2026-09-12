"""Deterministic validation selection for Phase 4E-R density redevelopment.

The Phase 4E-R protocol preserves the frozen density primary and secondary
metric policy while using the separately predeclared Phase 4E-R candidate
grid for deterministic final tie-breaking.

This module performs no dataset loading and cannot access TEST targets.
"""

from __future__ import annotations

from math import isfinite

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
)
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    enumerate_phase4er_density_specs,
)
from plasma_ai.surrogate.selection import (
    ValidationSelectionDecision,
    primary_equivalence_indices,
)


PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION = 0.02

PHASE4ER_PRIMARY_METRIC = (
    "p95_absolute_relative_error"
)

PHASE4ER_SECONDARY_METRICS = (
    "median_absolute_relative_error",
    "rmse",
)


def _validated_metric_value(
    result: ClassicalValidationResult,
    metric_name: str,
) -> float:
    if metric_name not in result.metrics:
        raise ValueError(
            f"Candidate {result.candidate_id!r} is missing "
            f"required metric {metric_name!r}."
        )

    value = float(
        result.metrics[
            metric_name
        ]
    )

    if not isfinite(value):
        raise ValueError(
            f"Candidate {result.candidate_id!r} metric "
            f"{metric_name!r} must be finite."
        )

    if value < 0.0:
        raise ValueError(
            f"Candidate {result.candidate_id!r} metric "
            f"{metric_name!r} must be non-negative."
        )

    return value


def _phase4er_canonical_order(
) -> dict[str, int]:
    return {
        spec.candidate_id: index
        for index, spec in enumerate(
            enumerate_phase4er_density_specs()
        )
    }


def select_phase4er_density_candidate(
    results,
) -> ValidationSelectionDecision:
    """Apply the frozen Phase 4E-R density validation-selection policy."""

    candidates = tuple(
        results
    )

    if not candidates:
        raise ValueError(
            "Phase 4E-R density selection requires "
            "at least one candidate."
        )

    canonical_order = (
        _phase4er_canonical_order()
    )

    expected_ids = set(
        canonical_order
    )

    observed_ids = {
        result.candidate_id
        for result in candidates
    }

    if len(observed_ids) != len(candidates):
        raise ValueError(
            "Phase 4E-R density selection candidate IDs "
            "must be unique."
        )

    if observed_ids != expected_ids:
        raise ValueError(
            "Phase 4E-R density selection requires exactly "
            "the frozen 16-candidate redevelopment grid."
        )

    for result in candidates:
        if result.target_name != DENSITY_TARGET:
            raise ValueError(
                "All Phase 4E-R selection candidates must "
                "target electron density."
            )

        if (
            result.transform_name
            != PHASE4ER_DENSITY_TRANSFORM
        ):
            raise ValueError(
                "All Phase 4E-R density candidates must use "
                "the frozen log10 transform."
            )

    primary_values = tuple(
        _validated_metric_value(
            result,
            PHASE4ER_PRIMARY_METRIC,
        )
        for result in candidates
    )

    best_primary_value = min(
        primary_values
    )

    if best_primary_value == 0.0:
        primary_strict_upper_boundary = 0.0
    else:
        primary_strict_upper_boundary = (
            best_primary_value
            * (
                1.0
                + PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
            )
        )

    equivalent_indices = (
        primary_equivalence_indices(
            primary_values,
            relative_fraction=(
                PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
            ),
        )
    )

    cohort = tuple(
        candidates[index]
        for index in equivalent_indices
    )

    primary_equivalent_candidate_ids = tuple(
        result.candidate_id
        for result in cohort
    )

    primary_equivalent_candidates = tuple(
        {
            "candidate_id": result.candidate_id,
            "transform": result.transform_name,
        }
        for result in cohort
    )

    def decision(
        selected_result,
        selection_stage,
    ):
        return ValidationSelectionDecision(
            selected_result=selected_result,
            selection_stage=selection_stage,
            primary_metric=PHASE4ER_PRIMARY_METRIC,
            secondary_metrics=PHASE4ER_SECONDARY_METRICS,
            primary_equivalent_candidate_ids=(
                primary_equivalent_candidate_ids
            ),
            primary_equivalent_candidates=(
                primary_equivalent_candidates
            ),
            best_primary_value=best_primary_value,
            practical_equivalence_relative_fraction=(
                PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
            ),
            primary_strict_upper_boundary=(
                primary_strict_upper_boundary
            ),
        )

    if len(cohort) == 1:
        return decision(
            cohort[0],
            "primary_metric",
        )

    for metric_name in PHASE4ER_SECONDARY_METRICS:
        metric_values = tuple(
            _validated_metric_value(
                result,
                metric_name,
            )
            for result in cohort
        )

        best_value = min(
            metric_values
        )

        cohort = tuple(
            result
            for result, value in zip(
                cohort,
                metric_values,
                strict=True,
            )
            if value == best_value
        )

        if len(cohort) == 1:
            return decision(
                cohort[0],
                f"secondary_metric:{metric_name}",
            )

    selected = min(
        cohort,
        key=lambda result: canonical_order[
            result.candidate_id
        ],
    )

    return decision(
        selected,
        "canonical_fallback",
    )
