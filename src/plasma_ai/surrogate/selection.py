"""Controlled validation-selection utilities for Phase 4D.

Phase 4D exhaustively enumerates the frozen classical candidate grid
declared in Phase 4A and implemented in ``classical.py``.

This module contains deterministic candidate enumeration and, later,
pure validation-selection logic. It does not load datasets and therefore
cannot access locked TEST targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite

from plasma_ai.surrogate.classical import (
    EXTRA_TREES_FROZEN_SEARCH_SPACE,
    EXTRA_TREES_MODEL,
    HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE,
    HIST_GRADIENT_BOOSTING_MODEL,
    POLYNOMIAL_DEGREES,
    POLYNOMIAL_MODEL,
    ClassicalCandidateSpec,
    ClassicalValidationResult,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)


def enumerate_phase4d_classical_specs(
) -> tuple[ClassicalCandidateSpec, ...]:
    """Enumerate the exact frozen Phase 4D classical search grid.

    The ordering is canonical and deterministic:

    1. polynomial degree;
    2. Extra Trees:
       n_estimators, max_depth, min_samples_leaf, max_features;
    3. HistGradientBoosting:
       learning_rate, max_iter, max_leaf_nodes, l2_regularization.

    Returns
    -------
    tuple[ClassicalCandidateSpec, ...]
        Exactly 36 uniquely identified candidate configurations.
    """
    specs: list[ClassicalCandidateSpec] = []

    for degree in POLYNOMIAL_DEGREES:
        specs.append(
            ClassicalCandidateSpec(
                candidate_id=(
                    f"polynomial_degree_{degree}"
                ),
                model_name=POLYNOMIAL_MODEL,
                parameters={
                    "degree": degree,
                },
            )
        )

    extra_trees = (
        EXTRA_TREES_FROZEN_SEARCH_SPACE
    )

    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        max_features,
    ) in product(
        extra_trees["n_estimators"],
        extra_trees["max_depth"],
        extra_trees["min_samples_leaf"],
        extra_trees["max_features"],
    ):
        depth_id = (
            "none"
            if max_depth is None
            else str(max_depth)
        )

        specs.append(
            ClassicalCandidateSpec(
                candidate_id=(
                    "extra_trees"
                    f"_n{n_estimators}"
                    f"_depth{depth_id}"
                    f"_leaf{min_samples_leaf}"
                    f"_features{max_features:g}"
                ),
                model_name=EXTRA_TREES_MODEL,
                parameters={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                },
            )
        )

    hist_gradient_boosting = (
        HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE
    )

    for (
        learning_rate,
        max_iter,
        max_leaf_nodes,
        l2_regularization,
    ) in product(
        hist_gradient_boosting["learning_rate"],
        hist_gradient_boosting["max_iter"],
        hist_gradient_boosting["max_leaf_nodes"],
        hist_gradient_boosting[
            "l2_regularization"
        ],
    ):
        specs.append(
            ClassicalCandidateSpec(
                candidate_id=(
                    "hist_gradient_boosting"
                    f"_lr{learning_rate:g}"
                    f"_iter{max_iter}"
                    f"_leaves{max_leaf_nodes}"
                    f"_l2{l2_regularization:g}"
                ),
                model_name=(
                    HIST_GRADIENT_BOOSTING_MODEL
                ),
                parameters={
                    "learning_rate": learning_rate,
                    "max_iter": max_iter,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": (
                        l2_regularization
                    ),
                },
            )
        )

    if len(specs) != 36:
        raise RuntimeError(
            "Frozen Phase 4D classical grid must "
            "contain exactly 36 candidates; "
            f"observed {len(specs)}."
        )

    candidate_ids = {
        spec.candidate_id
        for spec in specs
    }

    if len(candidate_ids) != len(specs):
        raise RuntimeError(
            "Frozen Phase 4D candidate IDs must "
            "be unique."
        )

    return tuple(specs)



def primary_equivalence_indices(
    values,
    *,
    relative_fraction: float,
) -> tuple[int, ...]:
    """Return candidates inside the strict primary-metric equivalence band.

    Phase 4D treats a primary-metric difference as practically marginal
    only when it is strictly smaller than the frozen relative fraction
    measured from the globally best score.

    Primary Phase 4 metrics are error metrics, so lower is better.

    If the best score is exactly zero, only candidates with the same
    zero score can be practically equivalent.
    """
    scores = tuple(
        float(value)
        for value in values
    )

    if not scores:
        raise ValueError(
            "Primary metric values must not be empty."
        )

    if not isfinite(relative_fraction):
        raise ValueError(
            "Practical-equivalence fraction must be finite."
        )

    if relative_fraction < 0.0:
        raise ValueError(
            "Practical-equivalence fraction must be non-negative."
        )

    if any(
        not isfinite(value)
        for value in scores
    ):
        raise ValueError(
            "Primary metric values must all be finite."
        )

    if any(
        value < 0.0
        for value in scores
    ):
        raise ValueError(
            "Primary error metrics must be non-negative."
        )

    best = min(scores)

    if best == 0.0:
        return tuple(
            index
            for index, value in enumerate(scores)
            if value == 0.0
        )

    strict_upper_boundary = (
        best * (1.0 + relative_fraction)
    )

    return tuple(
        index
        for index, value in enumerate(scores)
        if (
            value == best
            or value < strict_upper_boundary
        )
    )



@dataclass(frozen=True)
class ValidationSelectionDecision:
    """Auditable result of the frozen Phase 4D selection policy."""

    selected_result: ClassicalValidationResult
    selection_stage: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    primary_equivalent_candidate_ids: tuple[str, ...]
    primary_equivalent_candidates: tuple[dict[str, str], ...]
    best_primary_value: float
    practical_equivalence_relative_fraction: float
    primary_strict_upper_boundary: float


_SELECTION_METRICS = {
    DENSITY_TARGET: {
        "primary": "p95_absolute_relative_error",
        "secondary": (
            "median_absolute_relative_error",
            "rmse",
        ),
    },
    TEMPERATURE_TARGET: {
        "primary": "rmse_eV",
        "secondary": (
            "p95_absolute_error_eV",
            "mae_eV",
        ),
    },
}


_MODEL_FAMILY_SIMPLICITY_ORDER = {
    POLYNOMIAL_MODEL: 0,
    EXTRA_TREES_MODEL: 1,
    HIST_GRADIENT_BOOSTING_MODEL: 2,
}


_TRANSFORM_CANONICAL_ORDER = {
    "identity": 0,
    "log10": 1,
}


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
        result.metrics[metric_name]
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


def _canonical_candidate_order(
) -> dict[str, int]:
    return {
        spec.candidate_id: index
        for index, spec
        in enumerate(
            enumerate_phase4d_classical_specs()
        )
    }


def _simplicity_key(
    result: ClassicalValidationResult,
) -> tuple[int, int]:
    if (
        result.model_name
        not in _MODEL_FAMILY_SIMPLICITY_ORDER
    ):
        raise ValueError(
            "Unknown model family in Phase 4D "
            f"selection: {result.model_name!r}."
        )

    family_rank = (
        _MODEL_FAMILY_SIMPLICITY_ORDER[
            result.model_name
        ]
    )

    if result.model_name == POLYNOMIAL_MODEL:
        degree = result.parameters.get("degree")

        if degree not in POLYNOMIAL_DEGREES:
            raise ValueError(
                "Polynomial candidate has an invalid "
                f"frozen degree: {degree!r}."
            )

        within_family_rank = (
            POLYNOMIAL_DEGREES.index(degree)
        )
    else:
        # Phase 4A declares no scientific complexity ranking
        # among hyperparameter settings inside these families.
        within_family_rank = 0

    return (
        family_rank,
        within_family_rank,
    )


def _canonical_fallback_key(
    result: ClassicalValidationResult,
) -> tuple[int, int, str]:
    canonical_order = _canonical_candidate_order()

    if result.candidate_id not in canonical_order:
        raise ValueError(
            "Canonical fallback requires a candidate "
            "from the frozen Phase 4D grid; unknown "
            f"candidate ID {result.candidate_id!r}."
        )

    if (
        result.transform_name
        not in _TRANSFORM_CANONICAL_ORDER
    ):
        raise ValueError(
            "Unknown transform in Phase 4D canonical "
            f"fallback: {result.transform_name!r}."
        )

    return (
        canonical_order[result.candidate_id],
        _TRANSFORM_CANONICAL_ORDER[
            result.transform_name
        ],
        result.candidate_id,
    )


def select_validation_candidate(
    results,
    *,
    target_name: str,
    practical_equivalence_relative_fraction: float,
) -> ValidationSelectionDecision:
    """Apply the frozen Phase 4D validation-selection policy.

    The primary error metric defines the practical-equivalence cohort.
    Candidates outside the strict frozen relative band cannot be
    recovered by secondary metrics or simplicity.

    Inside that cohort, the declared secondary metrics are applied
    lexicographically. Simplicity is used only if the declared metrics
    remain exactly tied. Canonical grid order is the final deterministic
    reproducibility fallback.
    """
    candidates = tuple(results)

    if not candidates:
        raise ValueError(
            "Phase 4D selection requires at least one candidate."
        )

    if target_name not in _SELECTION_METRICS:
        raise KeyError(
            f"Unknown Phase 4 target {target_name!r}."
        )

    for result in candidates:
        if result.target_name != target_name:
            raise ValueError(
                "All selection candidates must match "
                f"target {target_name!r}; observed "
                f"{result.target_name!r}."
            )

    metric_policy = _SELECTION_METRICS[
        target_name
    ]

    primary_metric = metric_policy["primary"]
    secondary_metrics = tuple(
        metric_policy["secondary"]
    )

    primary_values = tuple(
        _validated_metric_value(
            result,
            primary_metric,
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
                + practical_equivalence_relative_fraction
            )
        )

    equivalent_indices = (
        primary_equivalence_indices(
            primary_values,
            relative_fraction=(
                practical_equivalence_relative_fraction
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

    if len(cohort) == 1:
        return ValidationSelectionDecision(
            selected_result=cohort[0],
            selection_stage="primary_metric",
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            primary_equivalent_candidate_ids=(
                primary_equivalent_candidate_ids
            ),
            primary_equivalent_candidates=(
                primary_equivalent_candidates
            ),
            best_primary_value=best_primary_value,
            practical_equivalence_relative_fraction=(
                practical_equivalence_relative_fraction
            ),
            primary_strict_upper_boundary=(
                primary_strict_upper_boundary
            ),
        )

    for metric_name in secondary_metrics:
        values = tuple(
            _validated_metric_value(
                result,
                metric_name,
            )
            for result in cohort
        )

        best_value = min(values)

        narrowed = tuple(
            result
            for result, value
            in zip(
                cohort,
                values,
                strict=True,
            )
            if value == best_value
        )

        if len(narrowed) == 1:
            return ValidationSelectionDecision(
                selected_result=narrowed[0],
                selection_stage=(
                    f"secondary_metric:{metric_name}"
                ),
                primary_metric=primary_metric,
                secondary_metrics=secondary_metrics,
                primary_equivalent_candidate_ids=(
                    primary_equivalent_candidate_ids
                ),
                primary_equivalent_candidates=(
                    primary_equivalent_candidates
                ),
                best_primary_value=best_primary_value,
                practical_equivalence_relative_fraction=(
                    practical_equivalence_relative_fraction
                ),
                primary_strict_upper_boundary=(
                    primary_strict_upper_boundary
                ),
            )

        cohort = narrowed

    simplicity_keys = tuple(
        _simplicity_key(result)
        for result in cohort
    )

    best_simplicity = min(
        simplicity_keys
    )

    simplest = tuple(
        result
        for result, key
        in zip(
            cohort,
            simplicity_keys,
            strict=True,
        )
        if key == best_simplicity
    )

    if len(simplest) == 1:
        return ValidationSelectionDecision(
            selected_result=simplest[0],
            selection_stage="simplicity",
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            primary_equivalent_candidate_ids=(
                primary_equivalent_candidate_ids
            ),
            primary_equivalent_candidates=(
                primary_equivalent_candidates
            ),
            best_primary_value=best_primary_value,
            practical_equivalence_relative_fraction=(
                practical_equivalence_relative_fraction
            ),
            primary_strict_upper_boundary=(
                primary_strict_upper_boundary
            ),
        )

    selected = min(
        simplest,
        key=_canonical_fallback_key,
    )

    return ValidationSelectionDecision(
        selected_result=selected,
        selection_stage="canonical_fallback",
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        primary_equivalent_candidate_ids=(
            primary_equivalent_candidate_ids
        ),
        primary_equivalent_candidates=(
            primary_equivalent_candidates
        ),
        best_primary_value=best_primary_value,
        practical_equivalence_relative_fraction=(
            practical_equivalence_relative_fraction
        ),
        primary_strict_upper_boundary=(
            primary_strict_upper_boundary
        ),
    )
