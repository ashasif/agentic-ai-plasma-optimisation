"""Tests for Phase 4E-R density validation selection."""

from __future__ import annotations

import pytest

from plasma_ai.surrogate.classical import (
    ClassicalValidationResult,
)
from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
)
from plasma_ai.surrogate.phase4er_density import (
    PHASE4ER_DENSITY_TRANSFORM,
    enumerate_phase4er_density_specs,
)
from plasma_ai.surrogate.phase4er_selection import (
    PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION,
    PHASE4ER_PRIMARY_METRIC,
    PHASE4ER_SECONDARY_METRICS,
    select_phase4er_density_candidate,
)


def _result(
    spec,
    *,
    p95,
    median,
    rmse,
    target_name=DENSITY_TARGET,
    transform_name=PHASE4ER_DENSITY_TRANSFORM,
):
    return ClassicalValidationResult(
        target_name=target_name,
        transform_name=transform_name,
        candidate_id=spec.candidate_id,
        model_name=spec.model_name,
        parameters=dict(spec.parameters),
        metrics={
            "p95_absolute_relative_error": p95,
            "median_absolute_relative_error": median,
            "rmse": rmse,
        },
        prediction_sanity={
            "all_finite": True,
            "minimum": 1.0,
            "maximum": 2.0,
            "negative_count": 0,
            "non_positive_count": 0,
        },
    )


def _full_grid_results():
    specs = enumerate_phase4er_density_specs()

    return tuple(
        _result(
            spec,
            p95=0.020 + index * 0.001,
            median=0.010 + index * 0.001,
            rmse=1.0e14 + index,
        )
        for index, spec in enumerate(specs)
    )


def test_phase4er_selection_policy_constants():
    assert (
        PHASE4ER_PRACTICAL_EQUIVALENCE_RELATIVE_FRACTION
        == 0.02
    )
    assert PHASE4ER_PRIMARY_METRIC == (
        "p95_absolute_relative_error"
    )
    assert PHASE4ER_SECONDARY_METRICS == (
        "median_absolute_relative_error",
        "rmse",
    )


def test_phase4er_selection_uses_primary_metric_when_unique():
    results = list(
        _full_grid_results()
    )

    selected_spec = (
        enumerate_phase4er_density_specs()[5]
    )

    for index, result in enumerate(results):
        p95 = (
            0.010
            if result.candidate_id
            == selected_spec.candidate_id
            else 0.020 + index * 0.001
        )

        results[index] = _result(
            enumerate_phase4er_density_specs()[index],
            p95=p95,
            median=0.005 + index * 0.001,
            rmse=1.0e14 + index,
        )

    decision = select_phase4er_density_candidate(
        results
    )

    assert (
        decision.selected_result.candidate_id
        == selected_spec.candidate_id
    )
    assert decision.selection_stage == (
        "primary_metric"
    )


def test_phase4er_selection_uses_secondary_metric_inside_2_percent():
    specs = enumerate_phase4er_density_specs()
    results = list(
        _full_grid_results()
    )

    first = specs[2]
    second = specs[7]

    for index, spec in enumerate(specs):
        if spec.candidate_id == first.candidate_id:
            results[index] = _result(
                spec,
                p95=0.0100,
                median=0.0060,
                rmse=2.0e14,
            )
        elif spec.candidate_id == second.candidate_id:
            results[index] = _result(
                spec,
                p95=0.0101,
                median=0.0050,
                rmse=3.0e14,
            )
        else:
            results[index] = _result(
                spec,
                p95=0.0200 + index * 0.001,
                median=0.0100,
                rmse=4.0e14,
            )

    decision = select_phase4er_density_candidate(
        results
    )

    assert (
        decision.selected_result.candidate_id
        == second.candidate_id
    )
    assert decision.selection_stage == (
        "secondary_metric:median_absolute_relative_error"
    )


def test_phase4er_selection_excludes_exact_2_percent_boundary():
    specs = enumerate_phase4er_density_specs()
    results = list(
        _full_grid_results()
    )

    best = specs[1]
    boundary = specs[3]

    for index, spec in enumerate(specs):
        if spec.candidate_id == best.candidate_id:
            results[index] = _result(
                spec,
                p95=0.0100,
                median=0.0090,
                rmse=3.0e14,
            )
        elif spec.candidate_id == boundary.candidate_id:
            results[index] = _result(
                spec,
                p95=0.0102,
                median=0.0010,
                rmse=1.0e14,
            )
        else:
            results[index] = _result(
                spec,
                p95=0.0300 + index * 0.001,
                median=0.0200,
                rmse=5.0e14,
            )

    decision = select_phase4er_density_candidate(
        results
    )

    assert (
        decision.selected_result.candidate_id
        == best.candidate_id
    )

    assert (
        boundary.candidate_id
        not in decision.primary_equivalent_candidate_ids
    )


def test_phase4er_selection_uses_canonical_fallback_for_exact_tie():
    specs = enumerate_phase4er_density_specs()

    results = tuple(
        _result(
            spec,
            p95=0.01,
            median=0.005,
            rmse=1.0e14,
        )
        for spec in specs
    )

    decision = select_phase4er_density_candidate(
        results
    )

    assert (
        decision.selected_result.candidate_id
        == specs[0].candidate_id
    )
    assert decision.selection_stage == (
        "canonical_fallback"
    )


def test_phase4er_selection_rejects_incomplete_grid():
    results = _full_grid_results()[:-1]

    with pytest.raises(
        ValueError,
        match="exactly the frozen 16-candidate",
    ):
        select_phase4er_density_candidate(
            results
        )


def test_phase4er_selection_rejects_wrong_target():
    specs = enumerate_phase4er_density_specs()

    results = [
        _result(
            spec,
            p95=0.01 + index * 0.001,
            median=0.005 + index * 0.001,
            rmse=1.0e14 + index,
        )
        for index, spec in enumerate(specs)
    ]

    results[0] = _result(
        specs[0],
        p95=0.01,
        median=0.005,
        rmse=1.0e14,
        target_name=TEMPERATURE_TARGET,
    )

    with pytest.raises(
        ValueError,
        match="electron density",
    ):
        select_phase4er_density_candidate(
            results
        )


def test_phase4er_selection_rejects_wrong_transform():
    specs = enumerate_phase4er_density_specs()

    results = [
        _result(
            spec,
            p95=0.01 + index * 0.001,
            median=0.005 + index * 0.001,
            rmse=1.0e14 + index,
        )
        for index, spec in enumerate(specs)
    ]

    results[0] = _result(
        specs[0],
        p95=0.01,
        median=0.005,
        rmse=1.0e14,
        transform_name="identity",
    )

    with pytest.raises(
        ValueError,
        match="frozen log10 transform",
    ):
        select_phase4er_density_candidate(
            results
        )
