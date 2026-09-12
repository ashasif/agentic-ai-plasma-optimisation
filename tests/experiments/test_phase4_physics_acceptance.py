"""Tests for approved Phase 4E physics-aware acceptance semantics."""

from __future__ import annotations

import numpy as np
import pytest

from plasma_ai.surrogate.physics_acceptance import (
    check_strict_positivity,
    compare_source_referenced_oscillation,
    compare_source_referenced_trend,
)


def test_strict_positivity_passes_positive_finite_values():
    result = check_strict_positivity(
        np.asarray(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    assert result.total_points == 3
    assert result.nonfinite_count == 0
    assert result.nonpositive_count == 0
    assert result.passed is True


def test_strict_positivity_rejects_zero_negative_and_nonfinite():
    result = check_strict_positivity(
        np.asarray(
            [
                1.0,
                0.0,
                -2.0,
                np.nan,
                np.inf,
            ]
        )
    )

    assert result.total_points == 5
    assert result.nonfinite_count == 2
    assert result.nonpositive_count == 2
    assert result.passed is False


def test_trend_check_passes_when_surrogate_matches_source_direction():
    source = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )

    surrogate = np.asarray(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
        ]
    )

    result = compare_source_referenced_trend(
        source,
        surrogate,
        axis=0,
    )

    assert result.total_intervals == 4
    assert result.unsupported_reversal_count == 0
    assert result.unsupported_reversal_locations == ()
    assert result.passed is True


def test_trend_check_counts_strict_opposite_direction():
    source = np.asarray(
        [
            [1.0, 5.0, 10.0],
            [2.0, 4.0, 11.0],
        ]
    )

    surrogate = np.asarray(
        [
            [2.0, 5.0, 20.0],
            [1.0, 6.0, 20.0],
        ]
    )

    result = compare_source_referenced_trend(
        source,
        surrogate,
        axis=0,
    )

    assert result.total_intervals == 3
    assert result.unsupported_reversal_count == 2
    assert result.unsupported_reversal_locations == (
        (0, 0),
        (0, 1),
    )
    assert result.passed is False


def test_trend_check_does_not_penalise_zero_source_or_surrogate_delta():
    source = np.asarray(
        [
            [1.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
        ]
    )

    surrogate = np.asarray(
        [
            [5.0, 8.0, 7.0],
            [4.0, 8.0, 7.0],
        ]
    )

    result = compare_source_referenced_trend(
        source,
        surrogate,
        axis=0,
    )

    # Column 0 has source delta zero, so the surrogate direction is ignored.
    # Columns 1 and 2 have surrogate delta zero, so neither is a reversal.
    assert result.unsupported_reversal_count == 0
    assert result.passed is True


def test_oscillation_check_detects_source_unsupported_turn():
    source = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
        ]
    )

    surrogate = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 11.0],
            [2.0, 12.0],
            [4.0, 13.0],
        ]
    )

    result = compare_source_referenced_oscillation(
        source,
        surrogate,
        axis=0,
    )

    assert result.source_turning_point_count == 0
    assert result.surrogate_turning_point_count == 2
    assert result.spurious_turning_point_count == 2
    assert result.spurious_turning_point_locations == (
        (1, 0),
        (2, 0),
    )
    assert result.passed is False


def test_oscillation_check_does_not_penalise_source_supported_turn():
    source = np.asarray(
        [
            [1.0],
            [3.0],
            [2.0],
        ]
    )

    surrogate = np.asarray(
        [
            [10.0],
            [20.0],
            [15.0],
        ]
    )

    result = compare_source_referenced_oscillation(
        source,
        surrogate,
        axis=0,
    )

    assert result.source_turning_point_count == 1
    assert result.surrogate_turning_point_count == 1
    assert result.spurious_turning_point_count == 0
    assert result.passed is True


def test_zero_increment_does_not_form_strict_turning_point():
    source = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )

    surrogate = np.asarray(
        [
            [1.0, 4.0, 1.0],
            [2.0, 4.0, 2.0],
            [2.0, 4.0, 3.0],
        ]
    )

    result = compare_source_referenced_oscillation(
        source,
        surrogate,
        axis=0,
    )

    assert result.surrogate_turning_point_count == 0
    assert result.spurious_turning_point_count == 0
    assert result.passed is True


def test_pressure_axis_turn_location_maps_to_original_grid_index():
    source = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    surrogate = np.asarray(
        [
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    result = compare_source_referenced_oscillation(
        source,
        surrogate,
        axis=1,
    )

    assert result.spurious_turning_point_locations == (
        (0, 1),
        (0, 2),
    )


def test_trend_and_oscillation_reject_shape_mismatch():
    source = np.ones(
        (3, 3)
    )
    surrogate = np.ones(
        (3, 2)
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        compare_source_referenced_trend(
            source,
            surrogate,
            axis=0,
        )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        compare_source_referenced_oscillation(
            source,
            surrogate,
            axis=1,
        )


def test_trend_and_oscillation_reject_nonfinite_inputs():
    source = np.ones(
        (3, 3)
    )
    surrogate = np.ones(
        (3, 3)
    )

    surrogate[1, 1] = np.nan

    with pytest.raises(
        ValueError,
        match="Surrogate diagnostic array contains non-finite values",
    ):
        compare_source_referenced_trend(
            source,
            surrogate,
            axis=0,
        )

    with pytest.raises(
        ValueError,
        match="Surrogate diagnostic array contains non-finite values",
    ):
        compare_source_referenced_oscillation(
            source,
            surrogate,
            axis=1,
        )
