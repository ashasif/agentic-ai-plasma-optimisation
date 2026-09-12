"""Tests for pure Phase 4F locked-evaluation analysis helpers.

These tests use synthetic arrays only. They do not load the Phase 4
dataset and cannot access TEST targets.
"""

import numpy as np
import pytest

from plasma_ai.surrogate.metrics import (
    DENSITY_TARGET,
    TEMPERATURE_TARGET,
    regression_metrics,
)
from plasma_ai.surrogate.phase4f_analysis import (
    build_error_space_masks,
    error_space_summary,
    normalised_coordinates,
    prediction_integrity,
    regional_metric_summary,
    worst_case_observations,
)


def _physical_features(
    power_norm,
    pressure_norm,
):
    power_norm = np.asarray(
        power_norm,
        dtype=np.float64,
    )

    pressure_norm = np.asarray(
        pressure_norm,
        dtype=np.float64,
    )

    return np.column_stack(
        (
            15.0 + 75.0 * power_norm,
            10.0 + 50.0 * pressure_norm,
        )
    )


def test_normalised_coordinates_exact_domain_points():
    X = np.asarray(
        [
            [15.0, 10.0],
            [52.5, 35.0],
            [90.0, 60.0],
        ],
        dtype=np.float64,
    )

    power_norm, pressure_norm = (
        normalised_coordinates(
            X
        )
    )

    np.testing.assert_allclose(
        power_norm,
        [0.0, 0.5, 1.0],
    )

    np.testing.assert_allclose(
        pressure_norm,
        [0.0, 0.5, 1.0],
    )


def test_normalised_coordinates_reject_out_of_domain_point():
    X = np.asarray(
        [
            [14.0, 35.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="outside the frozen physical domain",
    ):
        normalised_coordinates(
            X
        )


def test_boundary_interior_and_central_rules_are_frozen():
    X = _physical_features(
        [
            0.0,
            0.10,
            0.1001,
            0.20,
            0.50,
            0.80,
            0.8999,
            0.90,
            1.0,
        ],
        [0.50] * 9,
    )

    masks = build_error_space_masks(
        X
    )

    np.testing.assert_array_equal(
        masks["near_boundary"],
        [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    )

    np.testing.assert_array_equal(
        masks["interior"],
        ~masks["near_boundary"],
    )

    np.testing.assert_array_equal(
        masks["central_domain"],
        [
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
        ],
    )


def test_range_thirds_use_frozen_endpoint_rules():
    eps = 1.0e-6

    X = _physical_features(
        [
            0.0,
            1.0 / 3.0 - eps,
            1.0 / 3.0,
            2.0 / 3.0 - eps,
            2.0 / 3.0,
            1.0,
        ],
        [0.5] * 6,
    )

    masks = build_error_space_masks(
        X
    )

    np.testing.assert_array_equal(
        masks["power_low"],
        [
            True,
            True,
            False,
            False,
            False,
            False,
        ],
    )

    np.testing.assert_array_equal(
        masks["power_middle"],
        [
            False,
            False,
            True,
            True,
            False,
            False,
        ],
    )

    np.testing.assert_array_equal(
        masks["power_high"],
        [
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    )

    membership_count = (
        masks["power_low"].astype(int)
        + masks["power_middle"].astype(int)
        + masks["power_high"].astype(int)
    )

    np.testing.assert_array_equal(
        membership_count,
        np.ones(
            6,
            dtype=int,
        ),
    )


def test_corner_masks_match_cartesian_low_high_thirds():
    X = _physical_features(
        [
            0.1,
            0.1,
            0.9,
            0.9,
            0.5,
        ],
        [
            0.1,
            0.9,
            0.1,
            0.9,
            0.5,
        ],
    )

    masks = build_error_space_masks(
        X
    )

    assert np.flatnonzero(
        masks["corner_low_power_low_pressure"]
    ).tolist() == [0]

    assert np.flatnonzero(
        masks["corner_low_power_high_pressure"]
    ).tolist() == [1]

    assert np.flatnonzero(
        masks["corner_high_power_low_pressure"]
    ).tolist() == [2]

    assert np.flatnonzero(
        masks["corner_high_power_high_pressure"]
    ).tolist() == [3]


def test_regional_metric_summary_reuses_frozen_metrics():
    truth = np.asarray(
        [10.0, 20.0, 30.0, 40.0],
        dtype=np.float64,
    )

    prediction = np.asarray(
        [11.0, 19.0, 33.0, 38.0],
        dtype=np.float64,
    )

    mask = np.asarray(
        [True, True, True, True],
        dtype=bool,
    )

    observed = regional_metric_summary(
        DENSITY_TARGET,
        truth,
        prediction,
        mask,
    )

    expected = regression_metrics(
        DENSITY_TARGET,
        truth,
        prediction,
    )

    assert observed["rows"] == 4
    assert observed["metrics"] == expected


def test_complete_error_space_summary_partitions_rows():
    values = np.asarray(
        [
            0.05,
            0.15,
            0.25,
            0.40,
            0.50,
            0.60,
            0.75,
            0.85,
            0.95,
        ],
        dtype=np.float64,
    )

    power_norm, pressure_norm = np.meshgrid(
        values,
        values,
        indexing="ij",
    )

    X = _physical_features(
        power_norm.ravel(),
        pressure_norm.ravel(),
    )

    truth = (
        1.0
        + X[:, 0] / 100.0
        + X[:, 1] / 100.0
    )

    prediction = (
        truth
        * (
            1.0
            + 0.01
            * np.sin(
                np.arange(
                    truth.size,
                    dtype=np.float64,
                )
            )
        )
    )

    summary = error_space_summary(
        DENSITY_TARGET,
        X,
        truth,
        prediction,
    )

    expected_regions = {
        "near_boundary",
        "interior",
        "central_domain",
        "power_low",
        "power_middle",
        "power_high",
        "pressure_low",
        "pressure_middle",
        "pressure_high",
        "corner_low_power_low_pressure",
        "corner_low_power_high_pressure",
        "corner_high_power_low_pressure",
        "corner_high_power_high_pressure",
    }

    assert set(
        summary
    ) == expected_regions

    assert (
        summary["near_boundary"]["rows"]
        + summary["interior"]["rows"]
        == 81
    )

    assert (
        summary["power_low"]["rows"]
        + summary["power_middle"]["rows"]
        + summary["power_high"]["rows"]
        == 81
    )

    assert (
        summary["pressure_low"]["rows"]
        + summary["pressure_middle"]["rows"]
        + summary["pressure_high"]["rows"]
        == 81
    )


def test_prediction_integrity_reports_positive_finite_predictions():
    result = prediction_integrity(
        np.asarray(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        )
    )

    assert result == {
        "prediction_count": 3,
        "all_predictions_finite": True,
        "prediction_minimum": 1.0,
        "prediction_maximum": 3.0,
        "all_predictions_strictly_positive": True,
    }


def test_prediction_integrity_reports_nonfinite_vector_without_hiding_it():
    result = prediction_integrity(
        np.asarray(
            [1.0, np.nan, 3.0],
            dtype=np.float64,
        )
    )

    assert result["prediction_count"] == 3
    assert result["all_predictions_finite"] is False
    assert result["all_predictions_strictly_positive"] is False
    assert result["prediction_minimum"] == 1.0
    assert result["prediction_maximum"] == 3.0


def test_density_worst_case_ranking_uses_relative_error_and_split_index_tie_break():
    X = _physical_features(
        [0.2, 0.4, 0.6, 0.8],
        [0.5, 0.5, 0.5, 0.5],
    )

    truth = np.asarray(
        [100.0, 100.0, 100.0, 100.0],
        dtype=np.float64,
    )

    prediction = np.asarray(
        [110.0, 130.0, 120.0, 130.0],
        dtype=np.float64,
    )

    worst = worst_case_observations(
        DENSITY_TARGET,
        X,
        truth,
        prediction,
        count=4,
    )

    assert [
        row["test_split_index"]
        for row in worst
    ] == [
        2,
        4,
        3,
        1,
    ]


def test_temperature_worst_case_ranking_uses_absolute_error_ev():
    X = _physical_features(
        [0.2, 0.5, 0.8],
        [0.5, 0.5, 0.5],
    )

    truth = np.asarray(
        [1.0, 100.0, 1.0],
        dtype=np.float64,
    )

    prediction = np.asarray(
        [1.5, 110.0, 3.0],
        dtype=np.float64,
    )

    worst = worst_case_observations(
        TEMPERATURE_TARGET,
        X,
        truth,
        prediction,
        count=3,
    )

    assert [
        row["test_split_index"]
        for row in worst
    ] == [
        2,
        3,
        1,
    ]


def test_worst_case_unknown_target_is_rejected():
    X = _physical_features(
        [0.2, 0.8],
        [0.5, 0.5],
    )

    truth = np.asarray(
        [1.0, 2.0],
        dtype=np.float64,
    )

    prediction = np.asarray(
        [1.1, 2.2],
        dtype=np.float64,
    )

    with pytest.raises(
        KeyError,
        match="Unknown Phase 4 target",
    ):
        worst_case_observations(
            "unknown_target",
            X,
            truth,
            prediction,
        )
