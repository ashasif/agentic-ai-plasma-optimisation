"""Tests for Phase 4D controlled validation selection."""

from __future__ import annotations

from plasma_ai.surrogate.classical import (
    EXTRA_TREES_FROZEN_SEARCH_SPACE,
    EXTRA_TREES_MODEL,
    HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE,
    HIST_GRADIENT_BOOSTING_MODEL,
    POLYNOMIAL_DEGREES,
    POLYNOMIAL_MODEL,
    ClassicalValidationResult,
    build_classical_candidate,
)
from plasma_ai.surrogate.selection import (
    enumerate_phase4d_classical_specs,
    primary_equivalence_indices,
    select_validation_candidate,
)


def test_phase4d_frozen_grid_has_exactly_36_candidates():
    specs = enumerate_phase4d_classical_specs()

    assert len(specs) == 36
    assert len({
        spec.candidate_id
        for spec in specs
    }) == 36


def test_phase4d_frozen_grid_family_counts():
    specs = enumerate_phase4d_classical_specs()

    counts = {
        POLYNOMIAL_MODEL: 0,
        EXTRA_TREES_MODEL: 0,
        HIST_GRADIENT_BOOSTING_MODEL: 0,
    }

    for spec in specs:
        counts[spec.model_name] += 1

    assert counts == {
        POLYNOMIAL_MODEL: 2,
        EXTRA_TREES_MODEL: 18,
        HIST_GRADIENT_BOOSTING_MODEL: 16,
    }


def test_phase4d_grid_enumeration_is_deterministic():
    first = enumerate_phase4d_classical_specs()
    second = enumerate_phase4d_classical_specs()

    assert first == second



def test_phase4d_polynomial_grid_matches_frozen_degrees():
    specs = enumerate_phase4d_classical_specs()

    observed = {
        spec.parameters["degree"]
        for spec in specs
        if spec.model_name == POLYNOMIAL_MODEL
    }

    assert observed == set(POLYNOMIAL_DEGREES)


def test_phase4d_extra_trees_grid_matches_exact_cartesian_product():
    specs = enumerate_phase4d_classical_specs()

    observed = {
        (
            spec.parameters["n_estimators"],
            spec.parameters["max_depth"],
            spec.parameters["min_samples_leaf"],
            spec.parameters["max_features"],
        )
        for spec in specs
        if spec.model_name == EXTRA_TREES_MODEL
    }

    expected = {
        (
            n_estimators,
            max_depth,
            min_samples_leaf,
            max_features,
        )
        for n_estimators
        in EXTRA_TREES_FROZEN_SEARCH_SPACE["n_estimators"]
        for max_depth
        in EXTRA_TREES_FROZEN_SEARCH_SPACE["max_depth"]
        for min_samples_leaf
        in EXTRA_TREES_FROZEN_SEARCH_SPACE["min_samples_leaf"]
        for max_features
        in EXTRA_TREES_FROZEN_SEARCH_SPACE["max_features"]
    }

    assert observed == expected
    assert len(observed) == 18


def test_phase4d_hist_gradient_boosting_grid_matches_exact_cartesian_product():
    specs = enumerate_phase4d_classical_specs()

    observed = {
        (
            spec.parameters["learning_rate"],
            spec.parameters["max_iter"],
            spec.parameters["max_leaf_nodes"],
            spec.parameters["l2_regularization"],
        )
        for spec in specs
        if spec.model_name == HIST_GRADIENT_BOOSTING_MODEL
    }

    expected = {
        (
            learning_rate,
            max_iter,
            max_leaf_nodes,
            l2_regularization,
        )
        for learning_rate
        in HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE["learning_rate"]
        for max_iter
        in HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE["max_iter"]
        for max_leaf_nodes
        in HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE["max_leaf_nodes"]
        for l2_regularization
        in HIST_GRADIENT_BOOSTING_FROZEN_SEARCH_SPACE[
            "l2_regularization"
        ]
    }

    assert observed == expected
    assert len(observed) == 16


def test_every_phase4d_spec_is_accepted_by_frozen_model_builder():
    specs = enumerate_phase4d_classical_specs()

    for spec in specs:
        model = build_classical_candidate(spec)
        assert model is not None



def test_primary_equivalence_uses_strict_two_percent_band():
    values = (
        1.0,
        1.0199,
        1.02,
        1.0201,
        1.50,
    )

    indices = primary_equivalence_indices(
        values,
        relative_fraction=0.02,
    )

    assert indices == (0, 1)


def test_primary_equivalence_is_relative_to_global_best():
    values = (
        10.15,
        10.00,
        10.19,
        10.20,
    )

    indices = primary_equivalence_indices(
        values,
        relative_fraction=0.02,
    )

    assert indices == (0, 1, 2)


def test_primary_equivalence_handles_zero_best_without_division():
    values = (
        0.0,
        0.0,
        1.0e-12,
    )

    indices = primary_equivalence_indices(
        values,
        relative_fraction=0.02,
    )

    assert indices == (0, 1)


def test_primary_equivalence_rejects_invalid_inputs():
    import pytest

    with pytest.raises(ValueError):
        primary_equivalence_indices(
            (),
            relative_fraction=0.02,
        )

    with pytest.raises(ValueError):
        primary_equivalence_indices(
            (1.0, float("nan")),
            relative_fraction=0.02,
        )

    with pytest.raises(ValueError):
        primary_equivalence_indices(
            (1.0, 2.0),
            relative_fraction=-0.01,
        )



def _selection_result(
    *,
    target_name,
    transform_name,
    candidate_id,
    model_name,
    parameters,
    primary,
    secondary_1,
    secondary_2,
):
    if target_name == "true_electron_density_m3":
        metrics = {
            "mae": 1.0,
            "rmse": secondary_2,
            "r2": 0.99,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": secondary_1,
            "p95_absolute_relative_error": primary,
            "max_absolute_relative_error": 0.05,
        }
    else:
        metrics = {
            "mae_eV": secondary_2,
            "rmse_eV": primary,
            "r2": 0.99,
            "mean_absolute_relative_error": 0.01,
            "median_absolute_relative_error": 0.01,
            "p95_absolute_error_eV": secondary_1,
            "max_absolute_error_eV": 0.05,
        }

    return ClassicalValidationResult(
        target_name=target_name,
        transform_name=transform_name,
        candidate_id=candidate_id,
        model_name=model_name,
        parameters=parameters,
        metrics=metrics,
        prediction_sanity={
            "all_finite": True,
            "minimum": 1.0,
            "maximum": 2.0,
            "negative_count": 0,
            "non_positive_count": 0,
        },
    )


def test_density_material_primary_advantage_cannot_be_overridden():
    polynomial = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01021,
        secondary_1=0.0001,
        secondary_2=1.0,
    )

    extra_trees = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="extra_trees_test",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        primary=0.01000,
        secondary_1=0.0100,
        secondary_2=10.0,
    )

    decision = select_validation_candidate(
        (polynomial, extra_trees),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is extra_trees
    assert decision.selection_stage == "primary_metric"


def test_density_secondary_one_selects_inside_primary_band():
    polynomial = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01010,
        secondary_1=0.0010,
        secondary_2=5.0,
    )

    extra_trees = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="extra_trees_test",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        primary=0.01000,
        secondary_1=0.0020,
        secondary_2=1.0,
    )

    decision = select_validation_candidate(
        (extra_trees, polynomial),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is polynomial
    assert decision.selection_stage == (
        "secondary_metric:"
        "median_absolute_relative_error"
    )


def test_density_secondary_two_is_used_after_secondary_one_tie():
    polynomial = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01010,
        secondary_1=0.0010,
        secondary_2=1.0,
    )

    extra_trees = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="extra_trees_test",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        primary=0.01000,
        secondary_1=0.0010,
        secondary_2=2.0,
    )

    decision = select_validation_candidate(
        (extra_trees, polynomial),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is polynomial
    assert decision.selection_stage == "secondary_metric:rmse"


def test_temperature_uses_frozen_secondary_metric_order():
    polynomial = _selection_result(
        target_name="true_electron_temperature_eV",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.00101,
        secondary_1=0.0015,
        secondary_2=0.0008,
    )

    extra_trees = _selection_result(
        target_name="true_electron_temperature_eV",
        transform_name="identity",
        candidate_id="extra_trees_test",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        primary=0.00100,
        secondary_1=0.0020,
        secondary_2=0.0005,
    )

    decision = select_validation_candidate(
        (extra_trees, polynomial),
        target_name="true_electron_temperature_eV",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is polynomial
    assert decision.selection_stage == (
        "secondary_metric:p95_absolute_error_eV"
    )


def test_family_simplicity_is_used_only_after_metric_tie():
    polynomial = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_3",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 3},
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    extra_trees = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="extra_trees_test",
        model_name=EXTRA_TREES_MODEL,
        parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    decision = select_validation_candidate(
        (extra_trees, polynomial),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is polynomial
    assert decision.selection_stage == "simplicity"


def test_polynomial_degree_two_wins_exact_metric_tie():
    degree_2 = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    degree_3 = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_3",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 3},
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    decision = select_validation_candidate(
        (degree_3, degree_2),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is degree_2
    assert decision.selection_stage == "simplicity"


def test_same_family_exact_tie_uses_canonical_grid_order():
    specs = enumerate_phase4d_classical_specs()

    first_spec = specs[2]
    second_spec = specs[3]

    first = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id=first_spec.candidate_id,
        model_name=first_spec.model_name,
        parameters=dict(first_spec.parameters),
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    second = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id=second_spec.candidate_id,
        model_name=second_spec.model_name,
        parameters=dict(second_spec.parameters),
        primary=0.01,
        secondary_1=0.001,
        secondary_2=1.0,
    )

    decision = select_validation_candidate(
        (second, first),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.selected_result is first
    assert decision.selection_stage == "canonical_fallback"



def test_density_decision_trace_distinguishes_transform_candidates():
    identity = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01000,
        secondary_1=0.0010,
        secondary_2=1.0,
    )

    log10 = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="log10",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01010,
        secondary_1=0.0020,
        secondary_2=2.0,
    )

    decision = select_validation_candidate(
        (identity, log10),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.primary_equivalent_candidates == (
        {
            "candidate_id": "polynomial_degree_2",
            "transform": "identity",
        },
        {
            "candidate_id": "polynomial_degree_2",
            "transform": "log10",
        },
    )



def test_selection_decision_records_primary_equivalence_boundary():
    best = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="identity",
        candidate_id="polynomial_degree_2",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 2},
        primary=0.01000,
        secondary_1=0.0020,
        secondary_2=2.0,
    )

    equivalent = _selection_result(
        target_name="true_electron_density_m3",
        transform_name="log10",
        candidate_id="polynomial_degree_3",
        model_name=POLYNOMIAL_MODEL,
        parameters={"degree": 3},
        primary=0.01010,
        secondary_1=0.0010,
        secondary_2=1.0,
    )

    decision = select_validation_candidate(
        (best, equivalent),
        target_name="true_electron_density_m3",
        practical_equivalence_relative_fraction=0.02,
    )

    assert decision.best_primary_value == 0.01000
    assert decision.practical_equivalence_relative_fraction == 0.02
    assert decision.primary_strict_upper_boundary == 0.01020
