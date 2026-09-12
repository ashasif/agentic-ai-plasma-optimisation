"""Tests for the Phase 4F locked TEST evaluator.

The evaluator is exercised using a synthetic mocked TEST dataset only.
The real frozen Phase 4 TEST targets are never loaded by this test module.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import plasma_ai.surrogate.phase4f_evaluation as phase4f


class FakeCandidate:
    def __init__(
        self,
        *,
        target_name,
        candidate_id,
        model_name,
        transform_name,
        predictions,
    ):
        self._predictions = np.asarray(
            predictions,
            dtype=np.float64,
        )

        self.fit_rows = 6144

        self.configuration = SimpleNamespace(
            target_name=target_name,
            transform_name=transform_name,
            candidate_spec=SimpleNamespace(
                candidate_id=candidate_id,
                model_name=model_name,
            ),
        )

    def predict_physical(
        self,
        X,
    ):
        matrix = np.asarray(
            X,
            dtype=np.float64,
        )

        assert matrix.shape == (
            2048,
            2,
        )

        return self._predictions.copy()


class FakeDataset:
    def __init__(
        self,
        X,
        y,
    ):
        self.feature_names = (
            "nominal_absorbed_power_W",
            "target_pressure_mTorr",
        )

        self.target_names = (
            phase4f.DENSITY_TARGET,
            phase4f.TEMPERATURE_TARGET,
        )

        self._test = SimpleNamespace(
            X=np.asarray(
                X,
                dtype=np.float64,
            ),
            y=np.asarray(
                y,
                dtype=np.float64,
            ),
        )

    def split(
        self,
        name,
    ):
        assert name == "test"
        return self._test


def _synthetic_test_arrays():
    index = np.arange(
        2048,
        dtype=np.float64,
    )

    power = (
        15.0
        + 75.0
        * (
            (index % 64.0)
            / 63.0
        )
    )

    pressure = (
        10.0
        + 50.0
        * (
            (
                np.floor(
                    index / 32.0
                )
                % 64.0
            )
            / 63.0
        )
    )

    X = np.column_stack(
        (
            power,
            pressure,
        )
    )

    density_truth = (
        1.0e17
        + 8.0e14 * power
        + 3.0e14 * pressure
    )

    temperature_truth = (
        2.2
        - 0.006 * pressure
        + 0.0005 * power
    )

    density_prediction = (
        density_truth
        * (
            1.0
            + 0.01
            * np.sin(
                index / 17.0
            )
        )
    )

    temperature_prediction = (
        temperature_truth
        + 0.01
        * np.cos(
            index / 19.0
        )
    )

    y = np.column_stack(
        (
            density_truth,
            temperature_truth,
        )
    )

    return (
        X,
        y,
        density_prediction,
        temperature_prediction,
    )


def _fake_final_refit(
    density_prediction,
    temperature_prediction,
):
    return SimpleNamespace(
        fit_rows=6144,
        test_targets_accessed=False,
        density=FakeCandidate(
            target_name=phase4f.DENSITY_TARGET,
            candidate_id="frozen_density_candidate",
            model_name="hist_gradient_boosting",
            transform_name="log10",
            predictions=density_prediction,
        ),
        temperature=FakeCandidate(
            target_name=phase4f.TEMPERATURE_TARGET,
            candidate_id="frozen_temperature_candidate",
            model_name="extra_trees",
            transform_name="identity",
            predictions=temperature_prediction,
        ),
    )


def _fake_preflight():
    return {
        "phase": "4E-R",
        "stage": "final_train_validation_refit",
    }


def _fake_structural():
    return {
        "probe_grid": {
            "total_points": 1681,
            "absorbed_power_points": 41,
            "pressure_points": 41,
        },
        "density": {
            "passed": True,
        },
        "temperature": {
            "passed": True,
        },
        "overall_passed": True,
        "interpretation": {
            "selection_gate": False,
            "post_refit_reporting_diagnostic": True,
            "test_targets_used": False,
            "retuning_permitted_from_result": False,
        },
    }


def test_locked_evaluator_uses_only_explicit_test_unlock(
    monkeypatch,
):
    (
        X,
        y,
        density_prediction,
        temperature_prediction,
    ) = _synthetic_test_arrays()

    observed = {
        "include_test_targets": None,
    }

    def fake_loader(
        *,
        include_test_targets=False,
    ):
        observed[
            "include_test_targets"
        ] = include_test_targets

        return FakeDataset(
            X,
            y,
        )

    monkeypatch.setattr(
        phase4f,
        "validate_phase4f_preflight",
        lambda path: _fake_preflight(),
    )

    monkeypatch.setattr(
        phase4f,
        "run_phase4er_final_refit",
        lambda: _fake_final_refit(
            density_prediction,
            temperature_prediction,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "load_phase4_dataset",
        fake_loader,
    )

    monkeypatch.setattr(
        phase4f,
        "evaluate_phase4f_structural_diagnostics",
        lambda refit: _fake_structural(),
    )

    result = (
        phase4f.run_phase4f_locked_test_evaluation()
    )

    assert (
        observed[
            "include_test_targets"
        ]
        is True
    )

    assert result["phase"] == "4F"

    assert (
        result["stage"]
        == "one_time_locked_test_evaluation"
    )

    assert (
        result["irreversible_test_evaluation"]
        is True
    )

    assert (
        result["data_usage"][
            "final_fit_rows"
        ]
        == 6144
    )

    assert (
        result["data_usage"][
            "test_rows"
        ]
        == 2048
    )

    assert (
        result["data_usage"][
            "test_targets_accessed"
        ]
        is True
    )

    assert (
        result["data_usage"][
            "test_evaluation_performed"
        ]
        is True
    )

    assert (
        result["data_usage"][
            "retuning_after_test_permitted"
        ]
        is False
    )


def test_locked_evaluator_reports_complete_metrics_and_regions(
    monkeypatch,
):
    (
        X,
        y,
        density_prediction,
        temperature_prediction,
    ) = _synthetic_test_arrays()

    monkeypatch.setattr(
        phase4f,
        "validate_phase4f_preflight",
        lambda path: _fake_preflight(),
    )

    monkeypatch.setattr(
        phase4f,
        "run_phase4er_final_refit",
        lambda: _fake_final_refit(
            density_prediction,
            temperature_prediction,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "load_phase4_dataset",
        lambda **kwargs: FakeDataset(
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "evaluate_phase4f_structural_diagnostics",
        lambda refit: _fake_structural(),
    )

    result = (
        phase4f.run_phase4f_locked_test_evaluation()
    )

    assert set(
        result["density"]["metrics"]
    ) == {
        "mae",
        "rmse",
        "r2",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "p95_absolute_relative_error",
        "max_absolute_relative_error",
    }

    assert set(
        result["temperature"]["metrics"]
    ) == {
        "mae_eV",
        "rmse_eV",
        "r2",
        "mean_absolute_relative_error",
        "median_absolute_relative_error",
        "p95_absolute_error_eV",
        "max_absolute_error_eV",
    }

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
        result["density"]["error_space"]
    ) == expected_regions

    assert set(
        result["temperature"]["error_space"]
    ) == expected_regions

    assert len(
        result["density"][
            "worst_case_observations"
        ]
    ) == 10

    assert len(
        result["temperature"][
            "worst_case_observations"
        ]
    ) == 10


def test_locked_evaluator_reports_prediction_integrity(
    monkeypatch,
):
    (
        X,
        y,
        density_prediction,
        temperature_prediction,
    ) = _synthetic_test_arrays()

    monkeypatch.setattr(
        phase4f,
        "validate_phase4f_preflight",
        lambda path: _fake_preflight(),
    )

    monkeypatch.setattr(
        phase4f,
        "run_phase4er_final_refit",
        lambda: _fake_final_refit(
            density_prediction,
            temperature_prediction,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "load_phase4_dataset",
        lambda **kwargs: FakeDataset(
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "evaluate_phase4f_structural_diagnostics",
        lambda refit: _fake_structural(),
    )

    result = (
        phase4f.run_phase4f_locked_test_evaluation()
    )

    for target in (
        "density",
        "temperature",
    ):
        integrity = result[
            target
        ][
            "prediction_integrity"
        ]

        assert (
            integrity[
                "prediction_count"
            ]
            == 2048
        )

        assert (
            integrity[
                "all_predictions_finite"
            ]
            is True
        )

        assert (
            integrity[
                "all_predictions_strictly_positive"
            ]
            is True
        )


def test_locked_evaluator_preserves_no_retuning_interpretation(
    monkeypatch,
):
    (
        X,
        y,
        density_prediction,
        temperature_prediction,
    ) = _synthetic_test_arrays()

    monkeypatch.setattr(
        phase4f,
        "validate_phase4f_preflight",
        lambda path: _fake_preflight(),
    )

    monkeypatch.setattr(
        phase4f,
        "run_phase4er_final_refit",
        lambda: _fake_final_refit(
            density_prediction,
            temperature_prediction,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "load_phase4_dataset",
        lambda **kwargs: FakeDataset(
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        phase4f,
        "evaluate_phase4f_structural_diagnostics",
        lambda refit: _fake_structural(),
    )

    result = (
        phase4f.run_phase4f_locked_test_evaluation()
    )

    interpretation = result[
        "interpretation"
    ]

    assert (
        interpretation[
            "performance_threshold_introduced"
        ]
        is False
    )

    assert (
        interpretation[
            "test_is_final_generalisation_evidence"
        ]
        is True
    )

    assert (
        interpretation[
            "test_is_development_feedback"
        ]
        is False
    )

    assert (
        interpretation[
            "model_changes_permitted_from_test"
        ]
        is False
    )


def test_writer_refuses_existing_artifact_before_evaluation(
    tmp_path,
    monkeypatch,
):
    destination = (
        tmp_path
        / "locked_test_evaluation.json"
    )

    destination.write_text(
        "{}\n",
        encoding="utf-8",
    )

    called = {
        "evaluation": False,
    }

    def forbidden_evaluation(
        **kwargs,
    ):
        called["evaluation"] = True
        raise AssertionError(
            "Evaluation must not run when output exists."
        )

    monkeypatch.setattr(
        phase4f,
        "run_phase4f_locked_test_evaluation",
        forbidden_evaluation,
    )

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        phase4f.write_phase4f_locked_test_evaluation(
            destination
        )

    assert called["evaluation"] is False


def test_real_preflight_accepts_frozen_final_refit_without_test_access():
    payload = phase4f.validate_phase4f_preflight()

    assert payload["phase"] == "4E-R"

    assert (
        payload["stage"]
        == "final_train_validation_refit"
    )

    assert (
        payload["data_usage"][
            "final_refit_rows"
        ]
        == 6144
    )

    assert (
        payload["data_usage"][
            "test_targets_accessed"
        ]
        is False
    )

    assert (
        payload["data_usage"][
            "test_evaluation_performed"
        ]
        is False
    )
