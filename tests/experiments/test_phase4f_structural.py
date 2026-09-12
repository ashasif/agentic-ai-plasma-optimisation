"""Tests for Phase 4F post-refit structural diagnostics.

These tests use the frozen Phase 4 probe grid and the frozen Phase 4E-R
final-refit path. They do not expose Phase 4 TEST targets.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_ai.surrogate.phase4er_final_refit import (
    run_phase4er_final_refit,
)
from plasma_ai.surrogate.phase4f_structural import (
    evaluate_phase4f_structural_diagnostics,
)
from plasma_ai.surrogate.physics_acceptance_experiment import (
    EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256,
)


def test_structural_diagnostics_require_exact_final_refit_rows():
    fake = SimpleNamespace(
        fit_rows=4096,
        test_targets_accessed=False,
    )

    with pytest.raises(
        ValueError,
        match="6,144-row final refit",
    ):
        evaluate_phase4f_structural_diagnostics(
            fake
        )


def test_structural_diagnostics_reject_test_access_flag():
    fake = SimpleNamespace(
        fit_rows=6144,
        test_targets_accessed=True,
    )

    with pytest.raises(
        ValueError,
        match="TEST-target access",
    ):
        evaluate_phase4f_structural_diagnostics(
            fake
        )


def test_real_final_refit_structural_diagnostics_preserve_test_lock():
    final_refit = run_phase4er_final_refit()

    assert final_refit.fit_rows == 6144
    assert final_refit.test_targets_accessed is False

    result = evaluate_phase4f_structural_diagnostics(
        final_refit
    )

    assert result["probe_grid"]["total_points"] == 1681
    assert result["probe_grid"]["absorbed_power_points"] == 41
    assert result["probe_grid"]["pressure_points"] == 41

    assert (
        result["probe_grid"][
            "source_reference_array_sha256"
        ]
        == EXPECTED_SOURCE_REFERENCE_ARRAY_SHA256
    )

    assert (
        result["interpretation"][
            "selection_gate"
        ]
        is False
    )

    assert (
        result["interpretation"][
            "post_refit_reporting_diagnostic"
        ]
        is True
    )

    assert (
        result["interpretation"][
            "test_targets_used"
        ]
        is False
    )

    assert (
        result["interpretation"][
            "retuning_permitted_from_result"
        ]
        is False
    )

    for target in (
        "density",
        "temperature",
    ):
        assert np.isfinite(
            result[target][
                "prediction_minimum"
            ]
        )

        assert np.isfinite(
            result[target][
                "prediction_maximum"
            ]
        )

        assert (
            result[target][
                "prediction_minimum"
            ]
            > 0.0
        )

        assert (
            result[target][
                "prediction_maximum"
            ]
            > 0.0
        )

        assert (
            result[target][
                "positivity"
            ][
                "passed"
            ]
            is True
        )


def test_real_final_refit_structural_payload_contains_frozen_checks():
    final_refit = run_phase4er_final_refit()

    result = evaluate_phase4f_structural_diagnostics(
        final_refit
    )

    assert set(
        result["density"]["trends"]
    ) == {
        "vs_power_at_fixed_pressure",
        "vs_pressure_at_fixed_power",
    }

    assert set(
        result["density"]["oscillation"]
    ) == {
        "vs_power_at_fixed_pressure",
        "vs_pressure_at_fixed_power",
    }

    assert set(
        result["temperature"]["trends"]
    ) == {
        "vs_pressure_at_fixed_power",
    }

    assert set(
        result["temperature"]["oscillation"]
    ) == {
        "vs_pressure_at_fixed_power",
    }

    assert isinstance(
        result["density"]["passed"],
        bool,
    )

    assert isinstance(
        result["temperature"]["passed"],
        bool,
    )

    assert isinstance(
        result["overall_passed"],
        bool,
    )
