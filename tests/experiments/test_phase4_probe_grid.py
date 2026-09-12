"""Tests for the frozen Phase 4E physics-validation probe grid."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plasma_ai.surrogate.probe_grid import (
    build_phase4e_probe_grid,
)


def test_phase4e_probe_grid_has_frozen_dimensions():
    grid = build_phase4e_probe_grid()

    assert grid.absorbed_power_W.shape == (41,)
    assert grid.target_pressure_mTorr.shape == (41,)
    assert grid.features.shape == (1681, 2)
    assert grid.total_points == 1681


def test_phase4e_probe_grid_includes_frozen_endpoints():
    grid = build_phase4e_probe_grid()

    assert grid.absorbed_power_W[0] == pytest.approx(
        15.0
    )
    assert grid.absorbed_power_W[-1] == pytest.approx(
        90.0
    )

    assert (
        grid.target_pressure_mTorr[0]
        == pytest.approx(10.0)
    )
    assert (
        grid.target_pressure_mTorr[-1]
        == pytest.approx(60.0)
    )


def test_phase4e_probe_grid_is_complete_cartesian_product():
    grid = build_phase4e_probe_grid()

    expected = {
        (float(power), float(pressure))
        for power in grid.absorbed_power_W
        for pressure in grid.target_pressure_mTorr
    }

    actual = {
        (float(row[0]), float(row[1]))
        for row in grid.features
    }

    assert len(expected) == 1681
    assert len(actual) == 1681
    assert actual == expected


def test_phase4e_probe_grid_is_deterministic():
    first = build_phase4e_probe_grid()
    second = build_phase4e_probe_grid()

    assert np.array_equal(
        first.absorbed_power_W,
        second.absorbed_power_W,
    )

    assert np.array_equal(
        first.target_pressure_mTorr,
        second.target_pressure_mTorr,
    )

    assert np.array_equal(
        first.features,
        second.features,
    )


def test_phase4e_probe_grid_uses_only_frozen_features():
    grid = build_phase4e_probe_grid()

    assert grid.features.shape[1] == 2

    assert np.array_equal(
        grid.features[:, 0],
        np.repeat(
            grid.absorbed_power_W,
            len(grid.target_pressure_mTorr),
        ),
    )

    assert np.array_equal(
        grid.features[:, 1],
        np.tile(
            grid.target_pressure_mTorr,
            len(grid.absorbed_power_W),
        ),
    )


def test_phase4e_probe_grid_rejects_phase3_envelope_drift(
    tmp_path,
):
    phase3_path = (
        tmp_path
        / "base_dataset.json"
    )

    phase3 = {
        "absorbed_power_min_W": 14.0,
        "absorbed_power_max_W": 90.0,
        "target_pressure_min_mTorr": 10.0,
        "target_pressure_max_mTorr": 60.0,
    }

    phase3_path.write_text(
        json.dumps(phase3),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Phase 4A qualified domain does not "
            "match the Phase 3 base-model envelope"
        ),
    ):
        build_phase4e_probe_grid(
            phase3_config_path=phase3_path,
        )
