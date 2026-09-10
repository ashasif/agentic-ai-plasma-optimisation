from dataclasses import replace

import pytest

from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_faults import (
    fault_progress,
    fault_started,
    multiplicative_fault_factor,
    signed_fault_fraction,
)
from plasma_ai.experiments.monitoring_plan import (
    generate_monitoring_episode_plan,
)


CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)


def _config():
    return load_monitoring_config(
        CONFIG_PATH
    )


def _plan():
    return generate_monitoring_episode_plan(
        _config()
    )


def _fault_example(
    profile="step",
    direction="negative",
    magnitude=0.20,
    onset=16,
):
    source = next(
        item
        for item in _plan()
        if item.fault_present
    )

    return replace(
        source,
        fault_present=True,
        fault_profile=profile,
        fault_direction=direction,
        fault_magnitude_fraction=magnitude,
        fault_onset_step=onset,
    )


def test_normal_episode_has_zero_fault_effect():
    normal = next(
        item
        for item in _plan()
        if not item.fault_present
    )

    for step in range(64):
        assert not fault_started(
            normal,
            step,
            64,
        )

        assert fault_progress(
            normal,
            step,
            64,
        ) == 0.0

        assert signed_fault_fraction(
            normal,
            step,
            64,
        ) == 0.0

        assert multiplicative_fault_factor(
            normal,
            step,
            64,
        ) == 1.0


def test_step_fault_jumps_to_full_effect_at_onset():
    plan = _fault_example(
        profile="step",
        direction="negative",
        magnitude=0.20,
        onset=16,
    )

    assert not fault_started(
        plan,
        15,
        64,
    )

    assert fault_progress(
        plan,
        15,
        64,
    ) == 0.0

    assert fault_started(
        plan,
        16,
        64,
    )

    assert fault_progress(
        plan,
        16,
        64,
    ) == 1.0

    assert fault_progress(
        plan,
        63,
        64,
    ) == 1.0


def test_drift_fault_uses_frozen_smoothstep_profile():
    plan = _fault_example(
        profile="drift",
        direction="positive",
        magnitude=0.20,
        onset=16,
    )

    assert not fault_started(
        plan,
        15,
        64,
    )

    assert fault_started(
        plan,
        16,
        64,
    )

    assert fault_progress(
        plan,
        15,
        64,
    ) == 0.0

    assert fault_progress(
        plan,
        16,
        64,
    ) == 0.0

    assert fault_progress(
        plan,
        63,
        64,
    ) == pytest.approx(
        1.0
    )

    step = 40

    u = (
        (step - 16)
        / (63 - 16)
    )

    expected = (
        3.0 * u**2
        - 2.0 * u**3
    )

    assert fault_progress(
        plan,
        step,
        64,
    ) == pytest.approx(
        expected
    )

    progress = [
        fault_progress(
            plan,
            step_index,
            64,
        )
        for step_index in range(64)
    ]

    assert all(
        0.0 <= value <= 1.0
        for value in progress
    )

    assert all(
        later >= earlier
        for earlier, later in zip(
            progress,
            progress[1:],
        )
    )


def test_signed_fault_fraction_respects_direction():
    negative = _fault_example(
        profile="step",
        direction="negative",
        magnitude=0.20,
        onset=16,
    )

    positive = replace(
        negative,
        fault_direction="positive",
    )

    assert signed_fault_fraction(
        negative,
        16,
        64,
    ) == pytest.approx(
        -0.20
    )

    assert signed_fault_fraction(
        positive,
        16,
        64,
    ) == pytest.approx(
        0.20
    )


def test_multiplicative_fault_factor_is_one_plus_effect():
    negative = _fault_example(
        profile="step",
        direction="negative",
        magnitude=0.20,
        onset=16,
    )

    positive = replace(
        negative,
        fault_direction="positive",
    )

    assert multiplicative_fault_factor(
        negative,
        15,
        64,
    ) == pytest.approx(
        1.0
    )

    assert multiplicative_fault_factor(
        negative,
        16,
        64,
    ) == pytest.approx(
        0.80
    )

    assert multiplicative_fault_factor(
        positive,
        16,
        64,
    ) == pytest.approx(
        1.20
    )


def test_all_frozen_episode_fault_factors_are_positive_and_bounded():
    plan = _plan()

    for episode in plan:
        for step in range(64):
            factor = multiplicative_fault_factor(
                episode,
                step,
                64,
            )

            assert (
                0.80
                <= factor
                <= 1.20
            )


def test_fault_functions_reject_invalid_step_indices():
    plan = _fault_example()

    with pytest.raises(
        ValueError
    ):
        fault_progress(
            plan,
            -1,
            64,
        )

    with pytest.raises(
        ValueError
    ):
        fault_progress(
            plan,
            64,
            64,
        )

    with pytest.raises(
        ValueError
    ):
        fault_started(
            plan,
            64,
            64,
        )
