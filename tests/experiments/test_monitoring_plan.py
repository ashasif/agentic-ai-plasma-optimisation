from collections import Counter, defaultdict

from plasma_ai.experiments.monitoring_config import (
    FAULT_FAMILIES,
    load_monitoring_config,
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


def test_monitoring_plan_has_64_unique_episodes_and_recipes():
    plan = _plan()

    assert len(plan) == 64

    assert len(
        {
            item.episode_id
            for item in plan
        }
    ) == 64

    assert len(
        {
            (
                item.nominal_absorbed_power_W,
                item.target_pressure_mTorr,
            )
            for item in plan
        }
    ) == 64


def test_monitoring_plan_split_counts_are_episode_safe():
    plan = _plan()

    counts = Counter(
        item.split
        for item in plan
    )

    assert counts == {
        "train": 32,
        "validation": 16,
        "test": 16,
    }

    episode_splits = defaultdict(
        set
    )

    for item in plan:
        episode_splits[
            item.episode_id
        ].add(
            item.split
        )

    assert all(
        len(splits) == 1
        for splits in episode_splits.values()
    )


def test_monitoring_plan_respects_inner_recipe_envelope():
    config = _config()
    plan = _plan()

    recipe = config.nominal_recipe

    for item in plan:
        assert (
            recipe.absorbed_power_min_W
            <= item.nominal_absorbed_power_W
            <= recipe.absorbed_power_max_W
        )

        assert (
            recipe.target_pressure_min_mTorr
            <= item.target_pressure_mTorr
            <= recipe.target_pressure_max_mTorr
        )

        assert (
            item.nominal_flow_sccm
            == recipe.flow_sccm
        )


def test_monitoring_fault_family_counts_match_frozen_design():
    plan = _plan()

    counts = Counter(
        item.fault_family
        for item in plan
    )

    assert counts == {
        "none": 16,
        "power_coupling": 12,
        "flow_delivery": 12,
        "pumping_effectiveness": 12,
        "pressure_sensor_bias": 12,
    }


def test_each_fault_family_balances_severity_and_profile():
    plan = _plan()

    for family in FAULT_FAMILIES:
        episodes = [
            item
            for item in plan
            if item.fault_family == family
        ]

        severity_counts = Counter(
            item.fault_severity
            for item in episodes
        )

        profile_counts = Counter(
            item.fault_profile
            for item in episodes
        )

        assert severity_counts == {
            "mild": 4,
            "moderate": 4,
            "severe": 4,
        }

        assert profile_counts == {
            "step": 6,
            "drift": 6,
        }


def test_fault_directions_and_onsets_are_consistent():
    config = _config()
    plan = _plan()

    power = [
        item
        for item in plan
        if item.fault_family
        == "power_coupling"
    ]

    assert {
        item.fault_direction
        for item in power
    } == {
        "negative",
    }

    for family in (
        "flow_delivery",
        "pumping_effectiveness",
        "pressure_sensor_bias",
    ):
        episodes = [
            item
            for item in plan
            if item.fault_family == family
        ]

        direction_counts = Counter(
            item.fault_direction
            for item in episodes
        )

        assert direction_counts == {
            "negative": 6,
            "positive": 6,
        }

    faulty = [
        item
        for item in plan
        if item.fault_present
    ]

    assert len(
        faulty
    ) == 48

    for item in faulty:
        assert item.fault_onset_step is not None

        assert (
            config.fault_onset.minimum_step
            <= item.fault_onset_step
            <= config.fault_onset.maximum_step
        )

        expected_magnitude = (
            config.faults[
                item.fault_family
            ].magnitude_for_severity(
                item.fault_severity
            )
        )

        assert (
            item.fault_magnitude_fraction
            == expected_magnitude
        )

    normal = [
        item
        for item in plan
        if not item.fault_present
    ]

    assert len(
        normal
    ) == 16

    for item in normal:
        assert item.fault_domain == "none"
        assert item.fault_family == "none"
        assert item.fault_profile == "none"
        assert item.fault_severity == "none"
        assert item.fault_direction == "none"
        assert item.fault_magnitude_fraction == 0.0
        assert item.fault_onset_step is None


def test_monitoring_episode_plan_is_exactly_deterministic():
    config = _config()

    first = generate_monitoring_episode_plan(
        config
    )

    second = generate_monitoring_episode_plan(
        config
    )

    assert first == second
