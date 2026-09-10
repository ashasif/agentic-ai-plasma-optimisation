import numpy as np
import pytest

from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_plan import (
    generate_monitoring_episode_plan,
)
from plasma_ai.experiments.monitoring_random import (
    deterministic_relative_factor,
)
from plasma_ai.experiments.monitoring_variability import (
    sample_process_variability,
)


CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)


def _config():
    return load_monitoring_config(
        CONFIG_PATH
    )


def _plan():
    config = _config()

    return generate_monitoring_episode_plan(
        config
    )


def test_process_variability_is_exactly_deterministic():
    config = _config()
    episode = _plan()[0]

    first = sample_process_variability(
        config,
        episode,
        17,
    )

    second = sample_process_variability(
        config,
        episode,
        17,
    )

    assert first == second


def test_process_variability_does_not_depend_on_generation_order():
    config = _config()
    episodes = _plan()[:4]

    keys = [
        (
            episode.episode_id,
            step,
        )
        for episode in episodes
        for step in range(
            config.steps_per_episode
        )
    ]

    lookup = {
        episode.episode_id: episode
        for episode in episodes
    }

    forward = {
        key: sample_process_variability(
            config,
            lookup[key[0]],
            key[1],
        )
        for key in keys
    }

    reverse = {
        key: sample_process_variability(
            config,
            lookup[key[0]],
            key[1],
        )
        for key in reversed(
            keys
        )
    }

    assert forward == reverse


def test_all_process_variability_factors_respect_three_sigma_bounds():
    config = _config()
    plan = _plan()

    process = (
        config.process_variability
    )

    power_limit = (
        process.absorbed_power_relative_sigma
        * process.truncation_sigma
    )

    flow_limit = (
        process.flow_relative_sigma
        * process.truncation_sigma
    )

    pumping_limit = (
        process.pumping_relative_sigma
        * process.truncation_sigma
    )

    for episode in plan:
        for step in range(
            config.steps_per_episode
        ):
            sample = sample_process_variability(
                config,
                episode,
                step,
            )

            assert abs(
                sample.power_relative_deviation
            ) <= (
                power_limit
                + 1e-15
            )

            assert abs(
                sample.flow_relative_deviation
            ) <= (
                flow_limit
                + 1e-15
            )

            assert abs(
                sample.pumping_relative_deviation
            ) <= (
                pumping_limit
                + 1e-15
            )

            assert (
                sample.power_coupling_factor
                > 0.0
            )

            assert (
                sample.flow_delivery_factor
                > 0.0
            )

            assert (
                sample.pumping_effectiveness_factor
                > 0.0
            )


def test_process_variability_has_reasonable_sample_statistics():
    config = _config()
    plan = _plan()

    power = []
    flow = []
    pumping = []

    for episode in plan:
        for step in range(
            config.steps_per_episode
        ):
            sample = sample_process_variability(
                config,
                episode,
                step,
            )

            power.append(
                sample.power_coupling_factor
            )

            flow.append(
                sample.flow_delivery_factor
            )

            pumping.append(
                sample.pumping_effectiveness_factor
            )

    power = np.asarray(
        power
    )

    flow = np.asarray(
        flow
    )

    pumping = np.asarray(
        pumping
    )

    assert abs(
        float(
            np.mean(power)
        )
        - 1.0
    ) < 0.0015

    assert abs(
        float(
            np.mean(flow)
        )
        - 1.0
    ) < 0.00075

    assert abs(
        float(
            np.mean(pumping)
        )
        - 1.0
    ) < 0.0015

    expected_sigmas = (
        (
            power,
            config.process_variability.absorbed_power_relative_sigma,
        ),
        (
            flow,
            config.process_variability.flow_relative_sigma,
        ),
        (
            pumping,
            config.process_variability.pumping_relative_sigma,
        ),
    )

    for values, sigma in expected_sigmas:
        observed = float(
            np.std(
                values,
                ddof=0,
            )
        )

        assert (
            0.85 * sigma
            < observed
            < 1.10 * sigma
        )


def test_process_variability_channels_are_not_identical():
    config = _config()
    episode = _plan()[0]

    samples = [
        sample_process_variability(
            config,
            episode,
            step,
        )
        for step in range(
            config.steps_per_episode
        )
    ]

    power = [
        sample.power_relative_deviation
        for sample in samples
    ]

    flow = [
        sample.flow_relative_deviation
        for sample in samples
    ]

    pumping = [
        sample.pumping_relative_deviation
        for sample in samples
    ]

    assert power != flow
    assert power != pumping
    assert flow != pumping


def test_relative_factor_handles_zero_sigma_without_random_effect():
    factor = deterministic_relative_factor(
        base_seed=123,
        episode_id="episode",
        step_index=0,
        channel="test",
        relative_sigma=0.0,
        truncation_sigma=3.0,
    )

    assert factor == 1.0


def test_process_variability_rejects_invalid_inputs():
    config = _config()
    episode = _plan()[0]

    with pytest.raises(
        ValueError
    ):
        sample_process_variability(
            config,
            episode,
            -1,
        )

    with pytest.raises(
        ValueError
    ):
        sample_process_variability(
            config,
            episode,
            64,
        )

    with pytest.raises(
        ValueError
    ):
        deterministic_relative_factor(
            base_seed=123,
            episode_id="episode",
            step_index=0,
            channel="test",
            relative_sigma=-0.01,
            truncation_sigma=3.0,
        )

    with pytest.raises(
        ValueError
    ):
        deterministic_relative_factor(
            base_seed=123,
            episode_id="episode",
            step_index=0,
            channel="test",
            relative_sigma=0.5,
            truncation_sigma=2.0,
        )
