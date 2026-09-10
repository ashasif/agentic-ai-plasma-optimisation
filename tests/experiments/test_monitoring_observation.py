from dataclasses import replace

import pytest

from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_observation import (
    build_monitoring_observation,
)
from plasma_ai.experiments.monitoring_plan import (
    generate_monitoring_episode_plan,
)
from plasma_ai.experiments.monitoring_process import (
    simulate_monitoring_process_step,
)


CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)


def _config():
    return load_monitoring_config(
        CONFIG_PATH
    )


def _normal_plan():
    config = _config()

    source = generate_monitoring_episode_plan(
        config
    )[0]

    return replace(
        source,
        nominal_absorbed_power_W=50.0,
        target_pressure_mTorr=30.0,
        nominal_flow_sccm=20.0,
        fault_present=False,
        fault_domain="none",
        fault_family="none",
        fault_profile="none",
        fault_severity="none",
        fault_direction="none",
        fault_magnitude_fraction=0.0,
        fault_onset_step=None,
    )


def _sensor_fault_plan(
    direction="positive",
    magnitude=0.10,
):
    base = _normal_plan()

    return replace(
        base,
        fault_present=True,
        fault_domain="sensor",
        fault_family="pressure_sensor_bias",
        fault_profile="step",
        fault_severity="severe",
        fault_direction=direction,
        fault_magnitude_fraction=magnitude,
        fault_onset_step=16,
    )


def test_normal_observation_contains_measurement_noise_only():
    config = _config()
    plan = _normal_plan()

    state = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    observation = build_monitoring_observation(
        config,
        plan,
        state,
    )

    assert (
        observation.pressure_sensor_bias_factor
        == 1.0
    )

    assert (
        observation.pressure_sensor_bias_fraction
        == 0.0
    )

    assert observation.measured_absorbed_power_W > 0.0
    assert observation.measured_flow_sccm > 0.0
    assert observation.measured_pressure_mTorr > 0.0


def test_pressure_sensor_bias_does_not_change_true_process_state():
    config = _config()

    normal = _normal_plan()
    sensor = _sensor_fault_plan(
        direction="positive",
        magnitude=0.10,
    )

    normal_state = simulate_monitoring_process_step(
        config,
        normal,
        20,
    )

    sensor_state = simulate_monitoring_process_step(
        config,
        sensor,
        20,
    )

    assert (
        sensor_state.true_absorbed_power_W
        == pytest.approx(
            normal_state.true_absorbed_power_W
        )
    )

    assert (
        sensor_state.true_flow_sccm
        == pytest.approx(
            normal_state.true_flow_sccm
        )
    )

    assert (
        sensor_state.true_pumping_speed_m3_s
        == pytest.approx(
            normal_state.true_pumping_speed_m3_s
        )
    )

    assert (
        sensor_state.true_pressure_mTorr
        == pytest.approx(
            normal_state.true_pressure_mTorr
        )
    )

    assert (
        sensor_state.true_electron_density_m3
        == pytest.approx(
            normal_state.true_electron_density_m3
        )
    )

    assert (
        sensor_state.true_electron_temperature_eV
        == pytest.approx(
            normal_state.true_electron_temperature_eV
        )
    )


def test_positive_pressure_sensor_bias_is_applied_after_true_process():
    config = _config()

    plan = _sensor_fault_plan(
        direction="positive",
        magnitude=0.10,
    )

    state = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    observation = build_monitoring_observation(
        config,
        plan,
        state,
    )

    assert (
        observation.pressure_sensor_bias_factor
        == pytest.approx(
            1.10
        )
    )

    expected = (
        state.true_pressure_mTorr
        * 1.10
        * observation.pressure_measurement_noise_factor
    )

    assert (
        observation.measured_pressure_mTorr
        == pytest.approx(
            expected
        )
    )


def test_sensor_bias_is_zero_before_fault_onset():
    config = _config()

    plan = _sensor_fault_plan(
        direction="positive",
        magnitude=0.10,
    )

    state = simulate_monitoring_process_step(
        config,
        plan,
        15,
    )

    observation = build_monitoring_observation(
        config,
        plan,
        state,
    )

    assert (
        observation.pressure_sensor_bias_factor
        == 1.0
    )

    assert (
        observation.pressure_sensor_bias_fraction
        == 0.0
    )


def test_measurement_noise_respects_frozen_three_sigma_bounds():
    config = _config()
    plan = _normal_plan()

    limits = {
        "power": (
            config.measurement_noise.absorbed_power_relative_sigma
            * config.measurement_noise.truncation_sigma
        ),
        "flow": (
            config.measurement_noise.flow_relative_sigma
            * config.measurement_noise.truncation_sigma
        ),
        "pressure": (
            config.measurement_noise.pressure_relative_sigma
            * config.measurement_noise.truncation_sigma
        ),
    }

    for step in range(
        config.steps_per_episode
    ):
        state = simulate_monitoring_process_step(
            config,
            plan,
            step,
        )

        observation = build_monitoring_observation(
            config,
            plan,
            state,
        )

        assert abs(
            observation.power_measurement_relative_noise
        ) <= (
            limits["power"]
            + 1e-15
        )

        assert abs(
            observation.flow_measurement_relative_noise
        ) <= (
            limits["flow"]
            + 1e-15
        )

        assert abs(
            observation.pressure_measurement_relative_noise
        ) <= (
            limits["pressure"]
            + 1e-15
        )


def test_monitoring_observation_is_exactly_deterministic():
    config = _config()
    plan = _normal_plan()

    state = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    first = build_monitoring_observation(
        config,
        plan,
        state,
    )

    second = build_monitoring_observation(
        config,
        plan,
        state,
    )

    assert first == second


def test_observation_rejects_mismatched_episode():
    config = _config()
    plan = _normal_plan()

    state = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    bad_plan = replace(
        plan,
        episode_id="different_episode",
    )

    with pytest.raises(
        ValueError
    ):
        build_monitoring_observation(
            config,
            bad_plan,
            state,
        )
