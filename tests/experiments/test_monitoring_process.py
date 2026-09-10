from dataclasses import replace

import pytest

from plasma_ai.experiments.monitoring_config import (
    load_monitoring_config,
)
from plasma_ai.experiments.monitoring_plan import (
    generate_monitoring_episode_plan,
)
from plasma_ai.experiments.monitoring_process import (
    build_true_process_inputs,
    simulate_monitoring_process_step,
)


CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)


def _config():
    return load_monitoring_config(
        CONFIG_PATH
    )


def _base_plan():
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


def _fault(
    plan,
    *,
    family,
    domain="process",
    direction="positive",
    magnitude=0.20,
):
    return replace(
        plan,
        fault_present=True,
        fault_domain=domain,
        fault_family=family,
        fault_profile="step",
        fault_severity="severe",
        fault_direction=direction,
        fault_magnitude_fraction=magnitude,
        fault_onset_step=16,
    )


def test_process_inputs_apply_fault_only_to_intended_channel():
    config = _config()
    base = _base_plan()
    step = 20

    normal = build_true_process_inputs(
        config,
        base,
        step,
    )

    power = build_true_process_inputs(
        config,
        _fault(
            base,
            family="power_coupling",
            direction="negative",
            magnitude=0.20,
        ),
        step,
    )

    flow = build_true_process_inputs(
        config,
        _fault(
            base,
            family="flow_delivery",
            direction="positive",
            magnitude=0.15,
        ),
        step,
    )

    pumping = build_true_process_inputs(
        config,
        _fault(
            base,
            family="pumping_effectiveness",
            direction="positive",
            magnitude=0.20,
        ),
        step,
    )

    sensor = build_true_process_inputs(
        config,
        _fault(
            base,
            family="pressure_sensor_bias",
            domain="sensor",
            direction="positive",
            magnitude=0.10,
        ),
        step,
    )

    assert (
        power.true_absorbed_power_W
        == pytest.approx(
            normal.true_absorbed_power_W
            * 0.80
        )
    )

    assert (
        power.true_flow_sccm
        == pytest.approx(
            normal.true_flow_sccm
        )
    )

    assert (
        power.true_pumping_speed_m3_s
        == pytest.approx(
            normal.true_pumping_speed_m3_s
        )
    )

    assert (
        flow.true_flow_sccm
        == pytest.approx(
            normal.true_flow_sccm
            * 1.15
        )
    )

    assert (
        flow.true_pumping_speed_m3_s
        == pytest.approx(
            normal.true_pumping_speed_m3_s
        )
    )

    assert (
        pumping.true_pumping_speed_m3_s
        == pytest.approx(
            normal.true_pumping_speed_m3_s
            * 1.20
        )
    )

    assert (
        pumping.true_flow_sccm
        == pytest.approx(
            normal.true_flow_sccm
        )
    )

    # Sensor bias must not alter latent process inputs.
    assert (
        sensor.true_absorbed_power_W
        == pytest.approx(
            normal.true_absorbed_power_W
        )
    )

    assert (
        sensor.true_flow_sccm
        == pytest.approx(
            normal.true_flow_sccm
        )
    )

    assert (
        sensor.true_pumping_speed_m3_s
        == pytest.approx(
            normal.true_pumping_speed_m3_s
        )
    )


def test_flow_fault_does_not_rederive_pumping_speed():
    config = _config()
    base = _base_plan()

    normal = build_true_process_inputs(
        config,
        base,
        20,
    )

    flow_fault = build_true_process_inputs(
        config,
        _fault(
            base,
            family="flow_delivery",
            direction="positive",
            magnitude=0.15,
        ),
        20,
    )

    assert (
        flow_fault.true_flow_sccm
        > normal.true_flow_sccm
    )

    assert (
        flow_fault.true_pumping_speed_m3_s
        == pytest.approx(
            normal.true_pumping_speed_m3_s
        )
    )


def test_quasi_steady_process_faults_have_expected_physical_direction():
    config = _config()
    base = _base_plan()
    step = 20

    normal = simulate_monitoring_process_step(
        config,
        base,
        step,
    )

    flow_high = simulate_monitoring_process_step(
        config,
        _fault(
            base,
            family="flow_delivery",
            direction="positive",
            magnitude=0.15,
        ),
        step,
    )

    pumping_high = simulate_monitoring_process_step(
        config,
        _fault(
            base,
            family="pumping_effectiveness",
            direction="positive",
            magnitude=0.20,
        ),
        step,
    )

    power_low = simulate_monitoring_process_step(
        config,
        _fault(
            base,
            family="power_coupling",
            direction="negative",
            magnitude=0.20,
        ),
        step,
    )

    for state in (
        normal,
        flow_high,
        pumping_high,
        power_low,
    ):
        assert state.integration_success
        assert state.converged
        assert state.physical_state_valid
        assert state.balance_valid
        assert state.qualification_valid
        assert (
            state.model_validity_status
            == "valid"
        )

    # More delivered gas at unchanged pumping raises pressure.
    assert (
        flow_high.true_pressure_mTorr
        > normal.true_pressure_mTorr
    )

    # Greater pumping at unchanged delivered flow lowers pressure.
    assert (
        pumping_high.true_pressure_mTorr
        < normal.true_pressure_mTorr
    )

    # Reduced absorbed-power coupling lowers electron density.
    assert (
        power_low.true_electron_density_m3
        < normal.true_electron_density_m3
    )


def test_monitoring_process_step_is_exactly_deterministic():
    config = _config()
    plan = _base_plan()

    first = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    second = simulate_monitoring_process_step(
        config,
        plan,
        20,
    )

    assert first == second


def test_monitoring_process_rejects_invalid_step():
    config = _config()
    plan = _base_plan()

    with pytest.raises(
        ValueError
    ):
        build_true_process_inputs(
            config,
            plan,
            -1,
        )

    with pytest.raises(
        ValueError
    ):
        simulate_monitoring_process_step(
            config,
            plan,
            64,
        )
