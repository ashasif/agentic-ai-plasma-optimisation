from plasma_ai.experiments.monitoring_config import (
    FAULT_FAMILIES,
    load_monitoring_config,
)


CONFIG_PATH = (
    "configs/phase3/monitoring_dataset.json"
)


def _config():
    return load_monitoring_config(
        CONFIG_PATH
    )


def test_monitoring_dataset_has_frozen_size():
    config = _config()

    assert config.steps_per_episode == 64
    assert config.total_episodes == 64
    assert config.total_rows == 4096

    assert config.splits["train"].episodes == 32
    assert config.splits["validation"].episodes == 16
    assert config.splits["test"].episodes == 16


def test_monitoring_episode_allocations_match_frozen_design():
    config = _config()

    train = config.splits["train"]
    validation = config.splits["validation"]
    test = config.splits["test"]

    assert (
        train.normal,
        train.power_coupling,
        train.flow_delivery,
        train.pumping_effectiveness,
        train.pressure_sensor_bias,
    ) == (
        8,
        6,
        6,
        6,
        6,
    )

    assert (
        validation.normal,
        validation.power_coupling,
        validation.flow_delivery,
        validation.pumping_effectiveness,
        validation.pressure_sensor_bias,
    ) == (
        4,
        3,
        3,
        3,
        3,
    )

    assert (
        test.normal,
        test.power_coupling,
        test.flow_delivery,
        test.pumping_effectiveness,
        test.pressure_sensor_bias,
    ) == (
        4,
        3,
        3,
        3,
        3,
    )

    assert all(
        split.allocated_episodes
        == split.episodes
        for split in config.splits.values()
    )


def test_nominal_monitoring_recipe_uses_inner_envelope():
    recipe = _config().nominal_recipe

    assert recipe.absorbed_power_min_W == 25.0
    assert recipe.absorbed_power_max_W == 80.0

    assert recipe.target_pressure_min_mTorr == 20.0
    assert recipe.target_pressure_max_mTorr == 45.0

    assert recipe.flow_sccm == 20.0


def test_monitoring_variability_and_noise_are_frozen():
    config = _config()

    process = config.process_variability

    assert (
        process.absorbed_power_relative_sigma
        == 0.01
    )
    assert process.flow_relative_sigma == 0.005
    assert process.pumping_relative_sigma == 0.01
    assert process.truncation_sigma == 3.0

    noise = config.measurement_noise

    assert (
        noise.absorbed_power_relative_sigma
        == 0.01
    )
    assert noise.flow_relative_sigma == 0.005
    assert noise.pressure_relative_sigma == 0.005
    assert noise.truncation_sigma == 3.0


def test_monitoring_faults_match_frozen_families_and_onset():
    config = _config()

    assert set(config.faults) == set(
        FAULT_FAMILIES
    )

    assert config.fault_onset.minimum_step == 16
    assert config.fault_onset.maximum_step == 31

    assert config.fault_profiles == (
        "step",
        "drift",
    )

    assert (
        config.faults[
            "power_coupling"
        ].directions
        == ("negative",)
    )

    assert (
        config.faults[
            "pressure_sensor_bias"
        ].domain
        == "sensor"
    )

    assert (
        config.faults[
            "pumping_effectiveness"
        ].severe_fraction
        == 0.20
    )


def test_monitoring_reproducibility_seeds_are_frozen():
    seeds = _config().seeds

    assert seeds.recipe_design == 20260920
    assert seeds.episode_fault_plan == 20260921
    assert seeds.process_variability == 20260922
    assert seeds.measurement_noise == 20260923

    assert len(
        {
            seeds.recipe_design,
            seeds.episode_fault_plan,
            seeds.process_variability,
            seeds.measurement_noise,
        }
    ) == 4
