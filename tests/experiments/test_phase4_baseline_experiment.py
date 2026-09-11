from plasma_ai.surrogate.baseline_experiment import (
    run_reference_baselines,
)


def test_reference_baseline_experiment_uses_only_train_validation():
    payload = run_reference_baselines()

    usage = payload[
        "data_usage"
    ]

    assert usage["fit_split"] == "train"
    assert usage[
        "evaluation_split"
    ] == "validation"

    assert usage[
        "test_targets_accessed"
    ] is False

    assert usage[
        "train_rows"
    ] == 4096

    assert usage[
        "validation_rows"
    ] == 2048

    assert usage[
        "test_rows"
    ] == 2048


def test_reference_baseline_experiment_has_frozen_six_runs():
    payload = run_reference_baselines()

    results = payload[
        "results"
    ]

    assert len(results) == 6

    identities = {
        (
            row["target"],
            row["transform"],
            row["model"],
        )
        for row in results
    }

    assert identities == {
        (
            "true_electron_density_m3",
            "identity",
            "dummy_mean",
        ),
        (
            "true_electron_density_m3",
            "log10",
            "dummy_mean",
        ),
        (
            "true_electron_density_m3",
            "identity",
            "linear_regression",
        ),
        (
            "true_electron_density_m3",
            "log10",
            "linear_regression",
        ),
        (
            "true_electron_temperature_eV",
            "identity",
            "dummy_mean",
        ),
        (
            "true_electron_temperature_eV",
            "identity",
            "linear_regression",
        ),
    }


def test_reference_baseline_metrics_are_finite():
    payload = run_reference_baselines()

    for result in payload[
        "results"
    ]:
        for value in result[
            "metrics"
        ].values():
            assert float(
                value
            ) == float(
                value
            )
