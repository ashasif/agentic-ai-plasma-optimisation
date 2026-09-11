import numpy as np

from plasma_ai.surrogate.data import (
    load_phase4_dataset,
    load_surrogate_protocol,
)


EXPECTED_FEATURES = (
    "nominal_absorbed_power_W",
    "target_pressure_mTorr",
)

EXPECTED_TARGETS = (
    "true_electron_density_m3",
    "true_electron_temperature_eV",
)


def _dataset():
    return load_phase4_dataset()


def test_phase4_loader_preserves_frozen_feature_and_target_contract():
    dataset = _dataset()

    assert dataset.feature_names == EXPECTED_FEATURES
    assert dataset.target_names == EXPECTED_TARGETS


def test_phase4_loader_preserves_frozen_split_shapes():
    dataset = _dataset()

    train = dataset.split("train")
    validation = dataset.split("validation")
    test = dataset.split("test")

    assert train.X.shape == (4096, 2)
    assert train.y is not None
    assert train.y.shape == (4096, 2)

    assert validation.X.shape == (2048, 2)
    assert validation.y is not None
    assert validation.y.shape == (2048, 2)

    assert test.X.shape == (2048, 2)


def test_phase4_loader_withholds_test_targets_by_default():
    dataset = _dataset()

    assert dataset.split("test").y is None


def test_phase4_loader_returns_only_finite_development_values():
    dataset = _dataset()

    train = dataset.split("train")
    validation = dataset.split("validation")
    test = dataset.split("test")

    assert np.isfinite(train.X).all()
    assert np.isfinite(train.y).all()

    assert np.isfinite(validation.X).all()
    assert np.isfinite(validation.y).all()

    assert np.isfinite(test.X).all()


def test_phase4_loader_respects_frozen_input_domain():
    dataset = _dataset()

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        X = dataset.split(split_name).X

        power = X[:, 0]
        pressure = X[:, 1]

        assert np.all(power >= 15.0)
        assert np.all(power <= 90.0)

        assert np.all(pressure >= 10.0)
        assert np.all(pressure <= 60.0)


def test_phase4_development_targets_are_strictly_positive():
    dataset = _dataset()

    for split_name in (
        "train",
        "validation",
    ):
        y = dataset.split(split_name).y

        assert y is not None
        assert np.all(y[:, 0] > 0.0)
        assert np.all(y[:, 1] > 0.0)


def test_phase4_protocol_remains_frozen_and_test_locked():
    protocol = load_surrogate_protocol()

    assert protocol["status"] == "frozen"

    policy = protocol["data_use_policy"]

    assert policy[
        "test_locked_during_development"
    ] is True

    assert policy[
        "test_may_affect_model_selection"
    ] is False

    assert policy[
        "retuning_after_test"
    ] is False
