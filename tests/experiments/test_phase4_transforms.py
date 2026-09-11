import numpy as np
import pytest

from plasma_ai.surrogate.transforms import (
    TargetTransform,
    allowed_transforms_for_target,
)


def test_density_transform_candidates_match_frozen_protocol():
    assert allowed_transforms_for_target(
        "true_electron_density_m3"
    ) == (
        "identity",
        "log10",
    )


def test_temperature_transform_candidates_match_frozen_protocol():
    assert allowed_transforms_for_target(
        "true_electron_temperature_eV"
    ) == (
        "identity",
    )


def test_identity_transform_round_trip():
    source = np.array(
        [1.0, 2.5, 10.0],
        dtype=np.float64,
    )

    transform = TargetTransform("identity")

    transformed = transform.forward(source)
    recovered = transform.inverse(transformed)

    np.testing.assert_allclose(
        transformed,
        source,
    )

    np.testing.assert_allclose(
        recovered,
        source,
    )


def test_log10_transform_round_trip():
    source = np.array(
        [1.0e15, 5.0e16, 1.0e17],
        dtype=np.float64,
    )

    transform = TargetTransform("log10")

    transformed = transform.forward(source)
    recovered = transform.inverse(transformed)

    np.testing.assert_allclose(
        recovered,
        source,
        rtol=1e-12,
        atol=0.0,
    )


def test_log10_transform_rejects_nonpositive_targets():
    transform = TargetTransform("log10")

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        transform.forward(
            np.array(
                [1.0, 0.0, -1.0],
                dtype=np.float64,
            )
        )


def test_transform_rejects_nonfinite_input():
    transform = TargetTransform("identity")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        transform.forward(
            np.array(
                [1.0, np.nan],
                dtype=np.float64,
            )
        )


def test_unknown_transform_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported target transform",
    ):
        TargetTransform("sqrt")


def test_unknown_target_is_rejected():
    with pytest.raises(
        KeyError,
        match="Unknown Phase 4 target",
    ):
        allowed_transforms_for_target(
            "unknown_target"
        )
