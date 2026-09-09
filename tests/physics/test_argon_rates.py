import numpy as np
import pytest

from plasma_ai.physics.argon_rates import (
    elastic_rate_m3_s,
    excitation_rate_m3_s,
    ionization_rate_m3_s,
)


def test_rates_are_positive():
    te = np.array([1.0, 2.0, 3.0, 5.0, 7.0])

    assert np.all(ionization_rate_m3_s(te) > 0.0)
    assert np.all(excitation_rate_m3_s(te) > 0.0)
    assert np.all(elastic_rate_m3_s(te) > 0.0)


def test_ionization_rate_increases_over_reference_range():
    te = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    rates = ionization_rate_m3_s(te)

    assert np.all(np.diff(rates) > 0.0)


def test_excitation_rate_increases_over_reference_range():
    te = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    rates = excitation_rate_m3_s(te)

    assert np.all(np.diff(rates) > 0.0)


def test_scalar_reference_values_at_3_eV():
    assert ionization_rate_m3_s(3.0) == pytest.approx(
        2.539081323343745e-16,
        rel=1e-10,
    )

    assert excitation_rate_m3_s(3.0) == pytest.approx(
        1.1954415142627366e-16,
        rel=1e-10,
    )

    assert elastic_rate_m3_s(3.0) == pytest.approx(
        1.236630666806178e-13,
        rel=1e-10,
    )


@pytest.mark.parametrize("bad_temperature", [0.0, -1.0, np.nan, np.inf])
def test_invalid_temperature_is_rejected(bad_temperature):
    with pytest.raises(ValueError):
        ionization_rate_m3_s(bad_temperature)


def test_vector_input_preserves_shape():
    te = np.array([[2.0, 3.0], [4.0, 5.0]])

    result = ionization_rate_m3_s(te)

    assert result.shape == te.shape
