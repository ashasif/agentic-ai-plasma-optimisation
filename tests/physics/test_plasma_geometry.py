import pytest

from plasma_ai.physics.plasma_geometry import (
    axial_edge_factor,
    bohm_velocity_m_s,
    cylindrical_volume_m3,
    effective_loss_area_m2,
    ion_mean_free_path_m,
    radial_edge_factor,
    wall_loss_rate_s,
)


RADIUS_M = 0.17
LENGTH_M = 0.25
NEUTRAL_DENSITY_M3 = 1.0e21
TE_EV = 3.0


def test_cylindrical_volume_reference_value():
    assert cylindrical_volume_m3(RADIUS_M, LENGTH_M) == pytest.approx(
        0.02269800692218626,
        rel=1e-12,
    )


def test_bohm_velocity_reference_value():
    assert bohm_velocity_m_s(TE_EV) == pytest.approx(
        2691.8059861263746,
        rel=1e-10,
    )


def test_mean_free_path_reference_value():
    assert ion_mean_free_path_m(NEUTRAL_DENSITY_M3) == pytest.approx(
        1.0e-3,
        rel=1e-12,
    )


def test_edge_factors_reference_values():
    mean_free_path = ion_mean_free_path_m(NEUTRAL_DENSITY_M3)

    assert axial_edge_factor(
        LENGTH_M,
        mean_free_path,
    ) == pytest.approx(
        0.07601397897755385,
        rel=1e-10,
    )

    assert radial_edge_factor(
        RADIUS_M,
        mean_free_path,
    ) == pytest.approx(
        0.08479983040050879,
        rel=1e-10,
    )


def test_effective_loss_area_reference_value():
    mean_free_path = ion_mean_free_path_m(NEUTRAL_DENSITY_M3)
    h_l = axial_edge_factor(LENGTH_M, mean_free_path)
    h_r = radial_edge_factor(RADIUS_M, mean_free_path)

    assert effective_loss_area_m2(
        RADIUS_M,
        LENGTH_M,
        h_l,
        h_r,
    ) == pytest.approx(
        0.0364474811261349,
        rel=1e-10,
    )


def test_wall_loss_rate_reference_value():
    assert wall_loss_rate_s(
        TE_EV,
        NEUTRAL_DENSITY_M3,
        RADIUS_M,
        LENGTH_M,
    ) == pytest.approx(
        4322.386023182519,
        rel=1e-10,
    )


@pytest.mark.parametrize(
    "function,args",
    [
        (cylindrical_volume_m3, (0.0, 0.25)),
        (cylindrical_volume_m3, (0.17, -0.25)),
        (bohm_velocity_m_s, (0.0,)),
        (ion_mean_free_path_m, (0.0,)),
        (axial_edge_factor, (0.25, 0.0)),
        (radial_edge_factor, (0.17, 0.0)),
    ],
)
def test_invalid_inputs_are_rejected(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_bohm_velocity_increases_with_electron_temperature():
    assert bohm_velocity_m_s(5.0) > bohm_velocity_m_s(2.0)


def test_mean_free_path_decreases_with_neutral_density():
    low_density = ion_mean_free_path_m(1.0e20)
    high_density = ion_mean_free_path_m(1.0e21)

    assert high_density < low_density
