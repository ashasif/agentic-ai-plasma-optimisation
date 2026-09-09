import pytest

from plasma_ai.physics.energy_losses import (
    collisional_energy_loss_per_pair_eV,
    collisional_power_loss_density_w_m3,
    electron_energy_density_j_m3,
    electron_temperature_eV_from_energy_density,
    electron_wall_energy_loss_eV,
    ion_wall_energy_loss_eV,
    sheath_potential_drop_v,
    wall_pair_energy_loss_eV,
    wall_power_loss_density_w_m3,
)


TE_EV = 3.0
ELECTRON_DENSITY_M3 = 1.0e17
NEUTRAL_DENSITY_M3 = 3.2188332786743194e20
WALL_LOSS_RATE_S = 4322.386023182519


def test_sheath_potential_reference_value():
    assert sheath_potential_drop_v(TE_EV) == pytest.approx(
        14.036818716504616,
        rel=1e-10,
    )


def test_electron_wall_energy_reference_value():
    assert electron_wall_energy_loss_eV(TE_EV) == pytest.approx(
        6.0,
        rel=1e-12,
    )


def test_ion_wall_energy_reference_value():
    assert ion_wall_energy_loss_eV(TE_EV) == pytest.approx(
        15.536818716504616,
        rel=1e-10,
    )


def test_total_wall_pair_energy_reference_value():
    assert wall_pair_energy_loss_eV(TE_EV) == pytest.approx(
        21.536818716504616,
        rel=1e-10,
    )


def test_collisional_energy_loss_reference_value():
    assert collisional_energy_loss_per_pair_eV(
        TE_EV
    ) == pytest.approx(
        21.234584011949597,
        rel=1e-10,
    )


def test_electron_energy_density_reference_value():
    assert electron_energy_density_j_m3(
        ELECTRON_DENSITY_M3,
        TE_EV,
    ) == pytest.approx(
        0.07209794853,
        rel=1e-10,
    )


def test_energy_density_temperature_round_trip():
    energy_density = electron_energy_density_j_m3(
        ELECTRON_DENSITY_M3,
        TE_EV,
    )

    recovered_te = electron_temperature_eV_from_energy_density(
        energy_density,
        ELECTRON_DENSITY_M3,
    )

    assert recovered_te == pytest.approx(
        TE_EV,
        rel=1e-12,
    )


def test_collisional_power_loss_density_reference_value():
    assert collisional_power_loss_density_w_m3(
        TE_EV,
        NEUTRAL_DENSITY_M3,
        ELECTRON_DENSITY_M3,
    ) == pytest.approx(
        27805.406266388014,
        rel=1e-10,
    )


def test_wall_power_loss_density_reference_value():
    assert wall_power_loss_density_w_m3(
        TE_EV,
        ELECTRON_DENSITY_M3,
        WALL_LOSS_RATE_S,
    ) == pytest.approx(
        1491.4733455238597,
        rel=1e-10,
    )


def test_zero_density_gives_zero_collisional_power_loss():
    assert collisional_power_loss_density_w_m3(
        TE_EV,
        0.0,
        ELECTRON_DENSITY_M3,
    ) == 0.0


def test_zero_electron_density_gives_zero_wall_power_loss():
    assert wall_power_loss_density_w_m3(
        TE_EV,
        0.0,
        WALL_LOSS_RATE_S,
    ) == 0.0


def test_sheath_potential_increases_with_temperature():
    assert sheath_potential_drop_v(5.0) > sheath_potential_drop_v(2.0)


def test_collisional_energy_loss_decreases_over_reference_range():
    low_te = collisional_energy_loss_per_pair_eV(2.0)
    high_te = collisional_energy_loss_per_pair_eV(5.0)

    assert high_te < low_te


@pytest.mark.parametrize(
    "function,args",
    [
        (sheath_potential_drop_v, (0.0,)),
        (electron_wall_energy_loss_eV, (-1.0,)),
        (ion_wall_energy_loss_eV, (0.0,)),
        (collisional_energy_loss_per_pair_eV, (0.0,)),
        (electron_energy_density_j_m3, (-1.0, 3.0)),
        (
            electron_temperature_eV_from_energy_density,
            (0.0, 1.0e17),
        ),
        (
            wall_power_loss_density_w_m3,
            (3.0, 1.0e17, -1.0),
        ),
    ],
)
def test_invalid_inputs_are_rejected(function, args):
    with pytest.raises(ValueError):
        function(*args)
