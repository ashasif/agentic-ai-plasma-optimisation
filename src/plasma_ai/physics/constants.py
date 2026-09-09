"""Physical and model constants for the reduced-order argon plasma model."""

from scipy.constants import atomic_mass, elementary_charge, electron_mass


# Fundamental constants
ELEMENTARY_CHARGE_C = elementary_charge
ELECTRON_MASS_KG = electron_mass

# Argon
ARGON_ATOMIC_MASS_U = 39.948
ARGON_ION_MASS_KG = ARGON_ATOMIC_MASS_U * atomic_mass

# Effective ion-neutral collision cross-section used by Jan et al. (2013).
# This is a model parameter, not a universal constant for all Ar+/Ar collisions.
ARGON_ION_NEUTRAL_CROSS_SECTION_M2 = 1.0e-18

# Threshold energies used in the simplified reaction set.
ARGON_IONIZATION_ENERGY_EV = 15.76
ARGON_EXCITATION_ENERGY_EV = 11.5
