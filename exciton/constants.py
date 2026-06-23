"""
constants.py
Port of constant.fi from Exciton code (Bittner Group, UH)

Units throughout: TIME [ps], ENERGY [eV], DISTANCE [Angstrom], CHARGE [e]
"""

# Fundamental constants
ELECTRON_CHARGE   = 1.602188e-19   # C
AVOGADRO          = 6.0221367e+23
ATOMIC_MASS_UNIT  = 1.660540e-27   # kg
ONE_CALORIE       = 4.184          # J
C_MASS            = 12.0           # amu
ONE_ANGSTROM      = 1.0e-10        # m
H_PLANCK          = 6.6260755e-34  # J·s

# Working units: eV, ps, Angstrom
HBAR   = 0.000658   # eV·ps  (ħ = h/2π)
HBAR_FS= 0.658211951  # eV·fs  (used in QuDPy)
KB     = 8.617387467e-5  # eV/K  (Boltzmann constant)

EPS_ZERO = 0.0055238   # vacuum permittivity in working units
C_LIGHT  = 2.9979e6    # speed of light [Angstrom/ps]

VERY_SMALL = 1.0e-8
