"""
parameters.py
Port of parameters.fi / input.f namelist blocks from Exciton code.

All parameters use working units: eV, ps, Angstrom.
Default values are from Table 1 of:
  Karabunarliev & Bittner, J. Chem. Phys. 118, 4291 (2003)  [PPV model]
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ExcitonParams:
    """
    Complete parameter set for the Exciton Hamiltonian.

    Electronic structure
    --------------------
    nsites      : number of monomer sites
    ispinstate  : 0 = singlet, 1 = triplet, -1 = both
    egap        : half the energy gap between band centres (eV)
    ewidth      : inter-site transfer integral (eV)  [bandwidth = 4*ewidth]

    Phonon model (two branches)
    ---------------------------
    forcek1/2   : on-site spring constants (eV/Å²)
    forceo1/2   : off-site spring constants (eV/Å²)
    forcem      : reduced mass parameter (dimensionless; controls e/h band asymmetry)

    Electron–phonon coupling (Huang–Rhys parameters)
    -------------------------------------------------
    gediag1/2   : dimensionless coupling strengths for two phonon branches
                  fediag is derived from these in setup()

    Electron–electron interactions
    ------------------------------
    coulombe    : Coulomb amplitude J₀ (eV)
    coulombr    : Coulomb screening length r₀ (Å)
    dipole      : dipole–dipole amplitude D₀ (eV·Å³)
    dipolr      : dipole–dipole length scale r₀ (Å)
    potexe      : exchange amplitude K₀ (eV)
    potexr      : exchange decay length r₀ (Å)

    Bath / lineshape
    ----------------
    ekbt        : thermal energy k_B T (eV)  [0.025 eV ≈ 300 K]
    phgama      : phonon linewidth (eV/ps → converted to rad/ps internally)
    elgama      : electronic linewidth / dephasing (eV)
    knonrad     : non-radiative decay rate (1/ps)

    Disorder
    --------
    disorder    : apply off-diagonal (hopping) disorder
    sigma_beta  : std dev of hopping disorder (eV)

    Time evolution
    --------------
    timall      : total evolution time (ps)
    timstep     : time step (ps)
    timshot     : output interval (ps)

    Flags
    -----
    relax       : compute adiabatic (lattice-relaxed) states
    evolve      : integrate Redfield master equation
    exciton     : start from CI eigenstate iex
    chargesep   : start from charge-separated state
    absorb      : compute absorption spectrum
    pdensity    : output population density
    radrates    : compute radiative rates
    debug       : verbose output
    iex         : initial CI eigenstate index (1-based)
    maxrelax    : max number of states to relax
    maxits      : max SCF iterations for relaxation
    """

    # System size
    nsites:     int   = 4
    ispinstate: int   = 0       # 0=singlet, 1=triplet, -1=both

    # Electronic model (PPV defaults from JCP 118, 4291)
    egap:    float = 2.75       # eV
    ewidth:  float = 0.536      # eV
    forcem:  float = 1.1        # dimensionless

    # Phonon branches
    forcek1: float = 0.200      # eV/Å²
    forcek2: float = 0.0125     # eV/Å²
    forceo1: float = 0.020      # eV/Å²
    forceo2: float = 0.0013     # eV/Å²

    # Huang–Rhys / el–ph couplings
    gediag1: float = 0.60
    gediag2: float = 4.0

    # Coulomb interactions
    coulombe: float =  3.0921   # eV
    coulombr: float =  0.6840   # Å
    dipole:   float = -0.0321   # eV·Å³
    dipolr:   float =  1.0      # Å
    potexe:   float =  1.0574   # eV
    potexr:   float =  0.4743   # Å

    # Bath
    ekbt:    float = 0.025      # eV  (≈ 300 K)
    phgama:  float = 0.010      # eV/ps
    elgama:  float = 0.030      # eV
    knonrad: float = 0.0        # 1/ps

    # Disorder
    disorder:    bool  = False
    sigma_beta:  float = 0.0    # eV

    # Time evolution
    timall:   float = 100.0     # ps
    timstep:  float = 0.1       # ps
    timshot:  float = 10.0      # ps

    # Flags
    relax:     bool = False
    evolve:    bool = True
    exciton:   bool = True
    chargesep: bool = False
    absorb:    bool = True
    pdensity:  bool = False
    radrates:  bool = False
    debug:     bool = False

    # Initial state / output control
    iex:      int = 1
    maxrelax: int = 16
    maxits:   int = 100

    def __post_init__(self):
        """Derive fediag coupling constants from Huang-Rhys parameters."""
        from .constants import HBAR
        self._derive_fediag()

    def _derive_fediag(self):
        """
        Derived electron-phonon coupling constants (fediag) from
        Huang-Rhys parameters (gediag) and force constants.
        From input.f:
            fediag1 = sqrt(0.5 * forcek1^3 * gediag1 / hbar^2)
        """
        from .constants import HBAR
        self.fediag1 = np.sqrt(0.5 * self.forcek1**3 * self.gediag1 / HBAR**2)
        if self.gediag2 > 0.0:
            self.fediag2 = np.sqrt(0.5 * self.forcek2**3 * self.gediag2 / HBAR**2)
        else:
            self.fediag2 = 0.0

    @classmethod
    def from_input_file(cls, filepath: str) -> 'ExcitonParams':
        """
        Parse a SAMPLE.input style file and return an ExcitonParams instance.
        Reads site energies and hopping integrals separately via parse_geometry().
        """
        import re

        params = cls()

        # Map namelist keys to dataclass fields
        key_map = {
            'nsites': ('nsites', int),
            'ispinstate': ('ispinstate', int),
            'maxrelax': ('maxrelax', int),
            'maxits': ('maxits', int),
            'iex': ('iex', int),
            'disorder': ('disorder', lambda x: x.strip().upper() == 'T'),
            'relax': ('relax', lambda x: x.strip().upper() == 'T'),
            'evolve': ('evolve', lambda x: x.strip().upper() == 'T'),
            'exciton': ('exciton', lambda x: x.strip().upper() == 'T'),
            'chargesep': ('chargesep', lambda x: x.strip().upper() == 'T'),
            'absorb': ('absorb', lambda x: x.strip().upper() == 'T'),
            'pdensity': ('pdensity', lambda x: x.strip().upper() == 'T'),
            'radrates': ('radrates', lambda x: x.strip().upper() == 'T'),
            'debug': ('debug', lambda x: x.strip().upper() == 'T'),
            'sigma_beta': ('sigma_beta', float),
            'coulombe': ('coulombe', float),
            'coulombr': ('coulombr', float),
            'dipole': ('dipole', float),
            'dipolr': ('dipolr', float),
            'potexe': ('potexe', float),
            'potexr': ('potexr', float),
            'ekbt': ('ekbt', float),
            'phgama': ('phgama', float),
            'elgama': ('elgama', float),
            'knonrad': ('knonrad', float),
            'timall': ('timall', float),
            'timstep': ('timstep', float),
            'timshot': ('timshot', float),
            'forcek1': ('forcek1', float),
            'forcek2': ('forcek2', float),
            'forceo1': ('forceo1', float),
            'forceo2': ('forceo2', float),
            'gediag1': ('gediag1', float),
            'gediag2': ('gediag2', float),
            'forcem': ('forcem', float),
        }

        with open(filepath) as f:
            text = f.read()

        # Strip Fortran namelist block markers
        text = re.sub(r'&\w+', '', text)
        text = re.sub(r'/', '', text)

        for line in text.splitlines():
            line = line.strip().rstrip(',')
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            if '=' in line:
                key, _, val = line.partition('=')
                key = key.strip().lower()
                val = val.strip().rstrip(',')
                if key in key_map:
                    attr, conv = key_map[key]
                    try:
                        setattr(params, attr, conv(val))
                    except (ValueError, TypeError):
                        pass

        params._derive_fediag()
        return params

    def summary(self) -> str:
        lines = [
            "ExcitonParams",
            f"  nsites={self.nsites}  ispinstate={self.ispinstate}",
            f"  egap={self.egap:.4f} eV  ewidth={self.ewidth:.4f} eV  forcem={self.forcem:.4f}",
            f"  forcek1={self.forcek1:.4f}  forceo1={self.forceo1:.4f}  gediag1={self.gediag1:.4f}",
            f"  forcek2={self.forcek2:.4f}  forceo2={self.forceo2:.4f}  gediag2={self.gediag2:.4f}",
            f"  fediag1={self.fediag1:.4f}  fediag2={self.fediag2:.4f}",
            f"  coulombe={self.coulombe:.4f}  coulombr={self.coulombr:.4f}",
            f"  dipole={self.dipole:.4f}  dipolr={self.dipolr:.4f}",
            f"  potexe={self.potexe:.4f}  potexr={self.potexr:.4f}",
            f"  ekbt={self.ekbt:.4f} eV  phgama={self.phgama:.4f}  elgama={self.elgama:.4f}",
        ]
        return "\n".join(lines)
