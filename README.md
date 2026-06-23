# ExcitonPy

A Python port of the **Exciton** Fortran 90 code (Bittner Research Group, University of Houston), originally written in collaboration with the groups of Carlos Silva (Georgia Tech) and Jenny Nelson (Imperial College London).

ExcitonPy builds monoexcited configuration-interaction (CI) Hamiltonians for conjugated polymer chains, including:

- Electron–hole pair (exciton and charge-transfer) states
- Screened Coulomb, short-range exchange, and dipole–dipole interactions
- Two-branch electron–phonon coupling
- Phonon normal modes and el–ph coupling derivatives
- Singlet and triplet spin states

It is designed to feed directly into **[QuDPy](https://github.com/ebitnet65/QuDPy)** for computing nonlinear optical spectra (2DCS, etc.) via the Lindblad master equation.

---

## References

1. S. Karabunarliev & E. R. Bittner, *Polaron-excitons and electron-vibrational band shapes in conjugated polymers*, J. Chem. Phys. **118**, 4291 (2003)
2. S. Karabunarliev & E. R. Bittner, *Spin-dependent electron-hole recombination kinetics in luminescent organic polymers*, Phys. Rev. Lett. **90**, 057402 (2003)
3. E. R. Bittner & S. Karabunarliev, *Energy relaxation dynamics and universal scaling laws in organic light emitting diodes*, Int. J. Quant. Chem. **95**, 521 (2003)
4. S. A. Shah, H. Li, E. R. Bittner, C. Silva, A. Piryatinski, *QuDPy: A Python-based tool for computing ultrafast non-linear optical responses*, Comput. Phys. Commun. **292**, 108891 (2023)

---

## Installation

```bash
pip install numpy scipy
git clone https://github.com/ebitnet65/ExcitonPy.git
cd ExcitonPy
pip install -e .
```

To also install QuDPy integration dependencies:

```bash
pip install -e ".[qudpy]"
```

---

## Package Structure

```
ExcitonPy/
├── exciton/
│   ├── __init__.py
│   ├── constants.py      # Physical constants (eV, ps, Å units)
│   ├── parameters.py     # ExcitonParams dataclass (all model parameters)
│   ├── geometry.py       # Geometry class (site energies, coordinates, hoppings)
│   └── hamiltonian.py    # CI Hamiltonian builder and ExcitonHamiltonian class
├── tests/
│   └── test_hamiltonian.py   # Verification against F90 benchmark (4-site PPV)
├── examples/             # (coming soon)
├── docs/                 # (coming soon)
├── pyproject.toml
└── README.md
```

---

## Quick Start

```python
from exciton import ExcitonParams, Geometry, ExcitonHamiltonian

# Default PPV parameters (Table 1, JCP 118, 4291)
params = ExcitonParams(nsites=4, ispinstate=0)

# Uniform 1D chain geometry
geom = Geometry.uniform_chain(nsites=4, egap=5.5, ewidth=0.536)

# Build CI Hamiltonian
exH = ExcitonHamiltonian(geom, params)
exH.build()

print(exH.summary())
# Eigenvalues (eV), eigenvectors, transition dipoles all available:
print(exH.eigs)      # CI eigenvalues
print(exH.transdip)  # squared transition dipoles
```

### From an input file

```python
from exciton import ExcitonParams, Geometry, ExcitonHamiltonian

params = ExcitonParams.from_input_file("SAMPLE.input")
geom   = Geometry.from_input_file("SAMPLE.input", nsites=params.nsites)
exH    = ExcitonHamiltonian(geom, params).build()
```

### Feeding into QuDPy

```python
from qutip import Qobj
from qudpy.Classes import System

# Convert H and dipole operator to QuTiP objects
import numpy as np
H_qobj = Qobj(exH.H)
u_qobj = Qobj(exH.vecs.T @ np.diag(exH.transdip) @ exH.vecs)

sys = System(H=H_qobj, u=u_qobj, ...)
```

---

## Model Physics

ExcitonPy implements a monoexcited CI model on a 1D site lattice. Each site `m` carries an electron (conduction band) level `ε_m^e` and a hole (valence band) level `ε_m^h`. The basis states are electron-hole pairs `|h, e⟩`.

### Hamiltonian

```
H = Σ F_ij |i⟩⟨j|  +  V_ij (singlet/triplet)
```

**One-particle terms (F):**
- Diagonal: `f_e(m,m) = ε_m^e − g₁ x_m`,  `f_h(m,m) = −ε_m^h − g₁ x_m`
- Off-diagonal: `f_e(m,n) = t_{mn} + g₁·0.4·(x_m + x_n)`  (nearest neighbours)

**Two-particle interactions (V):**
| Interaction | Expression |
|-------------|------------|
| Direct Coulomb | `J(r) = J₀ / (1 + r/r₀)` |
| Exchange | `K(r) = K₀ exp(−r/r₀)` |
| Dipole–dipole | `D(r) = D₀ (r/r₀)⁻³` |

**Spin:**
- Singlet: `H = F − J + 2K`
- Triplet: `H = F − J`

### Default Parameters (PPV, JCP 118, 4291)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `egap` | 2.75 eV | Half band gap |
| `ewidth` | 0.536 eV | Hopping integral |
| `coulombe` | 3.092 eV | Coulomb amplitude |
| `coulombr` | 0.684 Å | Coulomb screening length |
| `potexe` | 1.057 eV | Exchange amplitude |
| `potexr` | 0.474 Å | Exchange range |
| `dipole` | −0.032 eV·Å³ | Dipole–dipole amplitude |
| `ekbt` | 0.025 eV | Thermal energy (≈300 K) |

---

## Verification

The diabatic CI eigenvalues are verified against the original F90 Exciton code benchmark (4-site PPV model) to sub-0.1 meV precision:

```
State   Python (eV)      F90 (eV)     Diff (meV)
    1   2.7679919    2.7679919        0.000  ✓
    2   2.8739943    2.8739943        0.000  ✓
    3   3.2731887    3.2731887        0.000  ✓
   ...
   16   6.2408820    6.2408820        0.000  ✓
```

Run the test suite:

```bash
cd ExcitonPy
python -m pytest tests/ -v
```

---

## Roadmap

- [x] CI Hamiltonian (diabatic)
- [x] Phonon normal modes
- [x] Electron–phonon coupling derivatives
- [x] Transition dipoles
- [ ] Redfield tensor (`CalcRateConst`)
- [ ] Adiabatic/polaron states (`CalcRelaxedStates`)
- [ ] QuDPy `System` builder
- [ ] Disorder averaging
- [ ] Input file parser (full SAMPLE.input support)

---

## License

GNU General Public License v2. See `LICENSE`.

Original F90 source code © 2004 Eric R. Bittner.
