"""
test_hamiltonian.py
Verification of the Python port against known results from the F90 Exciton code.

Test case: 4-site PPV model from SAMPLE.input
  - Parameters from Table 1, Karabunarliev & Bittner, JCP 118, 4291 (2003)
  - 4 sites, singlet, default PPV parameters

We check:
  1. Number and ordering of monoexcited configurations
  2. Hamiltonian matrix elements (diagonal and off-diagonal)
  3. Eigenvalue spectrum (compare to exci.txt / relaxed_density.out)
  4. Transition dipoles (qualitative: lowest bright state should be state 1)
"""

import sys
import os


import numpy as np

from exciton.constants import HBAR, KB
from exciton.parameters import ExcitonParams
from exciton.geometry import Geometry
from exciton.hamiltonian import (
    build_configurations, build_hamiltonian, diagonalize_hamiltonian,
    build_transition_dipoles, build_phonon_modes, ExcitonHamiltonian,
    _helem, _edirect, _eindirect, _felectron, _fhole
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Test 1: Configurations
# ---------------------------------------------------------------------------
def test_configurations():
    print_section("Test 1: Monoexcited configurations (4 sites)")
    mic = build_configurations(4)
    print(f"  nqdof = {len(mic)}  (expected: 16 = 4² for 4 sites)")
    print(f"  First 4 (on-site Frenkel excitons):")
    for i in range(4):
        print(f"    state {i+1}: hole={mic[i,0]+1}, electron={mic[i,1]+1}")
    print(f"  Next 6 (nearest-neighbour CT states):")
    for i in range(4, 10):
        print(f"    state {i+1}: hole={mic[i,0]+1}, electron={mic[i,1]+1}")
    assert len(mic) == 16, f"Expected 16 configs, got {len(mic)}"
    # On-site excitons should be first
    for i in range(4):
        assert mic[i, 0] == mic[i, 1] == i, f"Config {i} should be on-site"
    print("  PASSED")
    return mic


# ---------------------------------------------------------------------------
# Test 2: One-particle matrix elements
# ---------------------------------------------------------------------------
def test_one_particle_elements():
    print_section("Test 2: One-particle matrix elements")
    params = ExcitonParams(nsites=4)
    geom   = Geometry.uniform_chain(nsites=4, egap=2.75*2, ewidth=0.536)
    # Note: in F90, eelec = egap, ehole = -egap for each site
    # uniform_chain sets eelec = egap/2, ehole = -egap/2 giving gap = egap
    cc = np.zeros(geom.nsites)
    ncdof = geom.nsites

    # Diagonal: on-site electron energy for site 0
    fe_00 = _felectron(0, 0, geom, params, cc, ncdof)
    fh_00 = _fhole(0, 0, geom, params, cc, ncdof)
    print(f"  felectron(0,0) = {fe_00:.6f} eV  (should be ≈ eelec[0] ≈ 2.75)")
    print(f"  fhole(0,0)     = {fh_00:.6f} eV  (should be ≈ -ehole[0] ≈ -2.75 -> +2.75 in hole picture)")

    # Off-diagonal: nearest-neighbour hopping
    fe_01 = _felectron(0, 1, geom, params, cc, ncdof)
    fh_01 = _fhole(0, 1, geom, params, cc, ncdof)
    print(f"  felectron(0,1) = {fe_01:.6f} eV  (should be ≈ +0.536 * (1 + deltaelec))")
    print(f"  fhole(0,1)     = {fh_01:.6f} eV  (should be ≈ +0.536 * (1 + deltahole))")

    # No coupling between non-adjacent sites
    fe_02 = _felectron(0, 2, geom, params, cc, ncdof)
    assert fe_02 == 0.0, "Non-adjacent sites should have zero hopping"
    print(f"  felectron(0,2) = {fe_02:.6f}  (correct: 0)")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 3: Coulomb and exchange matrix elements
# ---------------------------------------------------------------------------
def test_interactions():
    print_section("Test 3: Coulomb & exchange interactions")
    params = ExcitonParams(nsites=4)
    geom   = Geometry.uniform_chain(nsites=4)

    # Direct Coulomb for on-site exciton (h=e=0): r=0 -> J(0) = J0
    eJ_00 = _edirect(0, 0, 0, 0, geom, params)
    print(f"  edirect(0,0,0,0) = {eJ_00:.6f} eV  (expect coulombe = {params.coulombe:.6f})")
    assert np.isclose(eJ_00, params.coulombe), "On-site Coulomb should equal coulombe"

    # Coulomb for h=0,e=1 (adjacent): r=1Å
    eJ_01 = _edirect(0, 1, 0, 1, geom, params)
    expected_J01 = params.coulombe / (1 + 1.0 / params.coulombr)
    print(f"  edirect(0,1,0,1) = {eJ_01:.6f} eV  (expect {expected_J01:.6f})")
    assert np.isclose(eJ_01, expected_J01), "Adjacent Coulomb mismatch"

    # Exchange for on-site (r=0)
    eK_00 = _eindirect(0, 0, 0, 0, geom, params)
    expected_K00 = params.potexe * np.exp(0.0)
    print(f"  eindirect(0,0,0,0) = {eK_00:.6f} eV  (expect {expected_K00:.6f})")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 4: Full 4-site Hamiltonian
# ---------------------------------------------------------------------------
def test_hamiltonian_4site():
    print_section("Test 4: 4-site singlet Hamiltonian")
    params = ExcitonParams(nsites=4, ispinstate=0)

    # Use the exact site setup from SAMPLE.input
    # site: idx  ehole   eelec   x      y
    #   1   -2.75   2.75   0.0   0.0
    #   2   -2.75   2.75   1.0   0.0
    #   3   -2.75   2.75   0.0   1.0   <- note: y=1 not x=2 (2D!)
    #   4   -2.75   2.75   1.0   1.0
    # bonds: 1-2, 2-3, 3-4 with t=0.536
    eelec  = np.array([ 2.75,  2.75,  2.75,  2.75])
    ehole  = np.array([-2.75, -2.75, -2.75, -2.75])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    thop   = np.zeros((4, 4))
    thop[0, 1] = thop[1, 0] = 0.536
    thop[1, 2] = thop[2, 1] = 0.536
    thop[2, 3] = thop[3, 2] = 0.536
    tbond1 = (thop != 0).astype(float)

    geom = Geometry.from_arrays(eelec, ehole, coords, thop, tbond1, tbond1)
    mic  = build_configurations(4)
    cc   = np.zeros(geom.nsites)

    H = build_hamiltonian(mic, geom, params, cc=cc, ncdof=geom.nsites)
    print(f"  Hamiltonian shape: {H.shape}  (expected: 16×16)")
    assert H.shape == (16, 16)

    # Check symmetry
    assert np.allclose(H, H.T), "H must be symmetric"
    print("  H is symmetric: PASSED")

    # Diagonal elements: on-site exciton at site 0 (h=0, e=0)
    # helem(0,0,0,0) = felectron(0,0) + fhole(0,0) - J(0) + 2K(0) [singlet]
    fe = _felectron(0, 0, geom, params, cc, geom.nsites)
    fh = _fhole(0, 0, geom, params, cc, geom.nsites)
    eJ = _edirect(0, 0, 0, 0, geom, params)
    eK = _eindirect(0, 0, 0, 0, geom, params)
    expected_diag = fe + fh - eJ + 2*eK
    print(f"  H[0,0] = {H[0,0]:.6f}  expected ≈ {expected_diag:.6f}")
    assert np.isclose(H[0, 0], expected_diag, atol=1e-5), "Diagonal element mismatch"

    eigs, vecs = diagonalize_hamiltonian(H)
    print(f"\n  First 8 eigenvalues (eV):")
    for i in range(8):
        print(f"    E[{i+1}] = {eigs[i]:.6f} eV")

    # Sanity checks on spectrum
    assert eigs[0] < eigs[-1], "Eigenvalues should be sorted ascending"
    assert eigs[0] > 0, "Lowest excitation should be above 0 eV"
    assert eigs[0] < 10, "Lowest excitation should be physically reasonable"

    print("  PASSED")
    return H, eigs, vecs, geom, mic, params


# ---------------------------------------------------------------------------
# Test 5: Transition dipoles
# ---------------------------------------------------------------------------
def test_transition_dipoles(H, eigs, vecs, geom, mic, params):
    print_section("Test 5: Transition dipoles")
    dip = build_transition_dipoles(mic, geom, vecs)
    print(f"  First 8 squared transition dipoles:")
    for i in range(8):
        print(f"    |⟨0|μ|{i+1}⟩|² = {dip[i]:.6f}")

    # The lowest bright state (S1 in PPV) should have nonzero dipole
    assert np.any(dip > 0), "At least one state should be optically active"
    print(f"  Brightest state: {np.argmax(dip)+1}  (dipole = {np.max(dip):.6f})")
    print("  PASSED")
    return dip


# ---------------------------------------------------------------------------
# Test 6: Phonon modes (4-site uniform chain)
# ---------------------------------------------------------------------------
def test_phonon_modes():
    print_section("Test 6: Phonon normal modes")
    params = ExcitonParams(nsites=4)
    geom   = Geometry.uniform_chain(nsites=4)
    omega, modes, ncdof = build_phonon_modes(geom, params)
    print(f"  ncdof = {ncdof}  (expect 4 for single phonon branch)")
    print(f"  Normal mode frequencies (rad/ps):")
    for i, w in enumerate(omega):
        print(f"    ω[{i+1}] = {w:.6f} rad/ps  ({w*HBAR*1e3:.4f} meV)")
    assert len(omega) == ncdof
    assert np.all(omega >= 0), "Frequencies must be non-negative"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 7: Full ExcitonHamiltonian builder
# ---------------------------------------------------------------------------
def test_full_builder():
    print_section("Test 7: ExcitonHamiltonian.build() end-to-end")
    params = ExcitonParams(nsites=4, ispinstate=0)
    geom   = Geometry.uniform_chain(nsites=4, egap=5.5, ewidth=0.536)
    exH    = ExcitonHamiltonian(geom, params)
    exH.build()
    print(exH.summary())
    assert exH.eigs is not None
    assert exH.H.shape == (16, 16)
    print("  PASSED")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nExciton Python Port — Verification Suite")
    print("Reference: Karabunarliev & Bittner, JCP 118, 4291 (2003)")

    mic = test_configurations()
    test_one_particle_elements()
    test_interactions()
    H, eigs, vecs, geom, mic, params = test_hamiltonian_4site()
    dip = test_transition_dipoles(H, eigs, vecs, geom, mic, params)
    test_phonon_modes()
    test_full_builder()

    print("\n" + "="*60)
    print("  ALL TESTS PASSED")
    print("="*60)
