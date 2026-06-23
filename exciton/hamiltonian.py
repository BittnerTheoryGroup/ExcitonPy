"""
hamiltonian.py
Port of prepare.f from Exciton code (Bittner Group, UH)

Builds the monoexcited CI Hamiltonian for an e-h pair on a polymer chain.

Key routines ported:
  PrepareElStates  -> build_configurations()
  PrepareModes     -> build_phonon_modes()
  PrepareCouplings -> build_el_ph_couplings()
  HSparseCalc      -> build_hamiltonian()
  TransitionDipole -> build_transition_dipoles()

References:
  Karabunarliev & Bittner, JCP 118, 4291 (2003)
  Karabunarliev & Bittner, PRL 90, 057402 (2003)
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .constants import HBAR, EPS_ZERO, C_LIGHT
from .parameters import ExcitonParams
from .geometry import Geometry


# ---------------------------------------------------------------------------
# Monoexcited configurations
# ---------------------------------------------------------------------------

def build_configurations(nsites: int) -> np.ndarray:
    """
    Port of PrepareElStates.

    Generate all monoexcited e-h pair configurations |h, e⟩.
    Returns mic of shape (nqdof, 2): mic[k] = [hole_site, electron_site] (0-based).

    Ordering (matches F90):
      1) On-site excitons: h==e, idisp=0
      2) CT states with increasing e-h separation idisp=1,2,...,nsites-1
    """
    configs = []

    # On-site (Frenkel) excitons
    for i in range(nsites):
        configs.append([i, i])

    # Charge-transfer states, increasing separation
    for idisp in range(1, nsites):
        for i in range(nsites - idisp):
            configs.append([i,        i + idisp])
            configs.append([i + idisp, i       ])

    return np.array(configs, dtype=int)   # shape (nqdof, 2)


# ---------------------------------------------------------------------------
# Phonon normal modes
# ---------------------------------------------------------------------------

def build_phonon_modes(geom: Geometry,
                       params: ExcitonParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of PrepareModes.

    Builds the phonon Hessian and diagonalises it to get normal modes and
    frequencies.

    Returns
    -------
    omega : np.ndarray, shape (ncdof,)   — normal mode frequencies (rad/ps)
    modes : np.ndarray, shape (ncdof, ncdof) — eigenvectors (columns)
    """
    nsites = geom.nsites

    # Number of phonon degrees of freedom
    if params.fediag2 != 0.0:
        ncdof = 2 * nsites
    else:
        ncdof = nsites

    # Build force-constant (Hessian) matrix for branch 1
    o0 = params.forcek1 / HBAR
    o1 = params.forcek1 / HBAR - params.forceo1 / HBAR
    K = (o0**2 + o1**2) / 2.0
    G = (o0**2 - o1**2) / 4.0

    H_ph = np.zeros((ncdof, ncdof))
    for i in range(nsites):
        H_ph[i, i] = K
        for j in range(nsites):
            if i != j:
                H_ph[i, j] = G * geom.tbond1[i, j]

    # Branch 2 if present
    if ncdof > nsites:
        o0 = params.forcek2 / HBAR
        o1 = params.forcek2 / HBAR - params.forceo2 / HBAR
        K2 = (o0**2 + o1**2) / 2.0
        G2 = (o0**2 - o1**2) / 4.0
        for i in range(nsites):
            H_ph[nsites + i, nsites + i] = K2
            for j in range(nsites):
                if i != j:
                    H_ph[nsites + i, nsites + j] = G2 * geom.tbond2[i, j]

    # Diagonalise (symmetric)
    eigvals, eigvecs = eigh(H_ph)
    omega = np.sqrt(np.abs(eigvals))   # frequencies in rad/ps

    return omega, eigvecs, ncdof


# ---------------------------------------------------------------------------
# Electron-phonon coupling derivatives
# ---------------------------------------------------------------------------

def build_el_ph_couplings(geom: Geometry,
                           params: ExcitonParams,
                           mic: np.ndarray,
                           omega: np.ndarray,
                           modes: np.ndarray,
                           ncdof: int) -> Tuple[np.ndarray, List, List]:
    """
    Port of PrepareCouplings.

    Computes hder[kx, i] = derivative of H_el w.r.t. normal mode kx,
    evaluated at the i-th off-diagonal element that has phonon coupling.

    Returns
    -------
    hder  : np.ndarray, shape (ncdof, nder) — el-ph coupling derivatives
    ider  : list of int — row index of coupled element
    jder  : list of int — col index of coupled element
    """
    nqdof = len(mic)

    # helem with cc=1 for all sites (displacement = 1 for each site)
    cc_unity = np.ones(ncdof)
    hsp_ref  = _hsp_calc(mic, geom, params, cc_unity, spinstate=-1, ncdof=ncdof)
    hsp_zero = _hsp_calc(mic, geom, params, np.zeros(ncdof), spinstate=-1, ncdof=ncdof)

    ider = []
    jder = []
    for ia in range(nqdof):
        for ib in range(ia + 1):
            ham1 = _helem(mic[ia, 0], mic[ia, 1], mic[ib, 0], mic[ib, 1],
                          geom, params, cc_unity, ncdof)
            ham2 = _helem(mic[ia, 0], mic[ia, 1], mic[ib, 0], mic[ib, 1],
                          geom, params, np.zeros(ncdof), spinstate=-1, ncdof=ncdof)
            if ham1 - ham2 != 0.0:
                ider.append(ia)
                jder.append(ib)

    nder = len(ider)
    hder = np.zeros((ncdof, nder))

    for kx in range(ncdof):
        cc_mode = modes[:, kx].copy()   # normal mode displacement
        for i in range(nder):
            ia = ider[i]
            ib = jder[i]
            ho = _helem(mic[ia, 0], mic[ia, 1], mic[ib, 0], mic[ib, 1],
                        geom, params, cc_mode, ncdof=ncdof)
            h0 = _helem(mic[ia, 0], mic[ia, 1], mic[ib, 0], mic[ib, 1],
                        geom, params, np.zeros(ncdof), ncdof=ncdof)
            hder[kx, i] = ho - h0

    return hder, ider, jder


# ---------------------------------------------------------------------------
# One-particle matrix elements
# ---------------------------------------------------------------------------

def _felectron(m1: int, m2: int, geom: Geometry, params: ExcitonParams,
               cc: np.ndarray, ncdof: int) -> float:
    """
    Port of felectron1.
    ⟨m1|f_e|m2⟩ — one-electron (CB) Hamiltonian matrix element.
    """
    nsites = geom.nsites
    deltaelec = (params.forcem - 1.0) / (params.forcem + 1.0)

    if geom.thop[m1, m2] != 0.0:
        fval = (geom.thop[m1, m2]
                + params.fediag1 * 0.4 * (cc[m1] + cc[m2]))
        if ncdof > nsites:
            fval += params.fediag2 * 0.4 * (cc[m1 + nsites] + cc[m2 + nsites])
        fcor = fval
        return fval + fcor * deltaelec
    elif m1 == m2:
        fval = (geom.eelec[m1]
                - params.fediag1 * cc[m1])
        if ncdof > nsites:
            fval -= params.fediag2 * cc[m1 + nsites]
        fcor = ((geom.eelec[m1] - geom.ehole[m1]) / 2.0
                - params.fediag1 * cc[m1])
        if ncdof > nsites:
            fcor -= params.fediag2 * cc[m1 + nsites]
        return fval + fcor * deltaelec
    else:
        return 0.0


def _fhole(m1: int, m2: int, geom: Geometry, params: ExcitonParams,
           cc: np.ndarray, ncdof: int) -> float:
    """
    Port of fhole1.
    ⟨m1|f_h|m2⟩ — one-hole (VB) Hamiltonian matrix element.
    """
    nsites = geom.nsites
    deltahole = (1.0 - params.forcem) / (params.forcem + 1.0)

    if geom.thop[m1, m2] != 0.0:
        fval = (geom.thop[m1, m2]
                + params.fediag1 * 0.4 * (cc[m1] + cc[m2]))
        if ncdof > nsites:
            fval += params.fediag2 * 0.4 * (cc[m1 + nsites] + cc[m2 + nsites])
        fcor = fval
        return fval + fcor * deltahole
    elif m1 == m2:
        fval = (-geom.ehole[m1]
                - params.fediag1 * cc[m1])
        if ncdof > nsites:
            fval -= params.fediag2 * cc[m1 + nsites]
        fcor = ((geom.eelec[m1] - geom.ehole[m1]) / 2.0
                - params.fediag1 * cc[m1])
        if ncdof > nsites:
            fcor -= params.fediag2 * cc[m1 + nsites]
        return fval + fcor * deltahole
    else:
        return 0.0


def _helem(h1: int, e1: int, h2: int, e2: int,
           geom: Geometry, params: ExcitonParams,
           cc: np.ndarray, spinstate: int = -1, ncdof: int = None) -> float:
    """
    Port of helem.
    CI matrix element ⟨h1,e1|F|h2,e2⟩ = δ(h1,h2)⟨e1|f_e|e2⟩ + δ(e1,e2)⟨h2|f_h|h1⟩
    """
    if ncdof is None:
        ncdof = geom.nsites
    f = 0.0
    if h1 == h2:
        f += _felectron(e1, e2, geom, params, cc, ncdof)
    if e1 == e2:
        f += _fhole(h1, h2, geom, params, cc, ncdof)
    return f


# ---------------------------------------------------------------------------
# Two-particle interaction terms
# ---------------------------------------------------------------------------

def _coulomb_int(r: float, params: ExcitonParams) -> float:
    """Screened Coulomb: J(r) = J₀ / (1 + r/r₀)"""
    if params.coulombr == 0.0:
        return 0.0
    return params.coulombe / (1.0 + r / params.coulombr)


def _exchange_int(r: float, params: ExcitonParams) -> float:
    """Short-range exchange: K(r) = K₀ exp(-r/r₀)"""
    return params.potexe * np.exp(-r / params.potexr)


def _dipole_int(r: float, params: ExcitonParams) -> float:
    """Dipole-dipole: D(r) = D₀ (r/r₀)⁻³"""
    if r > 0.0:
        return params.dipole / (r / params.dipolr)**3
    return 0.0


def _edirect(h1: int, e1: int, h2: int, e2: int,
             geom: Geometry, params: ExcitonParams) -> float:
    """
    Port of edirect.
    Direct Coulomb: ⟨h1,e1|V_C|h2,e2⟩ = δ(h1,h2)δ(e1,e2) J(|h-e|)
    """
    if h1 == h2 and e1 == e2:
        r = geom.distmat[h1, e1]
        return _coulomb_int(r, params)
    return 0.0


def _eindirect(h1: int, e1: int, h2: int, e2: int,
               geom: Geometry, params: ExcitonParams) -> float:
    """
    Port of eindirect.

    F90 signature: eindirect(m1, n1, n2, m2)
    F90 call:      eindirect(h1, e1, h2, e2)
    Internal mapping: m1=h1, n1=e1, n2=h2, m2=e2  (note: n2=h2, m2=e2)

    Exchange:     m1==m2 AND n1==n2  ->  K(distmat[m1,n1])
                  i.e. h1==e2 AND e1==h2  ->  K(distmat[h1,e1])
    Dipole-dipole: m1==n2 AND m2==n1 AND m1!=m2
                  i.e. h1==h2 AND e1==e2 AND h1!=e2 ... wait, re-derive:
                  m1=h1, n2=h2, m2=e2, n1=e1
                  m1==n2: h1==h2
                  m2==n1: e2==e1
                  m1!=m2: h1!=e2
                  -> D(distmat[h1,e2]) when h1==h2 and e1==e2 and h1!=e2
    """
    # Map F90 internal variables: m1=h1, n1=e1, n2=h2, m2=e2
    m1, n1, n2, m2 = h1, e1, h2, e2

    if m1 == m2 and n1 == n2:
        # Exchange: K(distmat[m1,n1]) = K(distmat[h1,e1])
        r = geom.distmat[m1, n1]
        return _exchange_int(r, params)
    elif m1 == n2 and m2 == n1 and m1 != m2:
        # Dipole-dipole: D(distmat[m1,m2]) = D(distmat[h1,e2])
        r = geom.distmat[m1, m2]
        return _dipole_int(r, params)
    return 0.0


# ---------------------------------------------------------------------------
# Full CI Hamiltonian
# ---------------------------------------------------------------------------

def _hsp_calc(mic: np.ndarray, geom: Geometry, params: ExcitonParams,
              cc: np.ndarray, spinstate: int, ncdof: int) -> np.ndarray:
    """Internal helper: build lower-triangular packed Hamiltonian."""
    nqdof = len(mic)
    nqdof2 = nqdof * (nqdof + 1) // 2
    hs = np.zeros(nqdof2)
    k = 0
    for i in range(nqdof):
        h1, e1 = mic[i]
        for j in range(i + 1):
            h2, e2 = mic[j]
            ho = _helem(h1, e1, h2, e2, geom, params, cc, ncdof=ncdof)
            eJ = _edirect(h1, e1, h2, e2, geom, params)
            eK = _eindirect(h1, e1, h2, e2, geom, params)
            if spinstate == -1:
                hel = ho
            elif spinstate == 0:   # singlet
                hel = ho - eJ + 2.0 * eK
            elif spinstate == 1:   # triplet
                hel = ho - eJ
            hs[k] = hel
            k += 1
    return hs


def build_hamiltonian(mic: np.ndarray,
                      geom: Geometry,
                      params: ExcitonParams,
                      cc: Optional[np.ndarray] = None,
                      ncdof: int = None) -> np.ndarray:
    """
    Port of HSparseCalc + unpack to full symmetric matrix.

    Parameters
    ----------
    mic      : monoexcited configurations, shape (nqdof, 2)
    geom     : Geometry object
    params   : ExcitonParams object
    cc       : nuclear displacements (Angstrom), shape (ncdof,). Default: zeros.
    ncdof    : number of phonon DOF. Default: nsites.

    Returns
    -------
    H : np.ndarray, shape (nqdof, nqdof) — full symmetric Hamiltonian (eV)
    """
    if ncdof is None:
        ncdof = geom.nsites
    if cc is None:
        cc = np.zeros(ncdof)

    spinstate = max(0, params.ispinstate)
    nqdof = len(mic)
    hs = _hsp_calc(mic, geom, params, cc, spinstate, ncdof)

    # Unpack lower-triangular to full matrix
    H = np.zeros((nqdof, nqdof))
    k = 0
    for i in range(nqdof):
        for j in range(i + 1):
            H[i, j] = hs[k]
            H[j, i] = hs[k]
            k += 1
    return H


def diagonalize_hamiltonian(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalise the CI Hamiltonian.

    Returns
    -------
    eigs : eigenvalues (eV), ascending order
    vecs : eigenvectors (columns), shape (nqdof, nqdof)
    """
    eigs, vecs = eigh(H)
    return eigs, vecs


# ---------------------------------------------------------------------------
# Transition dipoles
# ---------------------------------------------------------------------------

def build_transition_dipoles(mic: np.ndarray,
                              geom: Geometry,
                              vecs: np.ndarray) -> np.ndarray:
    """
    Port of TransitionDipole.

    Computes the squared one-photon transition dipole |⟨0|μ|n⟩|² for each
    CI eigenstate n.

    Uses a functional approximation to the PPV dipole matrix elements
    (from JCP 118, 4291):
        dx contribution: TDip_x * exp(-0.8 * r_nm) * cos(π * r_nm)
        dy contribution: TDip_y * exp(-1.9) * r_nm

    Returns
    -------
    dip : np.ndarray, shape (nqdof,) — squared transition dipoles (dimensionless)
    """
    nqdof = len(mic)
    TDip = np.array([[-1.1612, 0.2285],
                     [ 0.5838, 0.0324],
                     [-0.1493, 0.0166]])   # (separation, xy component)

    dip = np.zeros(nqdof)
    for nq in range(nqdof):
        dx = 0.0
        dy = 0.0
        for nq1 in range(nqdof):
            n = mic[nq1, 1]   # electron site
            m = mic[nq1, 0]   # hole site
            rnm = geom.distmat[n, m]
            dx += TDip[0, 0] * np.exp(-0.8 * rnm) * np.cos(np.pi * rnm) * vecs[nq1, nq]
            dy += TDip[0, 1] * np.exp(-1.9) * rnm * vecs[nq1, nq]
        dip[nq] = dx**2 + dy**2
    return dip


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class ExcitonHamiltonian:
    """
    Builds the full exciton Hamiltonian and derived quantities for a given
    set of parameters and geometry.

    Usage
    -----
    >>> geom   = Geometry.uniform_chain(nsites=4)
    >>> params = ExcitonParams()
    >>> exH    = ExcitonHamiltonian(geom, params)
    >>> exH.build()
    >>> print(exH.eigs)     # CI eigenvalues
    >>> print(exH.H)        # full Hamiltonian matrix
    """

    def __init__(self, geom: Geometry, params: ExcitonParams):
        self.geom   = geom
        self.params = params

        # Set after build()
        self.mic     = None   # monoexcited configurations
        self.nqdof   = None   # number of e-h pair states
        self.ncdof   = None   # number of phonon DOF
        self.H       = None   # Hamiltonian matrix
        self.eigs    = None   # eigenvalues
        self.vecs    = None   # eigenvectors
        self.omega   = None   # phonon frequencies
        self.modes   = None   # phonon normal modes
        self.hder    = None   # el-ph coupling derivatives
        self.ider    = None
        self.jder    = None
        self.transdip = None  # squared transition dipoles

    def build(self, cc: np.ndarray = None):
        """
        Full build sequence matching Prepare() in prepare.f.
        """
        # 1. Monoexcited configurations
        self.mic   = build_configurations(self.geom.nsites)
        self.nqdof = len(self.mic)

        # 2. Phonon modes
        self.omega, self.modes, self.ncdof = build_phonon_modes(
            self.geom, self.params)

        # 3. Nuclear displacements (default: equilibrium)
        if cc is None:
            cc = np.zeros(self.ncdof)

        # 4. Electron-phonon coupling derivatives
        self.hder, self.ider, self.jder = build_el_ph_couplings(
            self.geom, self.params, self.mic,
            self.omega, self.modes, self.ncdof)

        # 5. Electronic Hamiltonian and diagonalisation
        self.H    = build_hamiltonian(self.mic, self.geom, self.params,
                                      cc=cc, ncdof=self.ncdof)
        self.eigs, self.vecs = diagonalize_hamiltonian(self.H)

        # 6. Transition dipoles
        self.transdip = build_transition_dipoles(self.mic, self.geom, self.vecs)

        return self

    def summary(self) -> str:
        if self.eigs is None:
            return "ExcitonHamiltonian (not yet built)"
        lines = [
            f"ExcitonHamiltonian: {self.geom.nsites} sites, "
            f"{self.nqdof} CI states, {self.ncdof} phonon modes",
            f"{'State':>6} {'Energy (eV)':>14} {'Rad rate':>12} {'Trans. dip':>12}",
        ]
        for i in range(min(self.nqdof, 20)):
            lines.append(
                f"{i+1:>6} {self.eigs[i]:>14.6f} {'N/A':>12} {self.transdip[i]:>12.6f}"
            )
        return "\n".join(lines)
