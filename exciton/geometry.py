"""
geometry.py
Port of structure.fi, coords.fi, and geometry setup from input.f

Handles:
  - Site definition (electron/hole energies, positions)
  - Hopping matrix (thop) and phonon bond matrices (tbond1, tbond2)
  - Distance matrix
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Geometry:
    """
    Defines the spatial and electronic structure of the polymer chain.

    Attributes
    ----------
    nsites  : number of sites
    eelec   : electron (CB) site energies, shape (nsites,)
    ehole   : hole (VB) site energies, shape (nsites,)
    coords  : site coordinates, shape (nsites, 2)  [x, y in Angstrom]
    thop    : hopping integral matrix, shape (nsites, nsites)
    tbond1  : phonon branch 1 bond connectivity, shape (nsites, nsites)
    tbond2  : phonon branch 2 bond connectivity, shape (nsites, nsites)
    distmat : inter-site distance matrix, shape (nsites, nsites)
    """
    nsites:  int
    eelec:   np.ndarray   # shape (nsites,)
    ehole:   np.ndarray   # shape (nsites,)
    coords:  np.ndarray   # shape (nsites, 2)
    thop:    np.ndarray   # shape (nsites, nsites)
    tbond1:  np.ndarray   # shape (nsites, nsites)
    tbond2:  np.ndarray   # shape (nsites, nsites)
    distmat: np.ndarray   # shape (nsites, nsites)

    @classmethod
    def from_arrays(cls,
                    eelec: np.ndarray,
                    ehole: np.ndarray,
                    coords: np.ndarray,
                    thop: np.ndarray,
                    tbond1: Optional[np.ndarray] = None,
                    tbond2: Optional[np.ndarray] = None) -> 'Geometry':
        """
        Build a Geometry from arrays.  Distance matrix is computed automatically.
        tbond1/tbond2 default to the same connectivity as thop (nonzero = bonded).
        """
        nsites = len(eelec)
        assert ehole.shape == (nsites,)
        assert coords.shape == (nsites, 2)
        assert thop.shape == (nsites, nsites)

        if tbond1 is None:
            tbond1 = (thop != 0).astype(float)
        if tbond2 is None:
            tbond2 = tbond1.copy()

        distmat = cls._build_distmat(coords)
        return cls(nsites=nsites, eelec=eelec, ehole=ehole,
                   coords=coords, thop=thop,
                   tbond1=tbond1, tbond2=tbond2, distmat=distmat)

    @staticmethod
    def _build_distmat(coords: np.ndarray) -> np.ndarray:
        """Euclidean distance matrix from (nsites, 2) coordinate array."""
        n = len(coords)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r = np.sqrt(np.sum((coords[i] - coords[j])**2))
                d[i, j] = r
                d[j, i] = r
        return d

    @classmethod
    def uniform_chain(cls, nsites: int, egap: float = 2.75,
                      ewidth: float = 0.536, spacing: float = 1.0) -> 'Geometry':
        """
        Convenience constructor: uniform 1D chain with equal site energies
        and nearest-neighbour hopping.

        Parameters
        ----------
        nsites  : number of sites
        egap    : half the gap between band centres (eV);
                  eelec = +egap/2, ehole = -egap/2  per site
        ewidth  : nearest-neighbour transfer integral (eV)
        spacing : inter-site distance (Angstrom)
        """
        eelec  = np.full(nsites,  egap / 2.0)
        ehole  = np.full(nsites, -egap / 2.0)
        coords = np.zeros((nsites, 2))
        coords[:, 0] = np.arange(nsites) * spacing

        thop   = np.zeros((nsites, nsites))
        tbond1 = np.zeros((nsites, nsites))
        tbond2 = np.zeros((nsites, nsites))
        for i in range(nsites - 1):
            thop[i, i+1] = ewidth
            thop[i+1, i] = ewidth
            tbond1[i, i+1] = 1.0
            tbond1[i+1, i] = 1.0
            tbond2[i, i+1] = 1.0
            tbond2[i+1, i] = 1.0

        distmat = cls._build_distmat(coords)
        return cls(nsites=nsites, eelec=eelec, ehole=ehole,
                   coords=coords, thop=thop,
                   tbond1=tbond1, tbond2=tbond2, distmat=distmat)

    @classmethod
    def from_input_file(cls, filepath: str, nsites: int) -> 'Geometry':
        """
        Parse the geometry block from a SAMPLE.input style file.

        After the namelist blocks, the file contains:
          Line per site:  idx  ehole  eelec  x  y
          Bond lines:     i  j  thop  tbond1  tbond2   (terminated by 999 999 ...)
        """
        eelec  = np.zeros(nsites)
        ehole  = np.zeros(nsites)
        coords = np.zeros((nsites, 2))
        thop   = np.zeros((nsites, nsites))
        tbond1 = np.zeros((nsites, nsites))
        tbond2 = np.zeros((nsites, nsites))

        with open(filepath) as f:
            lines = f.readlines()

        # Skip namelist blocks: find lines that look like data (start with integer)
        data_lines = []
        in_nml = False
        for line in lines:
            s = line.strip()
            if not s or s.startswith('!') or s.startswith('#'):
                continue
            if s.startswith('&'):
                in_nml = True
                continue
            if s == '/':
                in_nml = False
                continue
            if not in_nml and s and s[0].isdigit():
                data_lines.append(s)

        # First nsites lines: site energies
        for k in range(nsites):
            parts = data_lines[k].split()
            idx  = int(parts[0]) - 1   # convert to 0-based
            ehole[idx]   = float(parts[1])
            eelec[idx]   = float(parts[2])
            coords[idx, 0] = float(parts[3])
            coords[idx, 1] = float(parts[4])

        # Remaining lines: bonds (terminated by 999)
        for line in data_lines[nsites:]:
            parts = line.split()
            i = int(parts[0])
            j = int(parts[1])
            if i == 999:
                break
            i -= 1; j -= 1   # 0-based
            t   = float(parts[2])
            tb1 = float(parts[3])
            tb2 = float(parts[4]) if len(parts) > 4 else tb1
            thop[i, j]   = t;   thop[j, i]   = t
            tbond1[i, j] = tb1; tbond1[j, i] = tb1
            tbond2[i, j] = tb2; tbond2[j, i] = tb2

        distmat = cls._build_distmat(coords)
        return cls(nsites=nsites, eelec=eelec, ehole=ehole,
                   coords=coords, thop=thop,
                   tbond1=tbond1, tbond2=tbond2, distmat=distmat)

    def summary(self) -> str:
        lines = [f"Geometry: {self.nsites} sites"]
        lines.append(f"{'Site':>6} {'eelec':>10} {'ehole':>10} {'x':>8} {'y':>8}")
        for i in range(self.nsites):
            lines.append(f"{i+1:>6} {self.eelec[i]:>10.4f} {self.ehole[i]:>10.4f} "
                         f"{self.coords[i,0]:>8.4f} {self.coords[i,1]:>8.4f}")
        lines.append("Hopping matrix:")
        for row in self.thop:
            lines.append("  " + "  ".join(f"{v:8.4f}" for v in row))
        return "\n".join(lines)
