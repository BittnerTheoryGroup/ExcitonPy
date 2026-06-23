"""
ExcitonPy — Python port of the Exciton code (Bittner Group, University of Houston)

Builds monoexcited CI Hamiltonians for conjugated polymer chains and interfaces
with QuDPy for nonlinear optical spectroscopy simulations.

References
----------
Karabunarliev & Bittner, J. Chem. Phys. 118, 4291 (2003)
Karabunarliev & Bittner, Phys. Rev. Lett. 90, 057402 (2003)
Shah, Li, Bittner, Silva, Piryatinski, Comput. Phys. Commun. 292, 108891 (2023)
"""

from .constants import HBAR, KB
from .parameters import ExcitonParams
from .geometry import Geometry
from .hamiltonian import ExcitonHamiltonian

__version__ = "0.1.0"
__author__  = "Bittner Research Group, University of Houston"
__all__ = ["ExcitonParams", "Geometry", "ExcitonHamiltonian"]
