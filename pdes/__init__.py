from pdes.base import PDESpec, BCSStructure, get_pde, list_pdes, register_pde
from pdes import burgers  # noqa: F401
from pdes import advection  # noqa: F401
from pdes import diffusion_reaction  # noqa: F401
from pdes import allen_cahn  # noqa: F401
from pdes import allen_cahn_2d  # noqa: F401

__all__ = [
    "PDESpec",
    "BCSStructure",
    "get_pde",
    "list_pdes",
    "register_pde",
]
