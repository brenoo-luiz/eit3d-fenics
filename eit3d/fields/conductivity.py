"""
eit3d/fields/conductivity.py
=============================
Conductivity field gamma and directional field eta (DG0).
"""

from __future__ import annotations

import logging
from typing import List

import dolfinx
import dolfinx.mesh
import numpy as np

from eit3d.config import ConductivityConfig, EtaConfig

logger = logging.getLogger(__name__)


class ConductivityField:
    """
    Conductivity field gamma defined as DG0 (constant per cell).

    gamma = gamma_in  inside the inclusion sphere
    gamma = gamma_out outside (background)

    Lax-Milgram conditions guaranteed by config:
        0 < gamma_min <= gamma(x) <= gamma_max  a.e. in Omega
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, config: ConductivityConfig) -> None:
        self._mesh   = mesh
        self._config = config
        self._space  = dolfinx.fem.functionspace(mesh, ("DG", 0))

    def build(self) -> dolfinx.fem.Function:
        """Build and return gamma as a DG0 function."""
        gamma     = dolfinx.fem.Function(self._space)
        midpoints = self._compute_midpoints()

        gamma.x.array[:] = self._config.gamma_out
        inside = self._sphere_indicator(midpoints, self._config.center, self._config.radius)
        gamma.x.array[inside] = self._config.gamma_in
        gamma.x.scatter_forward()

        logger.info(
            "gamma: %d cells with gamma=%.1f (sphere), %d with gamma=%.1f (background)",
            int(inside.sum()), self._config.gamma_in,
            len(gamma.x.array) - int(inside.sum()), self._config.gamma_out,
        )
        return gamma

    def _compute_midpoints(self) -> np.ndarray:
        tdim  = self._mesh.topology.dim
        n_loc = self._mesh.topology.index_map(tdim).size_local
        cells = np.arange(n_loc, dtype=np.int32)
        return dolfinx.mesh.compute_midpoints(self._mesh, tdim, cells)

    @staticmethod
    def _sphere_indicator(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        return np.sum((points - center) ** 2, axis=1) < radius ** 2

    @property
    def space(self) -> dolfinx.fem.FunctionSpace:
        return self._space


class DirectionalField:
    """
    Directional field eta for the derivative F'(gamma)eta (DG0).

    eta = eta_in  inside each eta sphere
    eta = eta_out outside all spheres

    Note: eta here is the derivative direction, NOT the outward normal.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, config: EtaConfig) -> None:
        self._mesh   = mesh
        self._config = config
        self._space  = dolfinx.fem.functionspace(mesh, ("DG", 0))

    def build(self) -> dolfinx.fem.Function:
        """Build and return eta as a DG0 function."""
        eta       = dolfinx.fem.Function(self._space)
        midpoints = self._compute_midpoints()

        eta.x.array[:] = self._config.eta_out
        n_marked = 0
        for center in self._config.centers:
            inside = ConductivityField._sphere_indicator(midpoints, center, self._config.radius)
            eta.x.array[inside] = self._config.eta_in
            n_marked += int(inside.sum())

        eta.x.scatter_forward()
        logger.info(
            "eta: %d cells with eta=%.1f (%d sphere(s)), rest eta=%.1f",
            n_marked, self._config.eta_in, len(self._config.centers), self._config.eta_out,
        )
        return eta

    def _compute_midpoints(self) -> np.ndarray:
        tdim  = self._mesh.topology.dim
        n_loc = self._mesh.topology.index_map(tdim).size_local
        cells = np.arange(n_loc, dtype=np.int32)
        return dolfinx.mesh.compute_midpoints(self._mesh, tdim, cells)

    @property
    def space(self) -> dolfinx.fem.FunctionSpace:
        return self._space
