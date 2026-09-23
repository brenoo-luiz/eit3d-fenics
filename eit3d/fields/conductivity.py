"""
Piecewise-constant fields defined by spheres: conductivity gamma and derivative direction eta.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import dolfinx
import dolfinx.fem
import dolfinx.mesh
import numpy as np

from eit3d.config import ConductivityConfig, EtaConfig

logger = logging.getLogger(__name__)


def sphere_indicator(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask of the points strictly inside the sphere (vectorized, O(N))."""
    return np.sum((points - center) ** 2, axis=1) < radius ** 2


class PiecewiseSphereField(ABC):
    """
    DG0 field (one value per cell) equal to `value_in` in the cells whose
    midpoint lies inside any of the spheres, and `value_out` elsewhere.

    Template Method: build() is the fixed algorithm; subclasses only say
    which spheres and values to use.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh) -> None:
        self._mesh  = mesh
        self._space = dolfinx.fem.functionspace(mesh, ("DG", 0))

    # Hooks

    @property
    @abstractmethod
    def name(self) -> str:
        """Field name used in log messages."""

    @property
    @abstractmethod
    def centers(self) -> Sequence[np.ndarray]:
        """Centers of the spheres."""

    @property
    @abstractmethod
    def radius(self) -> float:
        """Common radius of the spheres."""

    @property
    @abstractmethod
    def value_in(self) -> float:
        """Value inside the spheres."""

    @property
    @abstractmethod
    def value_out(self) -> float:
        """Value outside all spheres."""

    # Template Method
    def build(self) -> dolfinx.fem.Function:
        """Build and return the field as a DG0 function."""
        field_fn = dolfinx.fem.Function(self._space)
        field_fn.x.array[:] = self.value_out

        cells, midpoints = self._local_cells_and_midpoints()
        inside = np.zeros(len(cells), dtype=bool)
        for center in self.centers:
            inside |= sphere_indicator(midpoints, center, self.radius)

        # DG0 has one dof per cell; the dofmap gives the dof of each cell,
        # so the result does not depend on how dofs are numbered.
        dofs = self._space.dofmap.list[cells, 0]
        field_fn.x.array[dofs[inside]] = self.value_in
        field_fn.x.scatter_forward()

        logger.info(
            "%s: %d of %d cells inside %d sphere(s) set to %.2f, rest %.2f",
            self.name, int(inside.sum()), len(cells), len(self.centers),
            self.value_in, self.value_out,
        )
        return field_fn

    @property
    def space(self) -> dolfinx.fem.FunctionSpace:
        return self._space

    def _local_cells_and_midpoints(self):
        """Cells owned by this process (ghosts are filled by scatter_forward)."""
        tdim  = self._mesh.topology.dim
        n_loc = self._mesh.topology.index_map(tdim).size_local
        cells = np.arange(n_loc, dtype=np.int32)
        return cells, dolfinx.mesh.compute_midpoints(self._mesh, tdim, cells)


class ConductivityField(PiecewiseSphereField):
    """
    Conductivity gamma: gamma_in inside the inclusion sphere, gamma_out outside.

    Lax-Milgram conditions guaranteed by the config:
        0 < gamma_min <= gamma(x) <= gamma_max  a.e. in Omega
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, config: ConductivityConfig) -> None:
        super().__init__(mesh)
        self._config = config

    @property
    def name(self) -> str:
        return "gamma"

    @property
    def centers(self) -> Sequence[np.ndarray]:
        return (self._config.center,)

    @property
    def radius(self) -> float:
        return self._config.radius

    @property
    def value_in(self) -> float:
        return self._config.gamma_in

    @property
    def value_out(self) -> float:
        return self._config.gamma_out


class DirectionalField(PiecewiseSphereField):
    """
    Derivative direction eta for F'(gamma)eta: eta_in inside each sphere,
    eta_out outside all of them.

    Note: eta here is the derivative direction, NOT the outward normal.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, config: EtaConfig) -> None:
        super().__init__(mesh)
        self._config = config

    @property
    def name(self) -> str:
        return "eta"

    @property
    def centers(self) -> Sequence[np.ndarray]:
        return self._config.centers

    @property
    def radius(self) -> float:
        return self._config.radius

    @property
    def value_in(self) -> float:
        return self._config.eta_in

    @property
    def value_out(self) -> float:
        return self._config.eta_out