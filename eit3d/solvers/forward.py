from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import BOTTOM_TAG, TOP_TAG, SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class ForwardSolver(BaseSolver):
    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        V: dolfinx.fem.FunctionSpace,
        gamma: Coefficient,
        config: SolverConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
        g_top: float = 1.0,
        g_bot: float = -1.0,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._g_top = g_top
        self._g_bot = g_bot
        self._ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        logger.info(
            "Forward problem: g_top=%.2f, g_bot=%.2f",
            self._g_top,
            self._g_bot,
        )
        return self._g_top * v * self._ds(TOP_TAG) + self._g_bot * v * self._ds(
            BOTTOM_TAG
        )
