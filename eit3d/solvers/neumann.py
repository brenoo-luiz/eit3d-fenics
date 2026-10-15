from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class NeumannSolver(BaseSolver):

    def __init__(
        self,
        mesh  : dolfinx.mesh.Mesh,
        V     : dolfinx.fem.FunctionSpace,
        gamma : Coefficient,
        g     : ufl.core.expr.Expr,
        config: SolverConfig,
        comm  : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._g = g

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        return self._g * v * self._ds_all