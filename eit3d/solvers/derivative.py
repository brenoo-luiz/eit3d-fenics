from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class DerivativeSolver(BaseSolver):

    def __init__(
        self,
        mesh   : dolfinx.mesh.Mesh,
        V      : dolfinx.fem.FunctionSpace,
        gamma  : Coefficient,
        eta    : dolfinx.fem.Function,
        u_gamma: dolfinx.fem.Function,
        config : SolverConfig,
        comm   : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._eta = eta
        self._u_gamma = u_gamma

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        logger.info("Derivative problem, eq. (1.13)")
        return -ufl.inner(self._eta * ufl.grad(self._u_gamma), ufl.grad(v)) * ufl.dx