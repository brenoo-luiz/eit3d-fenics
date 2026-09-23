"""
Generic pure Neumann solver with a flux defined on the whole boundary.

Solves: int_Omega gamma grad(u).grad(v) dx = int_dOmega g v ds

Used by the manufactured-solution consistency test, where g is a UFL
expression such as dot(grad(u_exact), n).
"""

from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class NeumannSolver(BaseSolver):
    """
    Pure Neumann problem with an arbitrary boundary flux g (UFL expression).

    The compatibility condition int_dOmega g ds = 0 is the caller's
    responsibility; any residual mean is removed by the null space.
    """

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