from __future__ import annotations

import logging
from typing import Optional

import dolfinx
import dolfinx.fem
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class AdjointSolver(BaseSolver):

    def __init__(
        self,
        mesh   : dolfinx.mesh.Mesh,
        V      : dolfinx.fem.FunctionSpace,
        gamma  : Coefficient,
        u_gamma: dolfinx.fem.Function,
        h      : ufl.core.expr.Expr,
        config : SolverConfig,
        comm   : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._u_gamma = u_gamma
        self._h       = self._center_on_boundary(h)
        self._psi     : Optional[dolfinx.fem.Function] = None

    @property
    def psi(self) -> dolfinx.fem.Function:
        if self._psi is None:
            raise RuntimeError("psi is only available after solve()")
        return self._psi

    def solve(self) -> dolfinx.fem.Function:
        """Solve for psi and return F'(gamma)* h = -grad(u_gamma).grad(psi) in DG2."""
        self._psi = super().solve()

        W    = dolfinx.fem.functionspace(self._mesh, ("DG", 2))
        expr = dolfinx.fem.Expression(
            -ufl.inner(ufl.grad(self._u_gamma), ufl.grad(self._psi)),
            W.element.interpolation_points,
        )
        adj = dolfinx.fem.Function(W)
        adj.interpolate(expr)
        adj.x.scatter_forward()

        logger.info(
            "F'(gamma)* h in [%.4f, %.4f]",
            float(adj.x.array.min()), float(adj.x.array.max()),
        )
        return adj

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        return self._h * v * self._ds_all

    def _center_on_boundary(self, h: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        """Return h - mean_dOmega(h), so that int_dOmega h ds = 0."""
        one      = dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(1.0))
        integral = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(h * self._ds_all)), op=MPI.SUM,
        )
        area     = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * self._ds_all)), op=MPI.SUM,
        )
        logger.info("h boundary mean removed: %.3e", integral / area)
        return h - dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(integral / area))