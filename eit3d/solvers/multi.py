from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import dolfinx
import dolfinx.fem
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)

Expr = ufl.core.expr.Expr


class MultiForwardSolver(BaseSolver):

    def __init__(
        self,
        mesh    : dolfinx.mesh.Mesh,
        V       : dolfinx.fem.FunctionSpace,
        gamma   : Coefficient,
        currents: Sequence[Expr],
        ds_g    : ufl.Measure,
        config  : SolverConfig,
        comm    : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._currents = list(currents)
        self._ds_g     = ds_g

    def _linear_forms(self, v: ufl.Argument) -> List[ufl.Form]:
        return [self._centered_flux(g, self._ds_g, v) for g in self._currents]

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        return self._linear_forms(v)[0]


class MultiDerivativeSolver(BaseSolver):

    def __init__(
        self,
        mesh    : dolfinx.mesh.Mesh,
        V       : dolfinx.fem.FunctionSpace,
        gamma   : Coefficient,
        eta     : dolfinx.fem.Function,
        u_gammas: Sequence[dolfinx.fem.Function],
        config  : SolverConfig,
        comm    : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._eta      = eta
        self._u_gammas = list(u_gammas)

    def _linear_forms(self, v: ufl.Argument) -> List[ufl.Form]:
        return [
            -ufl.inner(self._eta * ufl.grad(u), ufl.grad(v)) * ufl.dx
            for u in self._u_gammas
        ]

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        return self._linear_forms(v)[0]


class MultiAdjointSolver(BaseSolver):

    def __init__(
        self,
        mesh    : dolfinx.mesh.Mesh,
        V       : dolfinx.fem.FunctionSpace,
        gamma   : Coefficient,
        u_gammas: Sequence[dolfinx.fem.Function],
        hs      : Sequence[Expr],
        ds_h    : ufl.Measure,
        config  : SolverConfig,
        comm    : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        if len(u_gammas) != len(hs):
            raise ValueError("u_gammas and hs must have the same length")
        super().__init__(mesh, V, gamma, config, comm)
        self._u_gammas = list(u_gammas)
        self._hs       = list(hs)
        self._ds_h     = ds_h
        self._psis     : Optional[List[dolfinx.fem.Function]] = None

    @property
    def psis(self) -> List[dolfinx.fem.Function]:
        if self._psis is None:
            raise RuntimeError("psis are only available after solve()")
        return self._psis

    def solve(self) -> dolfinx.fem.Function:
        self._psis = self.solve_all()

        W    = dolfinx.fem.functionspace(self._mesh, ("DG", 2))
        expr = dolfinx.fem.Expression(
            -sum(ufl.inner(ufl.grad(u), ufl.grad(psi)) for u, psi in zip(self._u_gammas, self._psis)),
            W.element.interpolation_points,
        )
        adj = dolfinx.fem.Function(W)
        adj.interpolate(expr)
        adj.x.scatter_forward()
        return adj

    def _linear_forms(self, v: ufl.Argument) -> List[ufl.Form]:
        return [self._centered_flux(h, self._ds_h, v) for h in self._hs]

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        return self._linear_forms(v)[0]