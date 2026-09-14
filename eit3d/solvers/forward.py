"""
EIT 3D forward problem solver.

Solves: int_Omega gamma grad(u).grad(v) dx = int_dOmega g v dS
"""

from __future__ import annotations

import logging

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig, TOP_TAG, BOTTOM_TAG
from eit3d.solvers.base import BaseSolver

logger = logging.getLogger(__name__)


class ForwardSolver(BaseSolver):
    """
    EIT forward problem (Proposition 1.2).

    Given gamma in L_inf_+(Omega) and current g in H_diamond^{-1/2}(dOmega),
    finds u in H^1_diamond(Omega) such that:

        int_Omega gamma grad(u).grad(v) dx = int_dOmega g v dS,  forall v

    Existence and uniqueness guaranteed by Lax-Milgram theorem.
    """

    def __init__(
        self,
        mesh      : dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        V         : dolfinx.fem.FunctionSpace,
        gamma     : dolfinx.fem.Function,
        config    : SolverConfig,
        comm      : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, config, comm)
        self._facet_tags = facet_tags
        self._gamma      = gamma
        self._ds         = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    def solve(self, g_top: float = 1.0, g_bot: float = -1.0) -> dolfinx.fem.Function:
        """
        Solve the forward problem for one current pattern (g_top, g_bot).

        Neumann BC:
            g = g_top  on top face    (z ~ +1)
            g = g_bot  on bottom face (z ~ -1)
            g = 0      on lateral     (no current)

        Existence condition: g_top * pi + g_bot * pi = 0  (g_top = -g_bot).
        """
        logger.info("Solving forward problem: g_top=%.2f, g_bot=%.2f", g_top, g_bot)

        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)

        a = ufl.inner(self._gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = g_top * v * self._ds(TOP_TAG) + g_bot * v * self._ds(BOTTOM_TAG)

        u_h = self._assemble_and_solve(dolfinx.fem.form(a), dolfinx.fem.form(L))

        logger.info("u in [%.4f, %.4f]", float(u_h.x.array.min()), float(u_h.x.array.max()))
        return u_h
