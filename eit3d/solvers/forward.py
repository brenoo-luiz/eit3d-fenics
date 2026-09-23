"""
EIT 3D forward problem solver.

Solves: int_Omega gamma grad(u).grad(v) dx = int_dOmega g v dS
"""

from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig, TOP_TAG, BOTTOM_TAG
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class ForwardSolver(BaseSolver):
    """
    EIT forward problem (Proposition 1.2).

    Given gamma in L_inf_+(Omega) and current g in H_diamond^{-1/2}(dOmega),
    finds u in H^1_diamond(Omega) such that:

        int_Omega gamma grad(u).grad(v) dx = int_dOmega g v dS,  forall v

    Neumann BC (current pattern given in the constructor):
        g = g_top  on top face
        g = g_bot  on bottom face
        g = 0      on lateral surface

    Existence condition: int_dOmega g ds = 0, i.e. g_top = -g_bot
    (top and bottom faces have the same area). Validated in CurrentConfig.

    Existence and uniqueness guaranteed by the Lax-Milgram theorem.
    """

    def __init__(
        self,
        mesh      : dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        V         : dolfinx.fem.FunctionSpace,
        gamma     : Coefficient,
        config    : SolverConfig,
        comm      : MPI.Comm = MPI.COMM_WORLD,
        g_top     : float    = 1.0,
        g_bot     : float    = -1.0,
    ) -> None:
        super().__init__(mesh, V, gamma, config, comm)
        self._g_top = g_top
        self._g_bot = g_bot
        self._ds    = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        logger.info(
            "Forward problem: g_top=%.2f, g_bot=%.2f", self._g_top, self._g_bot,
        )
        return (
            self._g_top * v * self._ds(TOP_TAG)
            + self._g_bot * v * self._ds(BOTTOM_TAG)
        )