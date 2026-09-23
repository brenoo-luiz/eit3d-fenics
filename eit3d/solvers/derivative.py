"""
Directional derivative of the EIT forward operator (eq. 1.13).

Solves: int_Omega gamma grad(w).grad(v) dx = -int_Omega eta grad(u_gamma).grad(v) dx

Reference: Margotti et al. (2023), eq. (1.13), Proposition 1.3.
"""

from __future__ import annotations

import logging

import dolfinx
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver, Coefficient

logger = logging.getLogger(__name__)


class DerivativeSolver(BaseSolver):
    """
    Directional derivative F'(gamma)eta of the EIT forward operator.

    Solves variational equation (1.13):

        int_Omega gamma grad(w).grad(v) dx = -int_Omega eta grad(u_gamma).grad(v) dx

    LHS: same bilinear operator as the forward problem (inherited).
    RHS: volumetric integral over Omega; u_gamma is used in the whole
        domain, not only on the boundary.

    Result: F'(gamma)eta = omega|_{dOmega}, with int_dOmega omega ds = 0.

    Note: eta is the derivative direction in L_inf(Omega),
        NOT the outward normal vector.
    """

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
        self._eta     = eta
        self._u_gamma = u_gamma

    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        logger.info("Derivative problem, eq. (1.13)")
        return -ufl.inner(self._eta * ufl.grad(self._u_gamma), ufl.grad(v)) * ufl.dx