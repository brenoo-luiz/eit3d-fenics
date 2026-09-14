"""
Directional derivative of the EIT forward operator (eq. 1.13).

Solves: int_Omega gamma grad(w).grad(v) dx = -int_Omega eta grad(u_gamma).grad(v) dx

Reference: Margotti et al. (2023), eq. (1.13), Proposition 1.3.
"""

from __future__ import annotations

import logging

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import BaseSolver

logger = logging.getLogger(__name__)


class DerivativeSolver(BaseSolver):
    """
    Directional derivative F'(gamma)eta of the EIT forward operator.

    Solves variational equation (1.13):

        int_Omega gamma grad(w).grad(v) dx = -int_Omega eta grad(u_gamma).grad(v) dx

    LHS: same bilinear operator as the forward problem (uses gamma).
    RHS: volumetric integral over Omega — u_gamma used ENTIRE in Omega,
        not just on the boundary.

    Result: F'(gamma)eta = omega|_{dOmega}

    Note: eta is the derivative direction in L_inf(Omega),
        NOT the outward normal vector.
    """

    def __init__(
        self,
        mesh   : dolfinx.mesh.Mesh,
        V      : dolfinx.fem.FunctionSpace,
        gamma  : dolfinx.fem.Function,
        eta    : dolfinx.fem.Function,
        u_gamma: dolfinx.fem.Function,
        config : SolverConfig,
        comm   : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        super().__init__(mesh, V, config, comm)
        self._gamma   = gamma
        self._eta     = eta
        self._u_gamma = u_gamma

    def solve(self) -> dolfinx.fem.Function:
        """
        Solve eq. (1.13) and return omega.

        Returns
        -------
        dolfinx.fem.Function
            omega in H^1_diamond(Omega) such that F'(gamma)eta = omega|_{dOmega}.
        """
        logger.info("Solving eq. (1.13) for omega ...")

        w = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)

        a = ufl.inner(self._gamma * ufl.grad(w), ufl.grad(v)) * ufl.dx
        L = -ufl.inner(self._eta * ufl.grad(self._u_gamma), ufl.grad(v)) * ufl.dx

        omega = self._assemble_and_solve(dolfinx.fem.form(a), dolfinx.fem.form(L))

        logger.info("omega in [%.4f, %.4f]", float(omega.x.array.min()), float(omega.x.array.max()))
        return omega
