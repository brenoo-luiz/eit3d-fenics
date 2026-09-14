"""
Abstract base class for all EIT 3D solvers.

Centralizes matrix assembly, null space and CG+HYPRE solver (DRY).
Subclasses implement solve() following the Template Method pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from eit3d.config import SolverConfig

logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """
    Abstract interface for EIT 3D variational solvers.

    Subclasses: ForwardSolver, DerivativeSolver.
    """

    def __init__(
        self,
        mesh  : dolfinx.mesh.Mesh,
        V     : dolfinx.fem.FunctionSpace,
        config: SolverConfig,
        comm  : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._mesh   = mesh
        self._V      = V
        self._config = config
        self._comm   = comm

    @abstractmethod
    def solve(self) -> dolfinx.fem.Function:
        """Solve the variational system and return the solution."""

    def _assemble_and_solve(
        self,
        a_form: ufl.Form,
        L_form: ufl.Form,
    ) -> dolfinx.fem.Function:
        """
        Assemble A and b, remove null space, solve Au = b.

        Template Method: shared by all concrete solvers.
        Null space = span{1} (constant functions) — enforces uniqueness
        (integral of u on boundary = 0), equivalent to Lagrange multiplier.
        Solver: CG + HYPRE AMG (efficient for elliptic problems).
        """
        A = dolfinx.fem.petsc.assemble_matrix(a_form)
        A.assemble()

        ns_vec = A.createVecLeft()
        ns_vec.set(1.0)
        ns_vec.normalize()
        ns = PETSc.NullSpace().create(vectors=[ns_vec], comm=self._comm)
        A.setNullSpace(ns)
        A.setTransposeNullSpace(ns)

        b = A.createVecRight()
        with b.localForm() as loc_b:
            loc_b.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ns.remove(b)

        ksp = PETSc.KSP().create(self._comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.setTolerances(rtol=self._config.rtol, atol=self._config.atol, max_it=self._config.max_it)
        ksp.setFromOptions()

        u_h = dolfinx.fem.Function(self._V)
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        logger.info("Solver converged in %d iterations", ksp.getIterationNumber())

        A.destroy(); b.destroy(); ns_vec.destroy(); ksp.destroy()
        return u_h
