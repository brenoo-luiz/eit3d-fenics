from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

import dolfinx
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from eit3d.config import SolverConfig

logger = logging.getLogger(__name__)

Coefficient = Union[dolfinx.fem.Function, dolfinx.fem.Constant]

_OPTIONS_PREFIX = "eit3d_"
_HYPRE_3D_OPTIONS = {
    "pc_hypre_type"                      : "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_coarsen_type"    : "HMIS",
    "pc_hypre_boomeramg_interp_type"     : "ext+i",
    "pc_hypre_boomeramg_P_max"           : 4,
    "pc_hypre_boomeramg_agg_nl"          : 1,
}


class BaseSolver(ABC):

    def __init__(
        self,
        mesh  : dolfinx.mesh.Mesh,
        V     : dolfinx.fem.FunctionSpace,
        gamma : Coefficient,
        config: SolverConfig,
        comm  : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._mesh   = mesh
        self._V      = V
        self._gamma  = gamma
        self._config = config
        self._comm   = comm
        self._ds_all = ufl.Measure("ds", domain=mesh)

    # Template Method
    def solve(self) -> dolfinx.fem.Function:
        """Solve the variational problem and return the normalized solution."""
        w = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)

        a_form = dolfinx.fem.form(self._bilinear_form(w, v))
        L_form = dolfinx.fem.form(self._linear_form(v))

        u_h = self._assemble_and_solve(a_form, L_form)
        self._normalize_boundary_mean(u_h)

        logger.info(
            "%s: solution in [%.4f, %.4f]",
            type(self).__name__,
            float(u_h.x.array.min()), float(u_h.x.array.max()),
        )
        return u_h

    # Hooks
    def _bilinear_form(self, w: ufl.Argument, v: ufl.Argument) -> ufl.Form:
        """a(w, v) = int_Omega gamma grad(w).grad(v) dx (shared by all solvers)."""
        return ufl.inner(self._gamma * ufl.grad(w), ufl.grad(v)) * ufl.dx

    @abstractmethod
    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        """Right-hand side L(v). Implemented by each concrete solver."""

    # Shared implementation
    def _assemble_and_solve(
        self,
        a_form: dolfinx.fem.Form,
        L_form: dolfinx.fem.Form,
    ) -> dolfinx.fem.Function:
        A = b = ns_vec = ns = ksp = None
        try:
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
            ksp.setTolerances(
                rtol=self._config.rtol,
                atol=self._config.atol,
                max_it=self._config.max_it,
            )
            self._set_amg_options(ksp)

            u_h = dolfinx.fem.Function(self._V)
            ksp.solve(b, u_h.x.petsc_vec)
            u_h.x.scatter_forward()

            reason = ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(
                    f"{type(self).__name__}: KSP diverged (reason={reason})"
                )
            logger.info(
                "%s: converged in %d iterations",
                type(self).__name__, ksp.getIterationNumber(),
            )
            return u_h
        finally:
            for obj in (ksp, ns, b, ns_vec, A):
                if obj is not None:
                    obj.destroy()

    @staticmethod
    def _set_amg_options(ksp: PETSc.KSP) -> None:
        opts = PETSc.Options()
        for key, value in _HYPRE_3D_OPTIONS.items():
            if not opts.hasName(_OPTIONS_PREFIX + key):
                opts[_OPTIONS_PREFIX + key] = value
        ksp.setOptionsPrefix(_OPTIONS_PREFIX)
        ksp.setFromOptions()

    def _normalize_boundary_mean(self, u_h: dolfinx.fem.Function) -> None:
        integral = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_h * self._ds_all)),
            op=MPI.SUM,
        )
        area = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(
                dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(1.0)) * self._ds_all
            )),
            op=MPI.SUM,
        )
        u_h.x.array[:] -= integral / area
        u_h.x.scatter_forward()