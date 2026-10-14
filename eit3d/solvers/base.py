from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Union

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
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_P_max": 4,
    "pc_hypre_boomeramg_agg_nl": 1,
}


class BaseSolver(ABC):

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        V: dolfinx.fem.FunctionSpace,
        gamma: Coefficient,
        config: SolverConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._mesh = mesh
        self._V = V
        self._gamma = gamma
        self._config = config
        self._comm = comm
        self._ds_all = ufl.Measure("ds", domain=mesh)

    # Template Method
    def solve(self) -> dolfinx.fem.Function:
        """Solve the variational problem and return the normalized solution."""
        return self.solve_all()[0]

    def solve_all(self) -> List[dolfinx.fem.Function]:
        w = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)

        a_form = dolfinx.fem.form(self._bilinear_form(w, v))
        L_forms = [dolfinx.fem.form(L) for L in self._linear_forms(v)]

        solutions = self._assemble_and_solve_many(a_form, L_forms)
        for u_h in solutions:
            self._normalize_boundary_mean(u_h)
            logger.info(
                "%s: solution in [%.4f, %.4f]",
                type(self).__name__,
                float(u_h.x.array.min()), float(u_h.x.array.max()),
            )
        return solutions

    # Hooks
    def _bilinear_form(self, w: ufl.Argument, v: ufl.Argument) -> ufl.Form:
        """a(w, v) = int_Omega gamma grad(w).grad(v) dx (shared by all solvers)."""
        return ufl.inner(self._gamma * ufl.grad(w), ufl.grad(v)) * ufl.dx

    @abstractmethod
    def _linear_form(self, v: ufl.Argument) -> ufl.Form:
        """Right-hand side L(v). Implemented by each concrete solver."""

    def _linear_forms(self, v: ufl.Argument) -> List[ufl.Form]:
        return [self._linear_form(v)]

    def _centered_flux(self, f: ufl.core.expr.Expr, ds_f: ufl.Measure, v: ufl.Argument) -> ufl.Form:
        one = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(1.0))
        area = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * self._ds_all)), op=MPI.SUM,
        )
        mean = self._comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * ds_f)), op=MPI.SUM,
        ) / area
        c = dolfinx.fem.Constant(self._mesh, PETSc.ScalarType(mean))
        return f * v * ds_f - c * v * self._ds_all

    # Shared implementation
    def _assemble_and_solve(
        self,
        a_form: dolfinx.fem.Form,
        L_form: dolfinx.fem.Form,
    ) -> dolfinx.fem.Function:
        return self._assemble_and_solve_many(a_form, [L_form])[0]

    def _assemble_and_solve_many(
        self,
        a_form: dolfinx.fem.Form,
        L_forms: List[dolfinx.fem.Form],
    ) -> List[dolfinx.fem.Function]:
        with NeumannSystem(a_form, self._V, self._config, self._comm, owner=type(self).__name__) as system:
            return [system.solve(L_form) for L_form in L_forms]

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


class NeumannSystem:
    """
    Stiffness matrix of -div(gamma grad u) with its constant null space and a
    CG + BoomerAMG solver, assembled once and reused for many right-hand sides.
    """

    def __init__(
        self,
        a_form: dolfinx.fem.Form,
        V: dolfinx.fem.FunctionSpace,
        config: SolverConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
        owner: str = "NeumannSystem",
    ) -> None:
        self._V = V
        self._comm = comm
        self._owner = owner
        self._A = dolfinx.fem.petsc.assemble_matrix(a_form)
        self._A.assemble()

        self._ns_vec = self._A.createVecLeft()
        self._ns_vec.set(1.0)
        self._ns_vec.normalize()
        self._ns = PETSc.NullSpace().create(vectors=[self._ns_vec], comm=comm)
        self._A.setNullSpace(self._ns)
        self._A.setTransposeNullSpace(self._ns)

        self._ksp = PETSc.KSP().create(comm)
        self._ksp.setOperators(self._A)
        self._ksp.setType(PETSc.KSP.Type.CG)
        self._ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        self._ksp.setTolerances(rtol=config.rtol, atol=config.atol, max_it=config.max_it)
        self._set_amg_options(self._ksp)

        self._b = self._A.createVecRight()

    def solve(self, L_form: dolfinx.fem.Form) -> dolfinx.fem.Function:
        """Solution of A u = b(L_form), with b projected onto the range of A."""
        with self._b.localForm() as loc_b:
            loc_b.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self._b, L_form)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self._ns.remove(self._b)

        u_h = dolfinx.fem.Function(self._V)
        self._ksp.solve(self._b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        reason = self._ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"{self._owner}: KSP diverged (reason={reason})")
        logger.info("%s: converged in %d iterations", self._owner, self._ksp.getIterationNumber())
        return u_h

    def close(self) -> None:
        for obj in (self._ksp, self._ns, self._b, self._ns_vec, self._A):
            obj.destroy()

    def __enter__(self) -> "NeumannSystem":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _set_amg_options(ksp: PETSc.KSP) -> None:
        opts = PETSc.Options()
        for key, value in _HYPRE_3D_OPTIONS.items():
            if not opts.hasName(_OPTIONS_PREFIX + key):
                opts[_OPTIONS_PREFIX + key] = value
        ksp.setOptionsPrefix(_OPTIONS_PREFIX)
        ksp.setFromOptions()