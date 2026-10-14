from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence

import dolfinx
import dolfinx.fem
import dolfinx.la
import numpy as np
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.base import NeumannSystem
from eit3d.solvers.multi import MultiAdjointSolver, MultiForwardSolver

logger = logging.getLogger(__name__)

TRANSFER_PADDING = 1e-2


class NonPositiveConductivity(RuntimeError):

    def __init__(self, iteration: int, minimum: float) -> None:
        super().__init__(f"conductivity became non-positive at iteration {iteration} (min {minimum:.3e})")
        self.iteration = iteration
        self.minimum = minimum


def transfer(u_from: dolfinx.fem.Function, V_to: dolfinx.fem.FunctionSpace,
            padding: float = TRANSFER_PADDING) -> dolfinx.fem.Function:
    mesh_to = V_to.mesh
    n_cells = mesh_to.topology.index_map(mesh_to.topology.dim).size_local
    cells = np.arange(n_cells, dtype=np.int32)
    data = dolfinx.fem.create_interpolation_data(V_to, u_from.function_space, cells, padding=padding)
    u_to = dolfinx.fem.Function(V_to)
    u_to.interpolate_nonmatching(u_from, cells, data)
    u_to.x.scatter_forward()
    return u_to


def project_dg0(f: dolfinx.fem.Function, V0: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Function:
    q = ufl.TestFunction(V0)
    rhs = dolfinx.fem.assemble_vector(dolfinx.fem.form(f * q * ufl.dx))
    vol = dolfinx.fem.assemble_vector(dolfinx.fem.form(q * ufl.dx))
    for vec in (rhs, vol):
        vec.scatter_reverse(dolfinx.la.InsertMode.add)
    n = V0.dofmap.index_map.size_local
    out = dolfinx.fem.Function(V0)
    out.x.array[:n] = rhs.array[:n] / vol.array[:n]
    out.x.scatter_forward()
    return out


@dataclass(frozen=True)
class IterationRecord:
    k: int
    residual: float
    error: Optional[float]
    inner: Optional[int] = None


@dataclass
class InversionResult:
    step: float
    history: List[IterationRecord] = field(default_factory=list)
    snapshots: Dict[int, dolfinx.fem.Function] = field(default_factory=dict)
    gamma: Optional[dolfinx.fem.Function] = None
    failure: Optional[NonPositiveConductivity] = None

    @property
    def residuals(self) -> np.ndarray:
        return np.array([r.residual for r in self.history])

    @property
    def errors(self) -> np.ndarray:
        return np.array([np.nan if r.error is None else r.error for r in self.history])

    @property
    def inner_iterations(self) -> np.ndarray:
        return np.array([r.inner for r in self.history if r.inner is not None], dtype=int)


class InverseProblem:
    """F_G, residual, error and gradient shared by the iterative methods."""

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        V: dolfinx.fem.FunctionSpace,
        currents: Sequence[ufl.core.expr.Expr],
        ds_g: ufl.Measure,
        data: Sequence[dolfinx.fem.Function],
        config: SolverConfig,
        gamma_true: Optional[dolfinx.fem.Function] = None,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        if len(currents) != len(data):
            raise ValueError("currents and data must have the same length")
        self._mesh = mesh
        self._V = V
        self._currents = list(currents)
        self._ds_g = ds_g
        self._config = config
        self._comm = comm
        self._ds = ufl.Measure("ds", domain=mesh)
        self._data = [self._center(d) for d in data]
        self._gamma_true = gamma_true
        self._norm_true = None if gamma_true is None else np.sqrt(self._integrate(gamma_true ** 2 * ufl.dx))

    def forward(self, gamma: dolfinx.fem.Function) -> List[dolfinx.fem.Function]:
        return MultiForwardSolver(
            self._mesh, self._V, gamma, self._currents, self._ds_g, self._config, self._comm,
        ).solve_all()

    def residual(self, u_list: Sequence[dolfinx.fem.Function]) -> float:
        return float(np.sqrt(sum(self._integrate((u - d) ** 2 * self._ds) for u, d in zip(u_list, self._data))))

    def objective(self, gamma: dolfinx.fem.Function) -> float:
        return 0.5 * self.residual(self.forward(gamma)) ** 2

    def gradient(self, gamma: dolfinx.fem.Function, u_list: Sequence[dolfinx.fem.Function]) -> dolfinx.fem.Function:
        hs = [u - d for u, d in zip(u_list, self._data)]
        adj = MultiAdjointSolver(
            self._mesh, self._V, gamma, u_list, hs, self._ds, self._config, self._comm,
        ).solve()
        return project_dg0(adj, gamma.function_space)

    def error(self, gamma: dolfinx.fem.Function) -> Optional[float]:
        if self._gamma_true is None:
            return None
        return float(np.sqrt(self._integrate((self._gamma_true - gamma) ** 2 * ufl.dx)) / self._norm_true)

    def _center(self, d: dolfinx.fem.Function) -> dolfinx.fem.Function:
        one = dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(1.0))
        mean = self._integrate(d * self._ds) / self._integrate(one * self._ds)
        out = d.copy()
        out.x.array[:] -= mean
        out.x.scatter_forward()
        return out

    def _integrate(self, expr) -> float:
        return self._comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)


class GradientMethod(InverseProblem):
    """gamma_{k+1} = gamma_k - lambda F'_G(gamma_k)* (F_G(gamma_k) - u~)."""

    def run(
        self,
        gamma0: dolfinx.fem.Function,
        step: float,
        n_iter: int,
        snapshot_every: int = 0,
        callback: Optional[Callable[[IterationRecord], None]] = None,
    ) -> InversionResult:
        if step <= 0:
            raise ValueError("step must be positive")
        result = InversionResult(step=step)
        gamma = gamma0.copy()

        for k in range(n_iter + 1):
            u_list = self.forward(gamma)
            record = IterationRecord(k, self.residual(u_list), self.error(gamma))
            result.history.append(record)
            if callback is not None:
                callback(record)
            if snapshot_every and k > 0 and k % snapshot_every == 0:
                result.snapshots[k] = gamma.copy()
            if k == n_iter:
                break

            grad = self.gradient(gamma, u_list)
            gamma.x.array[:] -= step * grad.x.array
            gamma.x.scatter_forward()

            minimum = self._comm.allreduce(float(gamma.x.array.min()), op=MPI.MIN)
            if minimum <= 0:
                result.failure = NonPositiveConductivity(k + 1, minimum)
                break

        result.gamma = gamma
        return result


class InexactNewton(InverseProblem):
    """
    Inexact Newton method with inner gradient (Landweber) iterations.

    Outer step k: b = u~ - F_G(gamma_k). The Newton step s solves F'_G(gamma_k) s ~ b
    approximately: starting from s = 0, s <- s - lambda F'_G(gamma_k)* (F'_G(gamma_k) s - b)
    until ||F'_G(gamma_k) s - b|| <= mu ||b|| or n_inner iterations. Then gamma_{k+1} = gamma_k + s.
    """

    def run(
        self,
        gamma0: dolfinx.fem.Function,
        step: float,
        n_outer: int,
        mu: float = 0.9,
        n_inner: int = 100,
        snapshot_every: int = 0,
        callback: Optional[Callable[[IterationRecord], None]] = None,
    ) -> InversionResult:
        if step <= 0:
            raise ValueError("step must be positive")
        if not 0 < mu < 1:
            raise ValueError("mu must be in (0, 1)")
        if n_inner < 1:
            raise ValueError("n_inner must be at least 1")
        result = InversionResult(step=step)
        gamma = gamma0.copy()

        for k in range(n_outer + 1):
            u_list = self.forward(gamma)
            b = [d.x.array - u.x.array for d, u in zip(self._data, u_list)]
            record = IterationRecord(k, self.residual(u_list), self.error(gamma))
            if snapshot_every and k > 0 and k % snapshot_every == 0:
                result.snapshots[k] = gamma.copy()
            if k == n_outer:
                result.history.append(record)
                if callback is not None:
                    callback(record)
                break

            with LinearizedOperator(self._mesh, self._V, gamma, u_list, self._config, self._comm) as op:
                s, n = self._newton_step(op, b, record.residual, step, mu, n_inner)
            record = replace(record, inner=n)
            result.history.append(record)
            if callback is not None:
                callback(record)

            gamma.x.array[:] += s.x.array
            gamma.x.scatter_forward()
            minimum = self._comm.allreduce(float(gamma.x.array.min()), op=MPI.MIN)
            if minimum <= 0:
                result.failure = NonPositiveConductivity(k + 1, minimum)
                break

        result.gamma = gamma
        return result

    @staticmethod
    def _newton_step(op, b, norm_b, step, mu, n_inner):
        s = dolfinx.fem.Function(op.parameter_space)
        lin_res = [-bj for bj in b]
        n = 0
        while op.norm(lin_res) > mu * norm_b and n < n_inner:
            s.x.array[:] -= step * op.adjoint(lin_res).x.array
            s.x.scatter_forward()
            lin_res = [w.x.array - bj for w, bj in zip(op.apply(s), b)]
            n += 1
        return s, n


class LinearizedOperator:
    """
    F'_G(gamma) and its adjoint for a fixed gamma, with the stiffness matrix,
    the AMG preconditioner and all forms built once and reused for every call.

    apply(s)    -> (omega_1, ..., omega_m),  -div(gamma grad omega_j) = div(s grad u_j)
    adjoint(h)  -> P_DG0( -sum_j grad u_j . grad psi_j ),  psi_j with flux h_j
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        V: dolfinx.fem.FunctionSpace,
        gamma: dolfinx.fem.Function,
        u_list: Sequence[dolfinx.fem.Function],
        config: SolverConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._V = V
        self._V0 = gamma.function_space
        self._comm = comm
        self._u_list = list(u_list)
        ds = ufl.Measure("ds", domain=mesh)
        w, v = ufl.TrialFunction(V), ufl.TestFunction(V)

        self._system = NeumannSystem(
            dolfinx.fem.form(ufl.inner(gamma * ufl.grad(w), ufl.grad(v)) * ufl.dx), V, config, comm,
            owner=type(self).__name__,
        )

        one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
        self._area = self._sum(dolfinx.fem.form(one * ds))
        self._probe = dolfinx.fem.Function(V)
        self._mean_form = dolfinx.fem.form(self._probe * ds)
        self._square_form = dolfinx.fem.form(self._probe ** 2 * ds)

        self._s = dolfinx.fem.Function(self._V0)
        self._derivative_forms = [
            dolfinx.fem.form(-ufl.inner(self._s * ufl.grad(u), ufl.grad(v)) * ufl.dx) for u in self._u_list
        ]

        self._hs = [dolfinx.fem.Function(V) for _ in self._u_list]
        self._h_means = [dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) for _ in self._u_list]
        self._adjoint_forms = [
            dolfinx.fem.form(h * v * ds - c * v * ds) for h, c in zip(self._hs, self._h_means)
        ]

        self._psis = [dolfinx.fem.Function(V) for _ in self._u_list]
        q = ufl.TestFunction(self._V0)
        density = -sum(ufl.inner(ufl.grad(u), ufl.grad(psi)) for u, psi in zip(self._u_list, self._psis))
        self._projection_form = dolfinx.fem.form(density * q * ufl.dx)
        volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(q * ufl.dx))
        volumes.scatter_reverse(dolfinx.la.InsertMode.add)
        self._n0 = self._V0.dofmap.index_map.size_local
        self._volumes = volumes.array[:self._n0].copy()

    @property
    def parameter_space(self) -> dolfinx.fem.FunctionSpace:
        return self._V0

    def apply(self, s: dolfinx.fem.Function) -> List[dolfinx.fem.Function]:
        self._s.x.array[:] = s.x.array
        self._s.x.scatter_forward()
        return [self._solve(form) for form in self._derivative_forms]

    def adjoint(self, hs: Sequence[np.ndarray]) -> dolfinx.fem.Function:
        for h, mean, values in zip(self._hs, self._h_means, hs):
            h.x.array[:] = values
            h.x.scatter_forward()
            mean.value = self._boundary_integral(h) / self._area
        for psi, form in zip(self._psis, self._adjoint_forms):
            psi.x.array[:] = self._solve(form).x.array
            psi.x.scatter_forward()

        rhs = dolfinx.fem.assemble_vector(self._projection_form)
        rhs.scatter_reverse(dolfinx.la.InsertMode.add)
        out = dolfinx.fem.Function(self._V0)
        out.x.array[:self._n0] = rhs.array[:self._n0] / self._volumes
        out.x.scatter_forward()
        return out

    def norm(self, functions: Sequence[np.ndarray]) -> float:
        total = 0.0
        for values in functions:
            self._probe.x.array[:] = values
            self._probe.x.scatter_forward()
            total += self._sum(self._square_form)
        return float(np.sqrt(total))

    def close(self) -> None:
        self._system.close()

    def __enter__(self) -> "LinearizedOperator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _solve(self, form: dolfinx.fem.Form) -> dolfinx.fem.Function:
        u = self._system.solve(form)
        u.x.array[:] -= self._boundary_integral(u) / self._area
        u.x.scatter_forward()
        return u

    def _boundary_integral(self, f: dolfinx.fem.Function) -> float:
        self._probe.x.array[:] = f.x.array
        return self._sum(self._mean_form)

    def _sum(self, form: dolfinx.fem.Form) -> float:
        return self._comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def data_from(
    mesh: dolfinx.mesh.Mesh,
    V: dolfinx.fem.FunctionSpace,
    gamma: dolfinx.fem.Function,
    currents: Sequence[ufl.core.expr.Expr],
    ds_g: ufl.Measure,
    config: SolverConfig,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> List[dolfinx.fem.Function]:
    return MultiForwardSolver(mesh, V, gamma, currents, ds_g, config, comm).solve_all()