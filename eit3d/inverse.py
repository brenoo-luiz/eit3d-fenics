from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import dolfinx
import dolfinx.fem
import dolfinx.la
import numpy as np
import ufl
from mpi4py import MPI

from eit3d.config import SolverConfig
from eit3d.solvers.multi import MultiAdjointSolver, MultiForwardSolver

logger = logging.getLogger(__name__)

TRANSFER_PADDING = 1e-2


class NonPositiveConductivity(RuntimeError):

    def __init__(self, iteration: int, minimum: float) -> None:
        super().__init__(f"conductivity became non-positive at iteration {iteration} (min {minimum:.3e})")
        self.iteration = iteration
        self.minimum   = minimum


def transfer(u_from: dolfinx.fem.Function, V_to: dolfinx.fem.FunctionSpace,
            padding: float = TRANSFER_PADDING) -> dolfinx.fem.Function:
    mesh_to = V_to.mesh
    n_cells = mesh_to.topology.index_map(mesh_to.topology.dim).size_local
    cells   = np.arange(n_cells, dtype=np.int32)
    data    = dolfinx.fem.create_interpolation_data(V_to, u_from.function_space, cells, padding=padding)
    u_to    = dolfinx.fem.Function(V_to)
    u_to.interpolate_nonmatching(u_from, cells, data)
    u_to.x.scatter_forward()
    return u_to


def project_dg0(f: dolfinx.fem.Function, V0: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Function:
    q   = ufl.TestFunction(V0)
    rhs = dolfinx.fem.assemble_vector(dolfinx.fem.form(f * q * ufl.dx))
    vol = dolfinx.fem.assemble_vector(dolfinx.fem.form(q * ufl.dx))
    for vec in (rhs, vol):
        vec.scatter_reverse(dolfinx.la.InsertMode.add)
    n   = V0.dofmap.index_map.size_local
    out = dolfinx.fem.Function(V0)
    out.x.array[:n] = rhs.array[:n] / vol.array[:n]
    out.x.scatter_forward()
    return out


@dataclass(frozen=True)
class IterationRecord:
    k       : int
    residual: float
    error   : Optional[float]


@dataclass
class InversionResult:
    step     : float
    history  : List[IterationRecord]                  = field(default_factory=list)
    snapshots: Dict[int, dolfinx.fem.Function]       = field(default_factory=dict)
    gamma    : Optional[dolfinx.fem.Function]        = None
    failure  : Optional[NonPositiveConductivity]     = None

    @property
    def residuals(self) -> np.ndarray:
        return np.array([r.residual for r in self.history])

    @property
    def errors(self) -> np.ndarray:
        return np.array([np.nan if r.error is None else r.error for r in self.history])


class GradientMethod:

    def __init__(
        self,
        mesh      : dolfinx.mesh.Mesh,
        V         : dolfinx.fem.FunctionSpace,
        currents  : Sequence[ufl.core.expr.Expr],
        ds_g      : ufl.Measure,
        data      : Sequence[dolfinx.fem.Function],
        config    : SolverConfig,
        gamma_true: Optional[dolfinx.fem.Function] = None,
        comm      : MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        if len(currents) != len(data):
            raise ValueError("currents and data must have the same length")
        self._mesh       = mesh
        self._V          = V
        self._currents   = list(currents)
        self._ds_g       = ds_g
        self._config     = config
        self._comm       = comm
        self._ds         = ufl.Measure("ds", domain=mesh)
        self._data       = [self._center(d) for d in data]
        self._gamma_true = gamma_true
        self._norm_true  = None if gamma_true is None else np.sqrt(self._integrate(gamma_true ** 2 * ufl.dx))

    def forward(self, gamma: dolfinx.fem.Function) -> List[dolfinx.fem.Function]:
        return MultiForwardSolver(
            self._mesh, self._V, gamma, self._currents, self._ds_g, self._config, self._comm,
        ).solve_all()

    def residual(self, u_list: Sequence[dolfinx.fem.Function]) -> float:
        return float(np.sqrt(sum(self._integrate((u - d) ** 2 * self._ds) for u, d in zip(u_list, self._data))))

    def objective(self, gamma: dolfinx.fem.Function) -> float:
        return 0.5 * self.residual(self.forward(gamma)) ** 2

    def gradient(self, gamma: dolfinx.fem.Function, u_list: Sequence[dolfinx.fem.Function]) -> dolfinx.fem.Function:
        hs  = [u - d for u, d in zip(u_list, self._data)]
        adj = MultiAdjointSolver(
            self._mesh, self._V, gamma, u_list, hs, self._ds, self._config, self._comm,
        ).solve()
        return project_dg0(adj, gamma.function_space)

    def error(self, gamma: dolfinx.fem.Function) -> Optional[float]:
        if self._gamma_true is None:
            return None
        return float(np.sqrt(self._integrate((self._gamma_true - gamma) ** 2 * ufl.dx)) / self._norm_true)

    def run(
        self,
        gamma0        : dolfinx.fem.Function,
        step          : float,
        n_iter        : int,
        snapshot_every: int = 0,
        callback      : Optional[Callable[[IterationRecord], None]] = None,
    ) -> InversionResult:
        if step <= 0:
            raise ValueError("step must be positive")
        result = InversionResult(step=step)
        gamma  = gamma0.copy()

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

    def _center(self, d: dolfinx.fem.Function) -> dolfinx.fem.Function:
        one  = dolfinx.fem.Constant(self._mesh, dolfinx.default_scalar_type(1.0))
        mean = self._integrate(d * self._ds) / self._integrate(one * self._ds)
        out  = d.copy()
        out.x.array[:] -= mean
        out.x.scatter_forward()
        return out

    def _integrate(self, expr) -> float:
        return self._comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)


def data_from(
    mesh    : dolfinx.mesh.Mesh,
    V       : dolfinx.fem.FunctionSpace,
    gamma   : dolfinx.fem.Function,
    currents: Sequence[ufl.core.expr.Expr],
    ds_g    : ufl.Measure,
    config  : SolverConfig,
    comm    : MPI.Comm = MPI.COMM_WORLD,
) -> List[dolfinx.fem.Function]:
    return MultiForwardSolver(mesh, V, gamma, currents, ds_g, config, comm).solve_all()