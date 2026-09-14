"""
Main orchestrator for the EIT 3D project.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import basix
import dolfinx
import dolfinx.fem
import ufl
from mpi4py import MPI

from eit3d.config import CACHE_DIR, OUTPUTS_DIR, MESH_FILE, EITConfig
from eit3d.fields.conductivity import ConductivityField, DirectionalField
from eit3d.mesh.cylinder import CylinderMesh
from eit3d.solvers.derivative import DerivativeSolver
from eit3d.solvers.forward import ForwardSolver

logger = logging.getLogger(__name__)


class EITPipeline:
    """
    Main orchestrator for the EIT 3D project.

    Cache strategy:
        Mesh     -> saved to disk (~90s generation)
        Gamma/Eta -> always recomputed (~1s)
        Solution  -> always solved    (~10s)
    """

    def __init__(
        self,
        config    : EITConfig,
        comm      : MPI.Comm = MPI.COMM_WORLD,
        force_mesh: bool     = False,
    ) -> None:
        self._config = config
        self._comm   = comm

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        self._mesh       : Optional[dolfinx.mesh.Mesh]         = None
        self._facet_tags : Optional[dolfinx.mesh.MeshTags]     = None
        self._V          : Optional[dolfinx.fem.FunctionSpace]  = None
        self._force_mesh = force_mesh

    def get_mesh(self) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        if self._mesh is None:
            cylinder = CylinderMesh(
                config=self._config.mesh,
                mesh_file=MESH_FILE,
                comm=self._comm,
                force=self._force_mesh,
            )
            self._mesh, self._facet_tags = cylinder.get()
        return self._mesh, self._facet_tags

    def get_function_space(self) -> dolfinx.fem.FunctionSpace:
        if self._V is None:
            mesh, _ = self.get_mesh()
            el      = basix.ufl.element("Lagrange", "tetrahedron", degree=2, shape=())
            self._V = dolfinx.fem.functionspace(mesh, el)
            logger.info("P2 space: %d DOFs", self._V.dofmap.index_map.size_global)
        return self._V

    def build_gamma(self) -> dolfinx.fem.Function:
        mesh, _ = self.get_mesh()
        return ConductivityField(mesh, self._config.conductivity).build()

    def build_eta(self) -> dolfinx.fem.Function:
        mesh, _ = self.get_mesh()
        return DirectionalField(mesh, self._config.eta).build()

    def solve_forward(
        self,
        pattern: int                            = 0,
        gamma  : Optional[dolfinx.fem.Function] = None,
    ) -> dolfinx.fem.Function:
        if gamma is None:
            gamma = self.build_gamma()

        mesh, facet_tags = self.get_mesh()
        V                = self.get_function_space()
        g_top, g_bot     = self._config.current.patterns[pattern]

        return ForwardSolver(
            mesh=mesh, facet_tags=facet_tags,
            V=V, gamma=gamma,
            config=self._config.solver, comm=self._comm,
        ).solve(g_top=g_top, g_bot=g_bot)

    def solve_derivative(
        self,
        u_gamma: dolfinx.fem.Function,
        gamma  : Optional[dolfinx.fem.Function] = None,
        eta    : Optional[dolfinx.fem.Function] = None,
    ) -> dolfinx.fem.Function:
        if gamma is None:
            gamma = self.build_gamma()
        if eta is None:
            eta = self.build_eta()

        self.get_mesh()
        V = self.get_function_space()

        return DerivativeSolver(
            mesh=self._mesh, V=V,
            gamma=gamma, eta=eta, u_gamma=u_gamma,
            config=self._config.solver, comm=self._comm,
        ).solve()

    def status(self) -> str:
        lines = [
            self._config.summary(),
            f"mesh cache:   {'available' if MESH_FILE.exists() else 'not generated'}",
        ]
        return "\n".join(lines)