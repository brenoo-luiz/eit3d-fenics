from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import dolfinx
import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI

from eit3d.config import EITConfig, MeshConfig
from eit3d.currents import cosine_currents, lateral_measure
from eit3d.fields import SpheresField
from eit3d.inverse import data_from, transfer
from eit3d.pipeline.eit_pipeline import EITPipeline

Point = Tuple[float, float, float]


@dataclass(frozen=True)
class TwoSpheres:
    centers: Tuple[Point, ...] = ((0.4, 0.0, 0.4), (-0.4, 0.0, -0.4))
    radius: float = 0.25
    value: float = 10.0
    background: float = 1.0
    ks: Tuple[int, ...] = tuple(range(1, 9))
    fine_mesh: MeshConfig = MeshConfig(size_max=0.05, size_min=0.02)
    coarse_mesh: MeshConfig = MeshConfig(size_max=0.1, size_min=0.05)

    @property
    def spheres(self) -> List[Tuple[Point, float]]:
        return [(c, self.radius) for c in self.centers]

    def conductivity(self, mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
        return SpheresField(mesh, self.centers, self.radius, self.value, self.background, name="gamma+").build()

    def initial_guess(self, mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
        return SpheresField(mesh, (), self.radius, self.background, self.background, name="gamma0").build()

    def currents(self, mesh: dolfinx.mesh.Mesh) -> List[ufl.core.expr.Expr]:
        return cosine_currents(mesh, self.ks)

    def fine_pipeline(self) -> EITPipeline:
        return EITPipeline(EITConfig(mesh=self.fine_mesh))

    def coarse_pipeline(self) -> EITPipeline:
        return EITPipeline(EITConfig(mesh=self.coarse_mesh))

    def data(self, pipe: EITPipeline, comm: MPI.Comm = MPI.COMM_WORLD) -> List[dolfinx.fem.Function]:
        mesh, facet_tags = pipe.get_mesh()
        return data_from(
            mesh, pipe.get_function_space(), self.conductivity(mesh), self.currents(mesh),
            lateral_measure(mesh, facet_tags), pipe.config.solver, comm,
        )


def transfer_all(functions: Sequence[dolfinx.fem.Function], V: dolfinx.fem.FunctionSpace) -> List[dolfinx.fem.Function]:
    return [transfer(f, V) for f in functions]


def boundary_norm(functions: Sequence[dolfinx.fem.Function], comm: MPI.Comm = MPI.COMM_WORLD) -> float:
    total = 0.0
    for f in functions:
        ds = ufl.Measure("ds", domain=f.function_space.mesh)
        total += comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(f ** 2 * ds)), op=MPI.SUM)
    return float(np.sqrt(total))