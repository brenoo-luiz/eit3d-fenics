from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import pyvista

TRUE_FIELD = "true"


def cell_values(function: dolfinx.fem.Function) -> np.ndarray:
    space = function.function_space
    mesh = space.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    dofmap = space.dofmap.list
    if dofmap.shape[1] != 1 or space.dofmap.index_map_bs != 1:
        raise ValueError("conductivity must be a scalar DG0 function")
    return np.asarray(function.x.array.real[dofmap[:n_cells, 0]])


def conductivity_grid(
    fields: Mapping[str, dolfinx.fem.Function],
) -> pyvista.UnstructuredGrid:
    mesh = next(iter(fields.values())).function_space.mesh
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    for name, function in fields.items():
        if function.function_space.mesh is not mesh:
            raise ValueError(f"field {name!r} is on a different mesh")
        grid.cell_data[name] = cell_values(function)
    return grid


def save_conductivity(fields: Mapping[str, dolfinx.fem.Function], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conductivity_grid(fields).save(path)
    return path


def iteration_fields(grid: pyvista.DataSet) -> list[str]:
    return [name for name in grid.cell_data if name != TRUE_FIELD]
