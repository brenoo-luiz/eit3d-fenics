"""
Generate and cache the cylinder mesh.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx.plot
import pyvista

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.static import StaticRenderer

logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg  = EITConfig()
pipe = EITPipeline(cfg)

print(pipe.status())

mesh, _ = pipe.get_mesh()
n_cells = mesh.topology.index_map(3).size_global
n_verts = mesh.topology.index_map(0).size_global
print(f"Cells: {n_cells}  Vertices: {n_verts}")

# Visualize mesh surface
topo, ct, geo = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim - 1)
StaticRenderer(cfg, OUTPUTS_DIR).render_mesh(pyvista.UnstructuredGrid(topo, ct, geo))