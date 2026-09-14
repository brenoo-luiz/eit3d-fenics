"""
Generate and cache the cylinder mesh.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyvista
import dolfinx.plot
import numpy as np
import matplotlib.pyplot as plt

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR

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
grid = pyvista.UnstructuredGrid(topo, ct, geo)

p = pyvista.Plotter(off_screen=True, window_size=(900, 900))
p.add_mesh(grid, color="lightblue", show_edges=True, edge_color="#2a2a4a", line_width=0.3)
p.set_background("#1e1e2e")
p.view_isometric()
p.camera.zoom(1.1)

out = OUTPUTS_DIR / "step1_mesh.png"
p.screenshot(str(out))
p.close()
print(f"Saved: {out}")
