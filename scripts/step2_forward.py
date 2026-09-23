"""
Solve and visualize the EIT 3D forward problem.
"""

import logging
import sys
from pathlib import Path

import dolfinx.plot
import pyvista

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.static import StaticRenderer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg  = EITConfig()
    pipe = EITPipeline(cfg)
    print(pipe.status())

    # Solve 
    gamma = pipe.build_gamma()
    u_h   = pipe.solve_forward(pattern=0, gamma=gamma)

    print(f"u in [{float(u_h.x.array.min()):.4f}, {float(u_h.x.array.max()):.4f}]")

    # Render
    V = pipe.get_function_space()
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topo, ct, geo)
    grid["u"] = u_h.x.array.real

    StaticRenderer(cfg, OUTPUTS_DIR).render_forward_overview(grid, "u")


if __name__ == "__main__":
    main()