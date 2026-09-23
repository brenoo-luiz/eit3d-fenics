import logging
import sys
from pathlib import Path

import dolfinx.plot
import pyvista

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline
from eit3d.visualization.interactive import InteractiveRenderer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg  = EITConfig()
    pipe = EITPipeline(cfg)
    print(pipe.status())

    # Solve
    gamma = pipe.build_gamma()
    u_h   = pipe.solve_forward(pattern=0, gamma=gamma)
    V     = pipe.get_function_space()

    print(f"u in [{u_h.x.array.min():.4f}, {u_h.x.array.max():.4f}]")

    # Build PyVista grid
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    grid          = pyvista.UnstructuredGrid(topo, ct, geo)
    grid["u"]     = u_h.x.array.real

    # Open interactive window
    InteractiveRenderer(cfg).show(grid, "u")


if __name__ == "__main__":
    main()