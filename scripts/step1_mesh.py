"""
Generate and cache the cylinder mesh.
"""

import logging
import sys
from pathlib import Path

import dolfinx.plot
import pyvista

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline, report
from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.static import StaticRenderer


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = EITConfig()
    pipe = EITPipeline(cfg)

    report.title("MALHA DO CILINDRO")
    report.info("Cilindro", f"raio {cfg.mesh.radius:g}, altura {cfg.mesh.height:g}")
    report.info("Tamanho dos elementos", f"entre {cfg.mesh.size_min:g} e {cfg.mesh.size_max:g}")
    report.mesh(pipe)

    mesh, _ = pipe.get_mesh()
    topo, ct, geo = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim - 1)
    out = StaticRenderer(cfg, OUTPUTS_DIR).render_mesh(pyvista.UnstructuredGrid(topo, ct, geo))

    report.files([out])


if __name__ == "__main__":
    main()