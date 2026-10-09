import logging
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import pyvista
import ufl
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline, report
from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.static import StaticRenderer


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = EITConfig()
    pipe = EITPipeline(cfg)
    gam = cfg.conductivity
    g_top, g_bot = cfg.current.patterns[0]

    report.title("PROBLEMA DIRETO  (encontrar o potencial u)")
    report.mesh(pipe)
    report.info("Condutividade γ", f"{gam.gamma_in:g} na esfera (r = {gam.radius:g}), {gam.gamma_out:g} fora")
    report.info("Corrente g", f"{g_top:+g} no topo, {g_bot:+g} na base, 0 na lateral")

    gamma = pipe.build_gamma()
    u_h = pipe.solve_forward(pattern=0, gamma=gamma)

    mesh, _ = pipe.get_mesh()
    ds = ufl.Measure("ds", domain=mesh)
    int_u = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_h * ds)), op=MPI.SUM)

    report.section("Solução")
    report.info("Potencial u", f"de {u_h.x.array.min():.3f} a {u_h.x.array.max():.3f}")
    ok = report.check_zero("∫u na fronteira", int_u)

    V = pipe.get_function_space()
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topo, ct, geo)
    grid["u"] = u_h.x.array.real
    out = StaticRenderer(cfg, OUTPUTS_DIR).render_forward_overview(grid, "u")

    report.result(ok, "problema direto resolvido", "normalização de u falhou")
    report.files([out])


if __name__ == "__main__":
    main()