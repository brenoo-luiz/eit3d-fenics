import logging
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import pyvista
import ufl
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline, report
from eit3d.config import OUTPUTS_DIR
from eit3d.solvers.neumann import NeumannSolver
from eit3d.visualization.static import StaticRenderer, SurfacePanel

TOL = 1e-6


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg = EITConfig()
    pipe = EITPipeline(cfg)

    report.title("TESTE COM SOLUÇÃO EXATA  u = x² - y²")
    report.mesh(pipe)
    report.info("Condutividade γ", "1 em todo o cilindro")
    report.info("Corrente g", "derivada normal da solução exata")

    mesh, _ = pipe.get_mesh()
    V = pipe.get_function_space()
    ds = ufl.Measure("ds", domain=mesh)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    one = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))

    def integrate(expr) -> float:
        return comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)

    u_exact_ufl = x[0]**2 - x[1]**2
    g = ufl.dot(ufl.grad(u_exact_ufl), n)
    c_val = integrate(u_exact_ufl * ds) / integrate(one * ds)

    u_exact = dolfinx.fem.Function(V)
    u_exact.interpolate(lambda xp: xp[0]**2 - xp[1]**2 - c_val)
    u_exact.x.scatter_forward()

    u_h = NeumannSolver(mesh=mesh, V=V, gamma=one, g=g, config=cfg.solver, comm=comm).solve()
    diff = u_h - u_exact

    err_L2 = np.sqrt(integrate(diff**2 * ufl.dx)) / np.sqrt(integrate(u_exact**2 * ufl.dx))
    err_H1 = np.sqrt(integrate(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)) \
        / np.sqrt(integrate(ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact)) * ufl.dx))
    err_flux = np.sqrt(integrate((ufl.dot(ufl.grad(u_h), n) - g)**2 * ds)) / np.sqrt(integrate(g**2 * ds))

    u_err = dolfinx.fem.Function(V)
    u_err.x.array[:] = np.abs(u_h.x.array - u_exact.x.array)
    err_max = float(u_err.x.array.max())

    report.section("Diferença entre a solução numérica e a exata")
    ok = all([
        report.check("Erro relativo em L2", err_L2, err_L2 < TOL),
        report.check("Erro relativo no gradiente (H1)", err_H1, err_H1 < TOL),
        report.check("Erro relativo do fluxo na borda", err_flux, err_flux < TOL),
        report.check("Maior erro ponto a ponto", err_max, err_max < TOL),
    ])

    topo, ct, geo = dolfinx.plot.vtk_mesh(V)

    def make_grid(name, values):
        grid = pyvista.UnstructuredGrid(topo, ct, geo)
        grid[name] = values
        return grid

    clim_u = (float(min(u_exact.x.array.min(), u_h.x.array.min())),
                float(max(u_exact.x.array.max(), u_h.x.array.max())))

    out = StaticRenderer(cfg, OUTPUTS_DIR).render_surfaces(
        [
            SurfacePanel(make_grid("u_exact", u_exact.x.array.real), "u_exact",
                        "Exact solution  u = x^2 - y^2 - c", clim=clim_u),
            SurfacePanel(make_grid("u_h", u_h.x.array.real), "u_h",
                        "Numerical solution  u_h", clim=clim_u),
            SurfacePanel(make_grid("error", u_err.x.array.real), "error",
                        "Pointwise error  |u_h - u_exact|", cmap="hot", clim=(0.0, err_max)),
        ],
        filename="consistency_test.png",
    )

    report.result(ok, "solução numérica igual à exata", "solução numérica diferente da exata")
    report.files([out])


if __name__ == "__main__":
    main()