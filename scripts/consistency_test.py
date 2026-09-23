import logging
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import ufl
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent))

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.static import StaticRenderer, SurfacePanel
from eit3d.solvers.neumann import NeumannSolver


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    comm = MPI.COMM_WORLD
    cfg  = EITConfig()
    pipe = EITPipeline(cfg)
    print(pipe.status())

    mesh, facet_tags = pipe.get_mesh()
    V                = pipe.get_function_space()

    gamma  = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    ds_all = ufl.Measure("ds", domain=mesh)
    x      = ufl.SpatialCoordinate(mesh)

    u_exact_ufl = x[0]**2 - x[1]**2
    n           = ufl.FacetNormal(mesh)
    g           = ufl.dot(ufl.grad(u_exact_ufl), n)

    # subtract c from u_exact before solving
    integral_u = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_exact_ufl * ds_all)), op=MPI.SUM
    )
    area = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0)) * ds_all)), op=MPI.SUM
    )
    c_val         = integral_u / area
    c_const       = dolfinx.fem.Constant(mesh, PETSc.ScalarType(c_val))
    u_exact_c_ufl = u_exact_ufl - c_const

    print(f"c = {c_val:.5e}  ->  int(u-c) ds = "
          f"{comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_exact_c_ufl * ds_all)), op=MPI.SUM):.2e}")

    u_exact_fn = dolfinx.fem.Function(V)
    u_exact_fn.interpolate(lambda xp: xp[0]**2 - xp[1]**2 - c_val)
    u_exact_fn.x.scatter_forward()

    # solve forward problem (the solver enforces int u_h ds = 0,
    # the same normalization used for u_exact above)
    u_h = NeumannSolver(
        mesh=mesh, V=V, gamma=gamma, g=g,
        config=cfg.solver, comm=comm,
    ).solve()

    # compare u_h with u_exact
    diff    = u_h - u_exact_fn
    norm_L2 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(u_exact_fn, u_exact_fn) * ufl.dx)), op=MPI.SUM))
    norm_H1 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(ufl.grad(u_exact_fn), ufl.grad(u_exact_fn)) * ufl.dx)), op=MPI.SUM))
    err_L2 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
    err_H1 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM))

    u_err = dolfinx.fem.Function(V)
    u_err.x.array[:] = np.abs(u_h.x.array - u_exact_fn.x.array)
    err_max = float(u_err.x.array.max())

    # flux error
    flux_num  = ufl.dot(ufl.grad(u_h), n)
    norm_flux = comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(g**2 * ds_all)), op=MPI.SUM)
    err_flux = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((flux_num - g)**2 * ds_all)), op=MPI.SUM))

    # Render
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)

    def make_grid(name, values):
        grid = pyvista.UnstructuredGrid(topo, ct, geo)
        grid[name] = values
        return grid

    clim_u = (float(min(u_exact_fn.x.array.min(), u_h.x.array.min())),
                float(max(u_exact_fn.x.array.max(), u_h.x.array.max())))

    out = StaticRenderer(cfg, OUTPUTS_DIR).render_surfaces(
        [
            SurfacePanel(make_grid("u_exact", u_exact_fn.x.array.real), "u_exact",
                        "Exact solution  u = x^2 - y^2 - c", clim=clim_u),
            SurfacePanel(make_grid("u_h", u_h.x.array.real), "u_h",
                        "Numerical solution  u_h", clim=clim_u),
            SurfacePanel(make_grid("error", u_err.x.array.real), "error",
                        "Pointwise error  |u_h - u_exact|", cmap="hot",
                        clim=(0.0, err_max)),
        ],
        filename="consistency_test.png",
    )

    # Results
    passed = (err_L2/norm_L2 < 1e-2 and err_flux/np.sqrt(norm_flux) < 5e-2)

    print("\nResults:")
    print(f"  exact solution:  u = x^2 - y^2 - c  (c={c_val:.2e})")
    print(f"  L2 error:        {err_L2/norm_L2*100:.4f}%")
    print(f"  H1 error:        {err_H1/norm_H1*100:.4f}%")
    print(f"  max error:       {err_max:.2e}")
    print(f"  flux error:      {err_flux/np.sqrt(norm_flux)*100:.4f}%")
    print(f"  result:          {'passed' if passed else 'failed'}")
    print(f"  saved:           {out}")


if __name__ == "__main__":
    main()