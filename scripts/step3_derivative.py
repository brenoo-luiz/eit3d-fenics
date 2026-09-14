"""scripts/step3_derivative.py — directional derivative F'(gamma)eta and consistency test."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx.plot
import dolfinx.fem.petsc
import ufl
import numpy as np
import pyvista
from mpi4py import MPI
from petsc4py import PETSc

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR, ConsistencyTestConfig
from eit3d.solvers.forward import ForwardSolver
from eit3d.visualization.static import StaticRenderer

logging.basicConfig(level=logging.INFO, format="%(message)s")

comm = MPI.COMM_WORLD
cfg  = EITConfig(consistency=ConsistencyTestConfig(n_iter=150, base=0.9))
pipe = EITPipeline(cfg)
print(pipe.status())

gamma = pipe.build_gamma()
eta   = pipe.build_eta()

u_gamma = pipe.solve_forward(pattern=0, gamma=gamma)
omega   = pipe.solve_derivative(u_gamma=u_gamma, gamma=gamma, eta=eta)

mesh, facet_tags = pipe.get_mesh()
ds_all = ufl.Measure("ds", domain=mesh)

# Ensure int(omega) ds = 0
integral_omega_before = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(omega * ds_all)), op=MPI.SUM
)
area_total = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0)) * ds_all)), op=MPI.SUM
)
c_omega = integral_omega_before / area_total
omega.x.array[:] -= c_omega
omega.x.scatter_forward()

integral_omega = comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(omega * ds_all)), op=MPI.SUM
)
norm_omega_bnd = np.sqrt(comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(omega**2 * ds_all)), op=MPI.SUM
))

print(f"int(omega): {integral_omega_before:.2e} -> {integral_omega:.2e}  (c={c_omega:.2e})")
print(f"||omega||:  {norm_omega_bnd:.4e}")

# Consistency test
print(f"\nconsistency test ({cfg.consistency.n_iter} iterations)")

V0      = dolfinx.fem.functionspace(mesh, ("DG", 0))
V       = pipe.get_function_space()
gamma_n = dolfinx.fem.Function(V0)
diff_fn = dolfinx.fem.Function(V)
t_vals  = cfg.consistency.t_values()
y_vals  = []

for k, t_n in enumerate(t_vals):
    gamma_n.x.array[:] = gamma.x.array + t_n * eta.x.array
    gamma_n.x.scatter_forward()

    u_n = ForwardSolver(
        mesh=mesh, facet_tags=facet_tags, V=V,
        gamma=gamma_n, config=cfg.solver, comm=comm,
    ).solve(
        g_top=cfg.current.patterns[0][0],
        g_bot=cfg.current.patterns[0][1],
    )

    diff_fn.x.array[:] = (u_n.x.array - u_gamma.x.array) / t_n
    diff_fn.x.scatter_forward()

    err = diff_fn - omega
    num = np.sqrt(comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(err**2 * ds_all)), op=MPI.SUM
    ))
    y_n = num / norm_omega_bnd
    y_vals.append(y_n)
    print(f"  n={k:3d}  t={t_n:.6f}  y={y_n:.4e}")

y_vals  = np.array(y_vals)
idx_min = int(np.argmin(y_vals))
log_t   = np.log10(t_vals)
log_y   = np.log10(y_vals)

# Fit only on the initial linear descent (avoids plateau distorting the slope)
n_fit    = min(30, idx_min)
coeffs   = np.polyfit(log_t[:n_fit], log_y[:n_fit], 1)
taxa     = coeffs[0]
fit_line = np.polyval(coeffs, log_t[:n_fit])

# Render
print("\nrendering...")
renderer = StaticRenderer(OUTPUTS_DIR)

renderer.render_geometry(
    sphere_center=cfg.conductivity.center,
    sphere_radius=cfg.conductivity.radius,
    eta_centers=cfg.eta.centers,
    eta_radius=cfg.eta.radius,
)

topo, ct, geo = dolfinx.plot.vtk_mesh(V)

grid_ug            = pyvista.UnstructuredGrid(topo, ct, geo)
grid_ug["u_gamma"] = u_gamma.x.array.real
renderer.render_forward(grid_ug, "u_gamma", cfg.conductivity.radius)

grid_om          = pyvista.UnstructuredGrid(topo, ct, geo)
grid_om["omega"] = omega.x.array.real
renderer.render_omega(grid_om, "omega", integral_omega)

renderer.render_consistency(y_vals, t_vals, taxa, fit_line, idx_min)

# Results
print("\nresults:")
print(f"  u_gamma:       [{u_gamma.x.array.min():.4f}, {u_gamma.x.array.max():.4f}]")
print(f"  omega:         [{omega.x.array.min():.4f}, {omega.x.array.max():.4f}]")
print(f"  int(omega):    {integral_omega:.2e}")
print(f"  y_min:         {y_vals[idx_min]:.4e}  (n={idx_min})")
print(f"  log-log slope: {taxa:.3f}  (expected: 1.0)")