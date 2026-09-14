"""
Consistency test using manufactured solution (7 steps from advisor).

Exact solution: u(x,y,z) = x^2 - y^2
gamma = 1, g = dot(grad(u_exact), n) over full boundary.
Constant c subtracted from u_exact BEFORE solving (Marcelo's approach).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx
import dolfinx.fem.petsc
import dolfinx.plot
import ufl
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from mpi4py import MPI
from petsc4py import PETSc

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR

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

# solve forward problem
u_t = ufl.TrialFunction(V)
v_t = ufl.TestFunction(V)
a   = ufl.inner(gamma * ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
L   = g * v_t * ds_all

A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
A.assemble()

ns_vec = A.createVecLeft()
ns_vec.set(1.0); ns_vec.normalize()
ns = PETSc.NullSpace().create(vectors=[ns_vec], comm=comm)
A.setNullSpace(ns); A.setTransposeNullSpace(ns)

b = A.createVecRight()
with b.localForm() as lb: lb.set(0.0)
dolfinx.fem.petsc.assemble_vector(b, dolfinx.fem.form(L))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
ns.remove(b)

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.HYPRE)
ksp.setTolerances(rtol=cfg.solver.rtol, atol=cfg.solver.atol, max_it=cfg.solver.max_it)
ksp.setFromOptions()

u_h = dolfinx.fem.Function(V)
ksp.solve(b, u_h.x.petsc_vec)
u_h.x.scatter_forward()
print(f"converged in {ksp.getIterationNumber()} iterations")
A.destroy(); b.destroy(); ns_vec.destroy(); ksp.destroy()

# compare u_h with u_exact
diff    = u_h - u_exact_fn
norm_L2 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(u_exact_fn, u_exact_fn) * ufl.dx)), op=MPI.SUM))
norm_H1 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(ufl.grad(u_exact_fn), ufl.grad(u_exact_fn)) * ufl.dx)), op=MPI.SUM))
erro_L2 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM))
erro_H1 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM))

u_err = dolfinx.fem.Function(V)
u_err.x.array[:] = np.abs(u_h.x.array - u_exact_fn.x.array)
err_max = float(u_err.x.array.max())

# flux error
flux_num  = ufl.dot(ufl.grad(u_h), n)
norm_flux = comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(g**2 * ds_all)), op=MPI.SUM)
erro_flux = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(
    dolfinx.fem.form((flux_num - g)**2 * ds_all)), op=MPI.SUM))

# Render
BG    = "#1e1e2e"
topo, ct, geo = dolfinx.plot.vtk_mesh(V)

grids  = {"u_exact": u_exact_fn.x.array.real,
            "u_h"    : u_h.x.array.real,
            "erro"   : u_err.x.array.real}
clim_u   = [float(min(u_exact_fn.x.array.min(), u_h.x.array.min())),
            float(max(u_exact_fn.x.array.max(), u_h.x.array.max()))]
clim_err = [0.0, float(u_err.x.array.max())]
cmaps    = {"u_exact": "turbo", "u_h": "turbo", "erro": "hot"}
clims    = {"u_exact": clim_u,  "u_h": clim_u,  "erro": clim_err}
tmps     = []

for name, values in grids.items():
    g_pv = pyvista.UnstructuredGrid(topo, ct, geo)
    g_pv[name] = values
    surf = g_pv.extract_surface(algorithm="dataset_surface")
    tmp  = OUTPUTS_DIR / f"_tmp_{name}.png"
    p    = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
    p.add_mesh(surf, scalars=name, cmap=cmaps[name], clim=clims[name],
                show_edges=False, lighting=True, smooth_shading=True,
                show_scalar_bar=True)
    p.set_background(BG); p.view_isometric()
    p.screenshot(str(tmp)); p.close()
    tmps.append(tmp)

imgs   = [np.array(Image.open(t)) for t in tmps]
titles = ["Exact solution  u = x^2 - y^2 - c",
            "Numerical solution  u_h",
            "Pointwise error  |u_h - u_exact|"]

fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, hspace=0.01, wspace=0.03,
                        left=0.02, right=0.98, top=0.93, bottom=0.01)
for i, (img, title) in enumerate(zip(imgs, titles)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img); ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=14, pad=8)

out = OUTPUTS_DIR / "consistency_test.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
for t in tmps: t.unlink()

# Results
passed = (erro_L2/norm_L2 < 1e-2 and erro_flux/np.sqrt(norm_flux) < 5e-2)

print("\nResults:")
print(f"  exact solution:  u = x^2 - y^2 - c  (c={c_val:.2e})")
print(f"  L2 error:        {erro_L2/norm_L2*100:.4f}%")
print(f"  H1 error:        {erro_H1/norm_H1*100:.4f}%")
print(f"  max error:       {err_max:.2e}")
print(f"  flux error:      {erro_flux/np.sqrt(norm_flux)*100:.4f}%")
print(f"  result:          {'passed' if passed else 'failed'}")
print(f"  saved:           {out}")