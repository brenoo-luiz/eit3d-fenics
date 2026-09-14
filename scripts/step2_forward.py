"""
Solve and visualize the EIT 3D forward problem.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx.plot
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from eit3d import EITConfig, EITPipeline
from eit3d.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg  = EITConfig()
pipe = EITPipeline(cfg)
print(pipe.status())

# Solve 
gamma = pipe.build_gamma()
u_h   = pipe.solve_forward(pattern=0, gamma=gamma)

u_min = float(u_h.x.array.min())
u_max = float(u_h.x.array.max())
clim  = [u_min, u_max]
print(f"u in [{u_min:.4f}, {u_max:.4f}]")

# PyVista grids
V = pipe.get_function_space()
topo, ct, geo = dolfinx.plot.vtk_mesh(V)

grid = pyvista.UnstructuredGrid(topo, ct, geo)
grid["u"] = u_h.x.array.real

surf  = grid.extract_surface(algorithm="dataset_surface")
clip  = grid.clip(normal="x", origin=(0, 0, 0))

theta      = np.linspace(0, 2 * np.pi, 200)
circle_pts = np.column_stack([
    np.zeros(200),
    cfg.conductivity.radius * np.cos(theta),
    cfg.conductivity.radius * np.sin(theta),
])
circle = pyvista.Spline(circle_pts, 200)

sphere = pyvista.Sphere(
    radius=cfg.conductivity.radius,
    center=cfg.conductivity.center.tolist(),
    theta_resolution=60, phi_resolution=60,
)
cyl = pyvista.Cylinder(
    center=(0, 0, 0), direction=(0, 0, 1),
    radius=1.0, height=2.0, resolution=100, capping=True,
).extract_surface()

BG    = "#1e1e2e"
sargs = dict(title="u", title_font_size=20, label_font_size=15,
            color="white", position_x=0.03, position_y=0.03,
            width=0.42, height=0.06)

# Renders
tmps = []

for name, setup in [
    ("_tmp_gamma.png", lambda p: [
        p.add_mesh(cyl, color="#1a3a6a", opacity=0.45, lighting=True, smooth_shading=True),
        p.add_mesh(sphere, color="#cc3333", opacity=0.95, lighting=True, smooth_shading=True),
        p.add_text("γ=2 (sphere) / γ=1 (background)", position="lower_left", font_size=12, color="white"),
        p.view_isometric(), p.camera.zoom(1.1),
    ]),
    ("_tmp_u.png", lambda p: [
        p.add_mesh(surf, scalars="u", cmap="turbo", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=True, scalar_bar_args=sargs),
        p.view_isometric(),
    ]),
    ("_tmp_clip.png", lambda p: [
        p.add_mesh(clip, scalars="u", cmap="turbo", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=False),
        p.add_mesh(circle, color="white", line_width=3),
        setattr(p, "camera_position", [(5, 0, 0), (0, 0, 0), (0, 0, 1)]),
        p.camera.zoom(1.4),
    ]),
]:
    path = OUTPUTS_DIR / name
    p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
    setup(p)
    p.set_background(BG)
    p.screenshot(str(path))
    p.close()
    tmps.append(path)

# Compose
imgs   = [np.array(Image.open(t)) for t in tmps]
titles = [
    "Conductivity γ (sphere inclusion)",
    "Solution u — cylinder surface",
    "Cross-section at x=0",
]

fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, hspace=0.01, wspace=0.03,
                        left=0.02, right=0.98, top=0.93, bottom=0.01)
for i, (img, title) in enumerate(zip(imgs, titles)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img); ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=14, pad=8)

out = OUTPUTS_DIR / "step2_forward.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

for t in tmps:
    t.unlink()

print(f"Saved: {out}")
