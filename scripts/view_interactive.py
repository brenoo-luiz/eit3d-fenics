"""
scripts/view_interactive.py
============================
Interactive 3-panel PyVista viewer for the EIT 3D forward solution.
Loads mesh from cache and solves fresh — no DOF inconsistency.

Run with:
    QT_QPA_PLATFORM=wayland python scripts/view_interactive.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dolfinx.plot
import pyvista

from eit3d import EITConfig, EITPipeline
from eit3d.visualization.interactive import InteractiveRenderer

logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg  = EITConfig()
pipe = EITPipeline(cfg)
print(pipe.status())

# ── Solve ─────────────────────────────────────────────────────────────────────
gamma = pipe.build_gamma()
u_h   = pipe.solve_forward(pattern=0, gamma=gamma)
V     = pipe.get_function_space()

print(f"u in [{u_h.x.array.min():.4f}, {u_h.x.array.max():.4f}]")

# ── Build PyVista grid ────────────────────────────────────────────────────────
topo, ct, geo = dolfinx.plot.vtk_mesh(V)
grid          = pyvista.UnstructuredGrid(topo, ct, geo)
grid["u"]     = u_h.x.array.real

# ── Open interactive window ───────────────────────────────────────────────────
renderer = InteractiveRenderer()
renderer.render(
    grid               = grid,
    scalar             = "u",
    conductivity_config = cfg.conductivity,
    eta_config         = cfg.eta,
)
