"""
Interactive PyVista renderer — 3-panel linked window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dolfinx.plot
import numpy as np
import pyvista

from eit3d.config import OUTPUTS_DIR, ConductivityConfig, EtaConfig
from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)


class InteractiveRenderer(BaseRenderer):
    """
    Opens an interactive PyVista window with 3 linked panels:
        Panel 0 — Conductivity gamma (cylinder + sphere)
        Panel 1 — Solution u on cylinder surface
        Panel 2 — Cross-section at x=0 with sphere contour

    Run with:
        QT_QPA_PLATFORM=wayland python scripts/view_interactive.py
    """

    def __init__(self, output_dir: Path = OUTPUTS_DIR) -> None:
        super().__init__(output_dir)

    def render(
        self,
        grid              : pyvista.UnstructuredGrid,
        scalar            : str,
        conductivity_config: ConductivityConfig,
        eta_config        : EtaConfig,
    ) -> None:
        """
        Open interactive 3-panel PyVista window.

        Parameters
        ----------
        grid : pyvista.UnstructuredGrid
            VTK grid with the scalar field attached.
        scalar : str
            Name of the scalar array to visualize.
        conductivity_config : ConductivityConfig
            Used to draw the gamma sphere geometry.
        eta_config : EtaConfig
            Used to draw the eta spheres geometry.
        """
        clim    = [float(grid[scalar].min()), float(grid[scalar].max())]
        surface = grid.extract_surface(algorithm="dataset_surface")
        clip    = grid.clip(normal="x", origin=(0, 0, 0))

        # Sphere contour on cross-section
        theta      = np.linspace(0, 2 * np.pi, 200)
        circle_pts = np.column_stack([
            np.zeros(200),
            conductivity_config.radius * np.cos(theta),
            conductivity_config.radius * np.sin(theta),
        ])
        circle = pyvista.Spline(circle_pts, 200)

        # Geometry meshes
        sphere_gam = pyvista.Sphere(
            radius=conductivity_config.radius,
            center=conductivity_config.center.tolist(),
            theta_resolution=60, phi_resolution=60,
        )
        cyl = pyvista.Cylinder(
            center=(0, 0, 0), direction=(0, 0, 1),
            radius=1.0, height=2.0, resolution=100, capping=True,
        ).extract_surface()

        # panel plotter 
        pl = pyvista.Plotter(shape=(1, 3), window_size=(1800, 700))

        # Panel 0 — gamma geometry
        pl.subplot(0, 0)
        pl.add_text("Conductivity γ", font_size=12, color="white")
        pl.add_mesh(cyl, color="lightblue", opacity=0.3,
                    show_edges=False, lighting=True)
        pl.add_mesh(sphere_gam, color="red", opacity=0.95,
                    show_edges=False, lighting=True, smooth_shading=True)
        pl.set_background(self.BG)
        pl.add_axes()

        # Panel 1 — solution on surface
        pl.subplot(0, 1)
        pl.add_text("Solution u", font_size=12, color="white")
        pl.add_mesh(surface, scalars=scalar, cmap="turbo", clim=clim,
                    show_scalar_bar=True, show_edges=False,
                    lighting=True, smooth_shading=True)
        pl.set_background(self.BG)
        pl.add_axes()

        # Panel 2 — cross-section
        pl.subplot(0, 2)
        pl.add_text("Cross-section at x=0", font_size=12, color="white")
        pl.add_mesh(clip, scalars=scalar, cmap="turbo", clim=clim,
                    show_scalar_bar=False, show_edges=False,
                    lighting=True, smooth_shading=True)
        pl.add_mesh(circle, color="white", line_width=3)
        pl.set_background(self.BG)
        pl.add_axes()
        pl.camera_position = [(5, 0, 0), (0, 0, 0), (0, 0, 1)]

        pl.link_views()
        logger.info("Opening interactive window ...")
        pl.show()
