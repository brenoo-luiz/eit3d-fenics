"""Interactive PyVista renderer"""

from __future__ import annotations

import logging

import pyvista

from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)


class InteractiveRenderer(BaseRenderer):

    def show(self, grid: pyvista.UnstructuredGrid, scalar: str) -> None:
        """Open the window (blocks until it is closed)."""
        gam    = self._config.conductivity
        plane  = float(gam.center[0])
        clim   = [float(grid[scalar].min()), float(grid[scalar].max())]
        circle = self._section_circle(gam.center, gam.radius, axis=0, plane=plane)

        pl = pyvista.Plotter(shape=(1, 3), window_size=(1800, 700))

        pl.subplot(0, 0)
        pl.add_text("Conductivity γ", font_size=12, color="white")
        pl.add_mesh(self._cylinder(), color="lightblue", opacity=0.3,
                    show_edges=False, lighting=True)
        pl.add_mesh(self._sphere(gam.center, gam.radius), color="red", opacity=0.95,
                    show_edges=False, lighting=True, smooth_shading=True)

        pl.subplot(0, 1)
        pl.add_text(f"Solution {scalar}", font_size=12, color="white")
        pl.add_mesh(grid.extract_surface(algorithm="dataset_surface"),
                    scalars=scalar, cmap="turbo", clim=clim,
                    show_scalar_bar=True, show_edges=False,
                    lighting=True, smooth_shading=True)

        pl.subplot(0, 2)
        pl.add_text(f"Cross-section at x={self._fmt(plane)}", font_size=12, color="white")
        pl.add_mesh(grid.clip(normal="x", origin=(plane, 0.0, 0.0)),
                    scalars=scalar, cmap="turbo", clim=clim,
                    show_scalar_bar=False, show_edges=False,
                    lighting=True, smooth_shading=True)
        if circle.n_points:
            pl.add_mesh(circle, color="white", line_width=3)
        pl.camera_position = [
            (plane + self._camera_distance(), 0.0, 0.0), (plane, 0.0, 0.0), (0.0, 0.0, 1.0),
        ]

        for col in range(3):
            pl.subplot(0, col)
            pl.set_background(self.BG)
            pl.add_axes()

        pl.link_views()
        logger.info("Opening interactive window ...")
        pl.show()