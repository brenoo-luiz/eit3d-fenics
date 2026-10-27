from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pyvista

from eit3d.config import EITConfig
from eit3d.visualization.base import BaseRenderer
from eit3d.visualization.grid import iteration_fields

logger = logging.getLogger(__name__)


class InteractiveRenderer(BaseRenderer):
    def show(self, grid: pyvista.UnstructuredGrid, scalar: str) -> None:
        gam = self._config.conductivity
        plane = float(gam.center[0])
        clim = [float(grid[scalar].min()), float(grid[scalar].max())]
        circle = self._section_circle(gam.center, gam.radius, axis=0, plane=plane)

        pl = pyvista.Plotter(shape=(1, 3), window_size=(1800, 700))

        pl.subplot(0, 0)
        pl.add_text("Conductivity γ", font_size=12, color="white")
        pl.add_mesh(
            self._cylinder(),
            color="lightblue",
            opacity=0.3,
            show_edges=False,
            lighting=True,
        )
        pl.add_mesh(
            self._sphere(gam.center, gam.radius),
            color="red",
            opacity=0.95,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
        )

        pl.subplot(0, 1)
        pl.add_text(f"Solution {scalar}", font_size=12, color="white")
        pl.add_mesh(
            grid.extract_surface(algorithm="dataset_surface"),
            scalars=scalar,
            cmap="turbo",
            clim=clim,
            show_scalar_bar=True,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
        )

        pl.subplot(0, 2)
        pl.add_text(
            f"Cross-section at x={self._fmt(plane)}", font_size=12, color="white"
        )
        pl.add_mesh(
            grid.clip(normal="x", origin=(plane, 0.0, 0.0)),
            scalars=scalar,
            cmap="turbo",
            clim=clim,
            show_scalar_bar=False,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
        )
        if circle.n_points:
            pl.add_mesh(circle, color="white", line_width=3)
        pl.camera_position = [
            (plane + self._camera_distance(), 0.0, 0.0),
            (plane, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

        for col in range(3):
            pl.subplot(0, col)
            pl.set_background(self.BG)
            pl.add_axes()

        pl.link_views()
        logger.info("Opening interactive window ...")
        pl.show()


class InversionViewer(BaseRenderer):
    BG = "white"
    INK = "#222222"
    FRAME = "#9a9a9a"
    TRUE_COLOR = "#d62728"
    BODY_COLOR = "#dfe7ef"
    CMAP = "viridis"
    LEVEL = 0.5
    WINDOW = (1600, 800)

    def __init__(
        self,
        config: EITConfig,
        grid: pyvista.DataSet,
        spheres: Sequence[tuple[Sequence[float], float]],
    ) -> None:
        super().__init__(config)
        self._names = iteration_fields(grid)
        if not self._names:
            raise ValueError("the grid has no reconstructed field")
        self._points = grid.cell_data_to_point_data()
        self._spheres = list(spheres)
        values = np.concatenate([grid.cell_data[n] for n in self._names])
        self._clim = (float(values.min()), float(values.max()))
        self._ks = [self._iteration(name) for name in self._names]
        self._index = len(self._names) - 1
        self._fraction = self.LEVEL
        self._plotter: pyvista.Plotter | None = None

    @property
    def names(self) -> list[str]:
        return self._names

    def level(self, name: str, fraction: float) -> float:
        values = self._points.point_data[name]
        low, high = float(np.median(values)), float(values.max())
        return low + fraction * (high - low)

    def isosurface(self, name: str, fraction: float) -> pyvista.PolyData:
        return self._points.contour([self.level(name, fraction)], scalars=name)

    def plotter(self, off_screen: bool = False) -> pyvista.Plotter:
        pl = pyvista.Plotter(
            shape=(1, 2), window_size=self.WINDOW, off_screen=off_screen, border=False
        )
        self._plotter = pl
        pl.set_background(self.BG)

        pl.subplot(0, 0)
        self._body(pl)
        for center, radius in self._spheres:
            pl.add_mesh(
                self._sphere(center, radius),
                color=self.TRUE_COLOR,
                smooth_shading=True,
                specular=0.3,
            )
        pl.add_text("Verdadeira", font_size=11, color=self.INK)

        pl.subplot(0, 1)
        self._body(pl)
        for center, radius in self._spheres:
            pl.add_mesh(
                self._sphere(center, radius),
                color=self.TRUE_COLOR,
                opacity=0.15,
                smooth_shading=True,
            )
        self._draw()
        if len(self._names) > 1:
            pl.add_slider_widget(
                self._on_iteration,
                (min(self._ks), max(self._ks)),
                value=self._ks[self._index],
                title="iteração k",
                fmt="%.0f",
                pointa=(0.1, 0.1),
                pointb=(0.45, 0.1),
                style="modern",
                color=self.INK,
            )
        pl.add_slider_widget(
            self._on_level,
            (0.05, 0.95),
            value=self._fraction,
            title="nível",
            fmt="%.2f",
            pointa=(0.55, 0.1),
            pointb=(0.9, 0.1),
            style="modern",
            color=self.INK,
        )

        pl.link_views()
        pl.enable_parallel_projection()
        pl.camera_position = [(1.0, -1.0, 0.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        pl.reset_camera()
        return pl

    def show(self) -> None:
        logger.info("Opening interactive window ...")
        self.plotter().show()

    def _body(self, pl: pyvista.Plotter) -> None:
        pl.add_mesh(
            self._cylinder(), color=self.BODY_COLOR, opacity=0.12, smooth_shading=True
        )
        mesh = self._config.mesh
        for z in (-mesh.height / 2, mesh.height / 2):
            pl.add_mesh(self._rim(mesh.radius, z), color=self.FRAME, line_width=2)

    @staticmethod
    def _rim(radius: float, z: float) -> pyvista.PolyData:
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        points = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z)]
        )
        return pyvista.Spline(points, 200)

    @staticmethod
    def _iteration(name: str) -> int:
        if not (name.startswith("k") and name[1:].isdigit()):
            raise ValueError(f"field {name!r} is not named k<iteration>")
        return int(name[1:])

    def _draw(self) -> None:
        pl = self._plotter
        pl.subplot(0, 1)
        name = self._names[self._index]
        level = self.level(name, self._fraction)
        pl.add_mesh(
            self.isosurface(name, self._fraction),
            scalars=name,
            cmap=self.CMAP,
            clim=self._clim,
            smooth_shading=True,
            specular=0.3,
            name="iso",
            scalar_bar_args={
                "title": "condutividade",
                "color": self.INK,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.3,
                "height": 0.45,
                "title_font_size": 14,
                "label_font_size": 12,
                "width": 0.05,
                "fmt": "%.2f",
            },
        )
        pl.add_text(
            f"Reconstruída  (k = {self._ks[self._index]}, isosuperfície {level:.2f})",
            font_size=11,
            color=self.INK,
            name="label",
        )

    def _on_iteration(self, value: float) -> None:
        index = int(np.argmin([abs(k - value) for k in self._ks]))
        if index != self._index:
            self._index = index
            self._draw()

    def _on_level(self, value: float) -> None:
        self._fraction = float(value)
        self._draw()
