from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar

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
    CMAP = "turbo"
    LEVEL = 0.5
    WINDOW = (1800, 700)
    PLANE = 0.0
    ZOOM = 0.75
    BAR: ClassVar[dict[str, object]] = {
        "title": "condutividade",
        "color": "white",
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.2,
        "width": 0.05,
        "height": 0.6,
        "title_font_size": 16,
        "label_font_size": 14,
        "fmt": "%.2f",
    }
    OFFSET = 0.01
    SLIDER: ClassVar[dict[str, float]] = {
        "tube_width": 0.004,
        "slider_width": 0.015,
        "title_height": 0.025,
    }

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
        self._grid = grid
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

    def section(self, name: str) -> pyvista.DataSet:
        grid = self._grid.copy()
        grid.set_active_scalars(name, preference="cell")
        return grid.clip(normal="y", origin=(0.0, self.PLANE, 0.0), invert=False)

    def plotter(self, off_screen: bool = False) -> pyvista.Plotter:
        pl = pyvista.Plotter(
            shape=(1, 3), window_size=self.WINDOW, off_screen=off_screen
        )
        self._plotter = pl

        pl.subplot(0, 0)
        pl.add_text("Condutividade verdadeira", font_size=12, color="white")
        self._body(pl)
        for center, radius in self._spheres:
            pl.add_mesh(
                self._sphere(center, radius),
                color="red",
                opacity=0.95,
                lighting=True,
                smooth_shading=True,
            )

        pl.subplot(0, 1)
        self._body(pl)
        if len(self._names) > 1:
            pl.add_slider_widget(
                self._on_iteration,
                (min(self._ks), max(self._ks)),
                value=self._ks[self._index],
                title="iteração k",
                fmt="%.0f",
                pointa=(0.1, 0.88),
                pointb=(0.9, 0.88),
                style="modern",
                color="white",
                **self.SLIDER,
            )
        pl.add_slider_widget(
            self._on_level,
            (0.05, 0.95),
            value=self._fraction,
            title="nível da isosuperfície",
            fmt="%.2f",
            pointa=(0.55, 0.08),
            pointb=(0.95, 0.08),
            style="modern",
            color="white",
            **self.SLIDER,
        )

        pl.subplot(0, 2)
        pl.add_text(
            f"Corte em y = {self._fmt(self.PLANE)}", font_size=12, color="white"
        )
        for center, radius in self._spheres:
            circle = self._section_circle(center, radius, axis=1, plane=self.PLANE)
            if circle.n_points:
                circle.translate((0.0, -self.OFFSET, 0.0), inplace=True)
                pl.add_mesh(circle, color="white", line_width=3)
        self._draw()

        for col in range(3):
            pl.subplot(0, col)
            pl.set_background(self.BG)
            pl.add_axes()

        pl.link_views()
        pl.camera_position = [
            (0.0, -self._camera_distance(), 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        pl.camera.zoom(self.ZOOM)
        return pl

    def show(self) -> None:
        logger.info("Opening interactive window ...")
        self.plotter().show()

    def _body(self, pl: pyvista.Plotter) -> None:
        pl.add_mesh(self._cylinder(), color="lightblue", opacity=0.3, lighting=True)

    @staticmethod
    def _iteration(name: str) -> int:
        if not (name.startswith("k") and name[1:].isdigit()):
            raise ValueError(f"field {name!r} is not named k<iteration>")
        return int(name[1:])

    def _draw(self) -> None:
        pl = self._plotter
        name = self._names[self._index]
        style = {
            "scalars": name,
            "cmap": self.CMAP,
            "clim": self._clim,
            "lighting": True,
            "smooth_shading": True,
        }

        pl.subplot(0, 1)
        pl.add_mesh(
            self.isosurface(name, self._fraction),
            name="iso",
            show_scalar_bar=False,
            **style,
        )
        level = self.level(name, self._fraction)
        pl.add_text(
            f"Reconstruída (k = {self._ks[self._index]}, nível {level:.2f})",
            font_size=12,
            color="white",
            name="label",
        )

        pl.subplot(0, 2)
        pl.add_mesh(
            self.section(name),
            name="section",
            scalar_bar_args=self.BAR,
            **style,
        )

    def _on_iteration(self, value: float) -> None:
        index = int(np.argmin([abs(k - value) for k in self._ks]))
        if index != self._index:
            self._index = index
            self._draw()

    def _on_level(self, value: float) -> None:
        self._fraction = float(value)
        self._draw()
