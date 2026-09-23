"""Shared state and geometry helpers for the renderers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyvista

from eit3d.config import EITConfig


class BaseRenderer:

    BG          = "#1e1e2e"
    WINDOW_SIZE = (1000, 1000)

    def __init__(self, config: EITConfig) -> None:
        self._config = config

    @property
    def config(self) -> EITConfig:
        return self._config

    # Geometry built from the config

    def _cylinder(self) -> pyvista.PolyData:
        """Cylinder surface matching the mesh domain (centered at the origin)."""
        return pyvista.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=self._config.mesh.radius, height=self._config.mesh.height,
            resolution=100, capping=True,
        )

    @staticmethod
    def _sphere(center: Sequence[float], radius: float) -> pyvista.PolyData:
        return pyvista.Sphere(
            radius=radius, center=tuple(float(c) for c in center),
            theta_resolution=60, phi_resolution=60,
        )

    @staticmethod
    def _section_circle(
        center: Sequence[float], radius: float, axis: int, plane: float,
    ) -> pyvista.PolyData:
        """
        Intersection of a sphere with the plane {x_axis = plane}, as a closed
        curve. Returns an empty PolyData if the plane misses the sphere.
        """
        dist = plane - float(center[axis])
        if abs(dist) >= radius:
            return pyvista.PolyData()
        r_cut = np.sqrt(radius**2 - dist**2)
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        u, v  = [i for i in range(3) if i != axis]
        pts   = np.zeros((200, 3))
        pts[:, axis] = plane
        pts[:, u]    = center[u] + r_cut * np.cos(theta)
        pts[:, v]    = center[v] + r_cut * np.sin(theta)
        return pyvista.Spline(pts, 200)

    def _camera_distance(self) -> float:
        """Distance that frames the whole cylinder in axis-aligned views."""
        return 2.5 * max(self._config.mesh.radius, self._config.mesh.height)

    @staticmethod
    def _scalar_bar_args(title: str) -> dict:
        """Standard scalar bar layout for PyVista."""
        return dict(
            title=title, title_font_size=20, label_font_size=15,
            color="white", position_x=0.03, position_y=0.03,
            width=0.42, height=0.06,
        )

    @staticmethod
    def _fmt(value: float) -> str:
        """Compact number formatting for titles (0.35, 2, -0.3)."""
        return f"{float(value):g}"

    @classmethod
    def _fmt_point(cls, point: Sequence[float]) -> str:
        return "(" + ", ".join(cls._fmt(c) for c in point) + ")"