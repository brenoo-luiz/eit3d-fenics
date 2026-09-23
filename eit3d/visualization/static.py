from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista

from eit3d.config import OUTPUTS_DIR, EITConfig
from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SurfacePanel:
    """One panel of a multi-panel figure: a scalar field on the domain surface."""
    grid  : pyvista.UnstructuredGrid
    scalar: str
    title : str
    cmap  : str                              = "turbo"
    clim  : Optional[Tuple[float, float]]    = None
    bar   : Optional[str]                    = None


class StaticRenderer(BaseRenderer):

    def __init__(self, config: EITConfig, output_dir: Path = OUTPUTS_DIR) -> None:
        super().__init__(config)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # Public figures

    def render_mesh(
        self, grid: pyvista.UnstructuredGrid, filename: str = "step1_mesh.png",
    ) -> Path:
        """Boundary mesh with element edges (single image)."""
        p = self._plotter(window_size=(900, 900))
        p.add_mesh(grid, color="lightblue", show_edges=True,
                    edge_color="#2a2a4a", line_width=0.3)
        p.view_isometric()
        p.camera.zoom(1.1)
        out = self._output_dir / filename
        p.screenshot(str(out))
        p.close()
        logger.info("Saved: %s", out)
        return out

    def render_geometry(self, filename: str = "step3_a_geometria.png") -> Path:
        """Conductivity inclusion (left) and derivative direction spheres (right)."""
        gam, eta = self._config.conductivity, self._config.eta
        cyl = (self._cylinder(), "#1a3a6a", 0.35)

        img_gamma = self._geometry_image(
            [cyl, (self._sphere(gam.center, gam.radius), "#cc3333", 0.95)],
            label=f"gamma = {self._fmt(gam.gamma_in)} (sphere)  /  "
                    f"gamma = {self._fmt(gam.gamma_out)} (background)",
        )
        img_eta = self._geometry_image(
            [cyl] + [(self._sphere(c, eta.radius), "#33aa33", 0.95) for c in eta.centers],
            label=f"eta = {self._fmt(eta.eta_in)} (spheres)  /  "
                    f"eta = {self._fmt(eta.eta_out)} (background)",
        )
        return self._compose([img_gamma, img_eta], self._geometry_titles(), filename)

    def render_forward(
        self,
        grid    : pyvista.UnstructuredGrid,
        scalar  : str,
        pattern : int = 0,
        filename: str = "step3_b_forward.png",
    ) -> Path:
        """Forward solution on the surface (left) and on a section through the inclusion (right)."""
        clim = self._clim(grid, scalar)
        return self._compose(
            [
                self._surface_image(grid, scalar, "turbo", clim, bar="u_γ"),
                self._inclusion_section_image(grid, scalar, "turbo", clim),
            ],
            self._forward_titles(pattern),
            filename,
        )

    def render_forward_overview(
        self,
        grid    : pyvista.UnstructuredGrid,
        scalar  : str,
        filename: str = "step2_forward.png",
    ) -> Path:
        """Inclusion geometry, surface solution and section, side by side."""
        gam  = self._config.conductivity
        clim = self._clim(grid, scalar)
        img_geo = self._geometry_image(
            [(self._cylinder(), "#1a3a6a", 0.45),
            (self._sphere(gam.center, gam.radius), "#cc3333", 0.95)],
            # VTK text has no Greek glyphs: labels drawn inside the 3D view use "gamma"
            label=f"gamma = {self._fmt(gam.gamma_in)} (sphere)  /  "
                    f"gamma = {self._fmt(gam.gamma_out)} (background)",
        )
        return self._compose(
            [
                img_geo,
                self._surface_image(grid, scalar, "turbo", clim, bar=scalar),
                self._inclusion_section_image(grid, scalar, "turbo", clim),
            ],
            [
                "Conductivity γ (sphere inclusion)",
                "Solution u: cylinder surface",
                f"Cross-section at x={self._fmt(gam.center[0])}",
            ],
            filename,
        )

    def render_omega(
        self,
        grid    : pyvista.UnstructuredGrid,
        scalar  : str,
        integral: float,
        filename: str = "step3_c_omega.png",
    ) -> Path:
        """Directional derivative on the surface and on a section through the eta spheres."""
        clim  = self._clim(grid, scalar)
        plane = float(self._config.eta.centers[0][AXES["y"]])
        return self._compose(
            [
                self._surface_image(grid, scalar, "coolwarm", clim, bar="ω"),
                self._section_image(grid, scalar, "coolwarm", clim, axis="y", plane=plane),
            ],
            [
                f"ω: surface  (F'(γ)η = ω|_{{∂Ω}})\n"
                f"[{clim[0]:.4f}, {clim[1]:.4f}]  |  ∫ω ds = {integral:.2e}",
                f"ω: cross-section at y={self._fmt(plane)}\n"
                "(internal structure reflects η spheres)",
            ],
            filename,
        )

    def render_surfaces(self, panels: Sequence[SurfacePanel], filename: str) -> Path:
        """Generic figure: one surface plot per panel, side by side."""
        images = [
            self._surface_image(
                pn.grid, pn.scalar, pn.cmap,
                pn.clim if pn.clim is not None else self._clim(pn.grid, pn.scalar),
                bar=pn.bar if pn.bar is not None else pn.scalar,
            )
            for pn in panels
        ]
        return self._compose(images, [pn.title for pn in panels], filename)

    def render_consistency(
        self,
        y_vals  : np.ndarray,
        t_vals  : np.ndarray,
        slope   : float,
        fit_line: np.ndarray,
        idx_min : int,
        filename: str = "step3_d_consistencia.png",
    ) -> Path:
        """y_n vs n (semilog) and log y_n vs log t_n with the fitted rate."""
        ns    = np.arange(len(y_vals))
        log_t = np.log10(t_vals)
        log_y = np.log10(y_vals)
        ref_y = log_t - log_t[0] + log_y[0]
        base  = self._fmt(self._config.consistency.base)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=self.BG)

        ax1.semilogy(ns, y_vals, "o-", color="#44aaff", linewidth=2,
                    markersize=5, markerfacecolor="white", label="$y_n$")
        ax1.axvline(idx_min, color="#ffcc44", linestyle="--", alpha=0.6,
                    label=f"minimum n={idx_min}")
        ax1.set_xlabel("n", color="white", fontsize=13)
        ax1.set_ylabel("$y_n$ (log scale)", color="white", fontsize=13)
        ax1.set_title(
            f"$y_n$ vs $n$,  $t_n = {base}^n$\n"
            r"$y_n = \|z_n - \omega\|_{L^2(\partial\Omega)} / \|\omega\|_{L^2(\partial\Omega)}$",
            color="white", fontsize=12,
        )
        ax1.annotate(
            f"min: y_{idx_min} = {y_vals[idx_min]:.2e}",
            xy=(idx_min, y_vals[idx_min]),
            xytext=(idx_min + 1, y_vals[idx_min] * 3),
            color="#ffcc44", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#ffcc44"),
        )

        ax2.plot(log_t, log_y, "o", color="#44aaff", markersize=5,
                markerfacecolor="white", label=r"$\log y_n$ vs $\log t_n$")
        ax2.plot(log_t[:len(fit_line)], fit_line, "--", color="#ff7744",
                linewidth=2, label=f"linear fit (slope ≈ {slope:.2f})")
        ax2.plot(log_t, ref_y, ":", color="#aaaaaa", linewidth=1.5,
                label="slope = 1 (theoretical)")
        ax2.set_xlabel(r"$\log_{10}(t_n)$", color="white", fontsize=13)
        ax2.set_ylabel(r"$\log_{10}(y_n)$", color="white", fontsize=13)
        ax2.set_title(
            "Convergence rate: $\\log t_n$ vs $\\log y_n$\n"
            "Slope $\\approx 1$ confirms $y_n = O(t_n)$ (Fréchet derivative)",
            color="white", fontsize=12,
        )
        ax2.annotate(
            f"estimated rate: {slope:.3f}",
            xy=(log_t[len(fit_line) - 1], fit_line[-1]),
            xytext=(log_t[len(fit_line) - 1] + 0.05, fit_line[-1] + 0.2),
            color="#ff7744", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#ff7744"),
        )

        for ax in (ax1, ax2):
            self._style_axes(ax)

        plt.tight_layout(pad=1.0)
        return self._save(fig, filename)

    # Titles
    def _geometry_titles(self) -> List[str]:
        gam, eta = self._config.conductivity, self._config.eta
        n        = len(eta.centers)
        centers  = ", ".join(self._fmt_point(c) for c in eta.centers)
        return [
            f"Conductivity γ\n(sphere r={self._fmt(gam.radius)}, "
            f"γ={self._fmt(gam.gamma_in)} inside / γ={self._fmt(gam.gamma_out)} outside)",
            f"Derivative direction η\n({n} sphere{'s' if n != 1 else ''} "
            f"r={self._fmt(eta.radius)} at {centers},\n"
            f"η={self._fmt(eta.eta_in)} inside / η={self._fmt(eta.eta_out)} outside)",
        ]

    def _forward_titles(self, pattern: int) -> List[str]:
        g_top, g_bot = self._config.current.patterns[pattern]
        x0 = self._fmt(self._config.conductivity.center[0])
        return [
            "u_γ: cylinder surface\n"
            f"(forward problem with γ, g={g_top:+g} top / g={g_bot:+g} base)",
            f"u_γ: cross-section at x={x0}\n(white circle = sphere boundary)",
        ]

    # Image builders
    def _plotter(self, window_size=None) -> pyvista.Plotter:
        p = pyvista.Plotter(off_screen=True, window_size=window_size or self.WINDOW_SIZE)
        p.set_background(self.BG)
        return p

    @staticmethod
    def _snap(p: pyvista.Plotter) -> np.ndarray:
        img = p.screenshot(return_img=True)
        p.close()
        return img

    def _geometry_image(self, meshes, label: str) -> np.ndarray:
        p = self._plotter()
        for mesh, color, opacity in meshes:
            p.add_mesh(mesh, color=color, opacity=opacity,
                        show_edges=False, lighting=True, smooth_shading=True)
        p.add_text(label, position="lower_left", font_size=12, color="white")
        p.view_isometric()
        p.camera.zoom(1.1)
        return self._snap(p)

    def _surface_image(self, grid, scalar: str, cmap: str, clim, bar: str) -> np.ndarray:
        p = self._plotter()
        p.add_mesh(grid.extract_surface(algorithm="dataset_surface"),
                    scalars=scalar, cmap=cmap, clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=True, scalar_bar_args=self._scalar_bar_args(bar))
        p.view_isometric()
        return self._snap(p)

    def _section_image(
        self, grid, scalar: str, cmap: str, clim,
        axis: str, plane: float, overlays: Sequence[pyvista.PolyData] = (),
    ) -> np.ndarray:
        """Clip the domain by the plane {axis = plane} and look at the cut face."""
        idx    = AXES[axis]
        origin = [0.0, 0.0, 0.0]
        origin[idx] = plane
        eye    = [0.0, 0.0, 0.0]
        eye[idx] = plane + self._camera_distance()

        p = self._plotter()
        p.add_mesh(grid.clip(normal=axis, origin=origin),
                    scalars=scalar, cmap=cmap, clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=False)
        for overlay in overlays:
            if overlay.n_points:
                p.add_mesh(overlay, color="white", line_width=3)
        p.camera_position = [tuple(eye), tuple(origin), (0.0, 0.0, 1.0)]
        p.camera.zoom(1.4)
        return self._snap(p)

    def _inclusion_section_image(self, grid, scalar: str, cmap: str, clim) -> np.ndarray:
        """Section x = x_center through the conductivity inclusion, with its contour."""
        gam   = self._config.conductivity
        plane = float(gam.center[AXES["x"]])
        circle = self._section_circle(gam.center, gam.radius, AXES["x"], plane)
        return self._section_image(grid, scalar, cmap, clim, "x", plane, [circle])

    # Figure assembly
    def _compose(self, images: Sequence[np.ndarray], titles: Sequence[str], filename: str) -> Path:
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), facecolor=self.BG)
        for ax, img, title in zip(np.atleast_1d(axes), images, titles):
            ax.imshow(img)
            ax.axis("off")
            ax.set_facecolor(self.BG)
            ax.set_title(title, color="white", fontsize=13, pad=8)
        plt.tight_layout(pad=0.3)
        return self._save(fig, filename)

    def _save(self, fig, filename: str) -> Path:
        out = self._output_dir / filename
        fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=self.BG)
        plt.close(fig)
        logger.info("Saved: %s", out)
        return out

    def _style_axes(self, ax) -> None:
        ax.set_facecolor(self.BG)
        ax.tick_params(colors="white")
        ax.legend(facecolor=self.BG, edgecolor="#4a4a6a", labelcolor="white", fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4a4a6a")
        ax.grid(True, color="#4a4a6a", linestyle="--", alpha=0.4)

    @staticmethod
    def _clim(grid, scalar: str) -> Tuple[float, float]:
        return float(grid[scalar].min()), float(grid[scalar].max())