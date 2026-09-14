"""
Static PNG renderer for EIT 3D results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dolfinx.plot
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from PIL import Image

from eit3d.config import OUTPUTS_DIR
from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)


class StaticRenderer(BaseRenderer):
    """
    Renders EIT 3D results as static PNG files using PyVista + Matplotlib.

    Produces 4 output files:
        step3_a_geometria.png   — gamma and eta geometries
        step3_b_forward.png     — u_gamma (surface + cross-section)
        step3_c_omega.png       — omega (surface + cross-section)
        step3_d_consistencia.png— consistency test plots
    """

    def __init__(self, output_dir: Path = OUTPUTS_DIR) -> None:
        super().__init__(output_dir)

    def render(self, **kwargs) -> None:
        """Entry point — delegates to specific render methods."""
        raise NotImplementedError("Call render_geometry, render_forward, etc. directly.")

    def render_geometry(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        eta_centers  : list,
        eta_radius   : float,
        filename     : str = "step3_a_geometria.png",
    ) -> Path:
        """Render gamma and eta geometries side by side."""
        cyl = pyvista.Cylinder(
            center=(0, 0, 0), direction=(0, 0, 1),
            radius=1.0, height=2.0, resolution=100, capping=True,
        ).extract_surface()

        sphere_gam = pyvista.Sphere(
            radius=sphere_radius, center=sphere_center.tolist(),
            theta_resolution=60, phi_resolution=60,
        )
        spheres_eta = [
            pyvista.Sphere(radius=eta_radius, center=c.tolist(),
                        theta_resolution=60, phi_resolution=60)
            for c in eta_centers
        ]

        imgs = []
        for meshes, colors, label in [
            ([(cyl, "#1a3a6a", 0.35), (sphere_gam, "#cc3333", 0.95)],
            "gamma = 2 (sphere)  /  gamma = 1 (background)"),
            ([(cyl, "#1a3a6a", 0.35)] + [(s, "#33aa33", 0.95) for s in spheres_eta],
            "eta = 1 (spheres)  /  eta = 0 (background)"),
        ]:
            tmp = self._output_dir / "_tmp_geo.png"
            p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
            for m, color, opacity in meshes:
                p.add_mesh(m, color=color, opacity=opacity,
                            show_edges=False, lighting=True, smooth_shading=True)
            p.add_text(label, position="lower_left", font_size=12, color="white")
            p.set_background(self.BG)
            p.view_isometric()
            p.camera.zoom(1.1)
            p.screenshot(str(tmp))
            p.close()
            imgs.append((np.array(Image.open(tmp)), label.split("/")[0].strip()))
            tmp.unlink()

        out = self._compose_2panel(imgs, titles=[
            "Conductivity γ\n(sphere r=0.35, γ=2 inside / γ=1 outside)",
            "Derivative direction η\n(2 spheres r=0.2 at x=±0.3, η=1 inside / η=0 outside)",
        ], filename=filename)
        logger.info("Saved: %s", out)
        return out

    def render_forward(
        self,
        grid         : pyvista.UnstructuredGrid,
        scalar       : str,
        sphere_radius: float,
        filename     : str = "step3_b_forward.png",
    ) -> Path:
        """Render u_gamma — surface and cross-section."""
        clim = [float(grid[scalar].min()), float(grid[scalar].max())]
        surf = grid.extract_surface(algorithm="dataset_surface")
        clip = grid.clip(normal="x", origin=(0, 0, 0))

        theta      = np.linspace(0, 2 * np.pi, 200)
        circle_pts = np.column_stack([
            np.zeros(200),
            sphere_radius * np.cos(theta),
            sphere_radius * np.sin(theta),
        ])
        circle = pyvista.Spline(circle_pts, 200)

        tmp1 = self._output_dir / "_tmp_surf.png"
        tmp2 = self._output_dir / "_tmp_clip.png"

        p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
        p.add_mesh(surf, scalars=scalar, cmap="turbo", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=True, scalar_bar_args=self._sargs("u_γ"))
        p.set_background(self.BG)
        p.view_isometric()
        p.screenshot(str(tmp1))
        p.close()

        p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
        p.add_mesh(clip, scalars=scalar, cmap="turbo", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=False)
        p.add_mesh(circle, color="white", line_width=3)
        p.set_background(self.BG)
        p.camera_position = [(5, 0, 0), (0, 0, 0), (0, 0, 1)]
        p.camera.zoom(1.4)
        p.screenshot(str(tmp2))
        p.close()

        imgs = [np.array(Image.open(t)) for t in [tmp1, tmp2]]
        for t in [tmp1, tmp2]:
            t.unlink()

        out = self._compose_2panel(
            list(zip(imgs, ["", ""])),
            titles=[
                "u_γ — Cylinder surface\n(forward problem with γ, g=+1 top / g=−1 base)",
                "u_γ — Cross-section at x=0\n(white circle = sphere boundary)",
            ],
            filename=filename,
        )
        logger.info("Saved: %s", out)
        return out

    def render_omega(
        self,
        grid    : pyvista.UnstructuredGrid,
        scalar  : str,
        integral: float,
        filename: str = "step3_c_omega.png",
    ) -> Path:
        """Render omega — surface and cross-section at y=0."""
        clim = [float(grid[scalar].min()), float(grid[scalar].max())]
        surf = grid.extract_surface(algorithm="dataset_surface")
        clip = grid.clip(normal="y", origin=(0, 0, 0))

        tmp1 = self._output_dir / "_tmp_osurf.png"
        tmp2 = self._output_dir / "_tmp_oclip.png"

        p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
        p.add_mesh(surf, scalars=scalar, cmap="coolwarm", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=True, scalar_bar_args=self._sargs("ω"))
        p.set_background(self.BG)
        p.view_isometric()
        p.screenshot(str(tmp1))
        p.close()

        p = pyvista.Plotter(off_screen=True, window_size=(1000, 1000))
        p.add_mesh(clip, scalars=scalar, cmap="coolwarm", clim=clim,
                    show_edges=False, lighting=True, smooth_shading=True,
                    show_scalar_bar=False)
        p.set_background(self.BG)
        p.camera_position = [(0, 5, 0), (0, 0, 0), (0, 0, 1)]
        p.camera.zoom(1.4)
        p.screenshot(str(tmp2))
        p.close()

        imgs = [np.array(Image.open(t)) for t in [tmp1, tmp2]]
        for t in [tmp1, tmp2]:
            t.unlink()

        out = self._compose_2panel(
            list(zip(imgs, ["", ""])),
            titles=[
                f"ω — Surface  (F'(γ)η = ω|_{{∂Ω}})\n"
                f"[{clim[0]:.4f}, {clim[1]:.4f}]  |  ∫ω ds = {integral:.2e}",
                "ω — Cross-section at y=0\n(internal structure reflects η spheres)",
            ],
            filename=filename,
        )
        logger.info("Saved: %s", out)
        return out

    def render_consistency(
        self,
        y_vals  : np.ndarray,
        t_vals  : np.ndarray,
        taxa    : float,
        fit_line: np.ndarray,
        idx_min : int,
        filename: str = "step3_d_consistencia.png",
    ) -> Path:
        """Render consistency test: semilogy + log-log convergence rate."""
        ns      = np.arange(len(y_vals))
        log_t   = np.log10(t_vals)
        log_y   = np.log10(y_vals)
        ref_y   = log_t - log_t[0] + log_y[0]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=self.BG)

        # semilogy
        ax1 = axes[0]
        ax1.set_facecolor(self.BG)
        ax1.semilogy(ns, y_vals, "o-", color="#44aaff", linewidth=2,
                    markersize=5, markerfacecolor="white", label="$y_n$")
        ax1.axvline(idx_min, color="#ffcc44", linestyle="--", alpha=0.6,
                    label=f"minimum n={idx_min}")
        ax1.set_xlabel("n", color="white", fontsize=13)
        ax1.set_ylabel("$y_n$ (log scale)", color="white", fontsize=13)
        ax1.set_title(
            "$y_n$ vs $n$  —  $t_n = 0.9^n$\n"
            r"$y_n = \|z_n - \omega\|_{L^2(\partial\Omega)} / \|\omega\|_{L^2(\partial\Omega)}$",
            color="white", fontsize=12,
        )
        ax1.tick_params(colors="white")
        ax1.legend(facecolor=self.BG, edgecolor="#4a4a6a", labelcolor="white", fontsize=11)
        for sp in ax1.spines.values():
            sp.set_edgecolor("#4a4a6a")
        ax1.grid(True, color="#4a4a6a", linestyle="--", alpha=0.4)
        ax1.annotate(
            f"min: y_{idx_min} = {y_vals[idx_min]:.2e}",
            xy=(idx_min, y_vals[idx_min]),
            xytext=(idx_min + 1, y_vals[idx_min] * 3),
            color="#ffcc44", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#ffcc44"),
        )

        # log-log 
        ax2 = axes[1]
        ax2.set_facecolor(self.BG)
        ax2.plot(log_t, log_y, "o", color="#44aaff", markersize=5,
                markerfacecolor="white", label=r"$\log y_n$ vs $\log t_n$")
        ax2.plot(log_t[:idx_min] if idx_min > 5 else log_t, fit_line,
                "--", color="#ff7744", linewidth=2,
                label=f"linear fit (slope ≈ {taxa:.2f})")
        ax2.plot(log_t, ref_y, ":", color="#aaaaaa", linewidth=1.5,
                label="slope = 1 (theoretical)")
        ax2.set_xlabel(r"$\log_{10}(t_n)$", color="white", fontsize=13)
        ax2.set_ylabel(r"$\log_{10}(y_n)$", color="white", fontsize=13)
        ax2.set_title(
            "Convergence rate  —  $\\log t_n$ vs $\\log y_n$\n"
            "Slope $\\approx 1$ confirms $y_n = O(t_n)$ (Fréchet derivative)",
            color="white", fontsize=12,
        )
        ax2.tick_params(colors="white")
        ax2.legend(facecolor=self.BG, edgecolor="#4a4a6a", labelcolor="white", fontsize=11)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#4a4a6a")
        ax2.grid(True, color="#4a4a6a", linestyle="--", alpha=0.4)
        ax2.annotate(
            f"estimated rate: {taxa:.3f}",
            xy=(log_t[5], fit_line[5] if idx_min > 5 else log_y[5]),
            xytext=(log_t[5] + 0.05, (fit_line[5] if idx_min > 5 else log_y[5]) + 0.2),
            color="#ff7744", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="#ff7744"),
        )

        plt.tight_layout(pad=1.0)
        out = self._output_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=self.BG)
        plt.close()
        logger.info("Saved: %s", out)
        return out

    def _compose_2panel(self, imgs_titles: list, titles: list, filename: str) -> Path:
        """Compose two images side by side with matplotlib."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=self.BG)
        for ax, (img, _), title in zip(axes, imgs_titles, titles):
            ax.imshow(img)
            ax.axis("off")
            ax.set_facecolor(self.BG)
            ax.set_title(title, color="white", fontsize=13, pad=8)
        plt.tight_layout(pad=0.3)
        out = self._output_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=self.BG)
        plt.close()
        return out
