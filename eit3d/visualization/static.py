from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from eit3d.config import OUTPUTS_DIR, EITConfig
from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)

AXES = {"x": 0, "y": 1, "z": 2}

PRIMARY = "#1f3a68"
ACCENT = "#c0392b"
MUTED = "#8a8f98"
TEXT = "#222222"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": TEXT,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "#d9d9d9",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "font.size": 11,
    "text.color": TEXT,
}


@dataclass(frozen=True)
class SurfacePanel:
    """One panel of a multi-panel figure: a scalar field on the domain surface."""
    grid: pyvista.UnstructuredGrid
    scalar: str
    title: str
    cmap: str = "coolwarm"
    clim: Optional[Tuple[float, float]] = None
    bar: Optional[str] = None


@dataclass(frozen=True)
class _Panel:
    image: np.ndarray
    title: str
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    label: Optional[str] = None


class StaticRenderer(BaseRenderer):

    BG = "white"
    WINDOW_SIZE = (1100, 1100)
    DPI = 300

    def __init__(self, config: EITConfig, output_dir: Path = OUTPUTS_DIR) -> None:
        super().__init__(config)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # Public figures
    def render_mesh(self, grid: pyvista.UnstructuredGrid, filename: str = "step1_mesh.png") -> Path:
        p = self._plotter()
        p.add_mesh(grid, color="#e8edf3", show_edges=True, edge_color=PRIMARY, line_width=0.3)
        p.view_isometric()
        mesh = self._config.mesh
        return self._compose(
            [_Panel(self._snap(p), f"Malha  (raio {self._fmt(mesh.radius)}, altura {self._fmt(mesh.height)})")],
            filename, width=6,
        )

    def render_geometry(self, filename: str = "step3_a_geometry.png") -> Path:
        gam, eta = self._config.conductivity, self._config.eta
        left, right = self._geometry_titles()
        img_gamma = self._geometry_image([(self._sphere(gam.center, gam.radius), ACCENT)])
        img_eta = self._geometry_image([(self._sphere(c, eta.radius), "#2e7d4f") for c in eta.centers])
        return self._compose([_Panel(img_gamma, left), _Panel(img_eta, right)], filename)

    def render_forward(
        self,
        grid: pyvista.UnstructuredGrid,
        scalar: str,
        pattern: int = 0,
        filename: str = "step3_b_forward.png",
    ) -> Path:
        clim = self._clim(grid, scalar)
        left, right = self._forward_titles(pattern)
        return self._compose(
            [
                _Panel(self._surface_image(grid, scalar, "coolwarm", clim), left, "coolwarm", clim, r"$u_\gamma$"),
                _Panel(self._inclusion_section_image(grid, scalar, "coolwarm", clim), right,
                        "coolwarm", clim, r"$u_\gamma$"),
            ],
            filename,
        )

    def render_forward_overview(
        self,
        grid: pyvista.UnstructuredGrid,
        scalar: str,
        filename: str = "step2_forward.png",
    ) -> Path:
        gam = self._config.conductivity
        clim = self._clim(grid, scalar)
        img_geo = self._geometry_image([(self._sphere(gam.center, gam.radius), ACCENT)])
        return self._compose(
            [
                _Panel(img_geo, f"Condutividade  (γ={self._fmt(gam.gamma_in)} na esfera, "
                                f"{self._fmt(gam.gamma_out)} fora)"),
                _Panel(self._surface_image(grid, scalar, "coolwarm", clim), "Potencial u na superfície",
                        "coolwarm", clim, "u"),
                _Panel(self._inclusion_section_image(grid, scalar, "coolwarm", clim),
                        f"Corte em x = {self._fmt(gam.center[0])}", "coolwarm", clim, "u"),
            ],
            filename,
        )

    def render_omega(
        self,
        grid: pyvista.UnstructuredGrid,
        scalar: str,
        integral: float,
        filename: str = "step3_c_omega.png",
    ) -> Path:
        clim = self._symmetric_clim(grid, scalar)
        plane = float(self._config.eta.centers[0][AXES["y"]])
        overlays = [
            self._section_circle(c, self._config.eta.radius, AXES["y"], plane) for c in self._config.eta.centers
        ]
        return self._compose(
            [
                _Panel(self._surface_image(grid, scalar, "RdBu_r", clim),
                        f"Derivada ω na superfície  (∫ω ds = {integral:.1e})", "RdBu_r", clim, "ω"),
                _Panel(self._section_image(grid, scalar, "RdBu_r", clim, axis="y", plane=plane, overlays=overlays),
                        f"Corte em y = {self._fmt(plane)}  (círculos: esferas de η)", "RdBu_r", clim, "ω"),
            ],
            filename,
        )

    def render_surfaces(self, panels: Sequence[SurfacePanel], filename: str) -> Path:
        images = []
        for pn in panels:
            clim = pn.clim if pn.clim is not None else self._clim(pn.grid, pn.scalar)
            images.append(_Panel(
                self._surface_image(pn.grid, pn.scalar, pn.cmap, clim), pn.title, pn.cmap, clim,
                pn.bar if pn.bar is not None else pn.scalar,
            ))
        return self._compose(images, filename)

    def render_consistency(
        self,
        y_vals: np.ndarray,
        t_vals: np.ndarray,
        slope: float,
        fit_line: np.ndarray,
        idx_min: int,
        filename: str = "step3_d_consistency.png",
    ) -> Path:
        ns = np.arange(len(y_vals))
        base = self._fmt(self._config.consistency.base)
        n_fit = len(fit_line)

        with plt.rc_context(STYLE):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")

            ax1.semilogy(ns, y_vals, "o-", color=PRIMARY, markersize=3, linewidth=1, label="$y_n$")
            ax1.semilogy(idx_min, y_vals[idx_min], "o", markersize=8, markerfacecolor="none",
                        markeredgecolor=ACCENT, markeredgewidth=1.5,
                        label=f"mínimo: $y_{{{idx_min}}}$ = {y_vals[idx_min]:.1e}")
            ax1.set_xlabel("Índice $n$")
            ax1.set_ylabel("$y_n$")
            ax1.set_title(f"Teste de consistência  ($t_n = {base}^n$)")
            ax1.legend()

            ax2.loglog(t_vals, y_vals, "o", color=PRIMARY, markersize=3, label="$y_n$")
            ax2.loglog(t_vals[:n_fit], 10 ** fit_line, "-", color=ACCENT, linewidth=1.5,
                        label=f"ajuste: inclinação {slope:.3f}")
            ax2.loglog(t_vals, t_vals * y_vals[0] / t_vals[0], ":", color=MUTED, linewidth=1.2,
                        label="inclinação 1 (teoria)")
            ax2.invert_xaxis()
            ax2.set_xlabel("$t_n$")
            ax2.set_ylabel("$y_n$")
            ax2.set_title("Taxa de convergência")
            ax2.legend()
            return self._save(fig, filename)

    def render_inversion_history(
        self,
        residuals: np.ndarray,
        errors: np.ndarray,
        step: float,
        filename: str,
        floor: Optional[float] = None,
    ) -> Path:
        ks = np.arange(len(residuals))
        with plt.rc_context(STYLE):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
            fig.suptitle(f"Método do gradiente,  λ = {step:g}", fontsize=14)

            ax1.semilogy(ks, residuals, "-", color=PRIMARY, linewidth=1.6, label="resíduo")
            if floor is not None:
                ax1.axhline(floor, color=MUTED, linestyle="--", linewidth=1,
                            label=f"resíduo de γ⁺  ({floor:.1e})")
            ax1.set_title(r"Resíduo  $\|F_G(\gamma_k) - \tilde u\|$")
            ax1.set_ylabel("resíduo")

            ax2.plot(ks, errors, "-", color=ACCENT, linewidth=1.6, label="erro relativo")
            ax2.set_title(r"Erro relativo  $\|\gamma^+ - \gamma_k\| \,/\, \|\gamma^+\|$")
            ax2.set_ylabel("erro relativo")

            for ax in (ax1, ax2):
                ax.set_xlabel("Iteração $k$")
                ax.set_xlim(0, ks[-1])
                ax.legend(loc="upper right")
            return self._save(fig, filename)

    def render_conductivity_sections(
        self,
        fields: Sequence[Tuple[str, object]],
        filename: str,
        spheres: Sequence[Tuple[Sequence[float], float]] = (),
        axis: str = "y",
        plane: float = 0.0,
        clim: Optional[Tuple[float, float]] = None,
        ncols: int = 4,
        clims: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> Path:
        import dolfinx.plot

        overlays = [self._section_circle(c, r, AXES[axis], plane) for c, r in spheres]
        clims = clims if clims is not None else [clim] * len(fields)
        panels = []
        for (title, f), panel_clim in zip(fields, clims):
            mesh = f.function_space.mesh
            n = mesh.topology.index_map(mesh.topology.dim).size_local
            cells = np.arange(n, dtype=np.int32)
            topo, ct, geo = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim, cells)
            grid = pyvista.UnstructuredGrid(topo, ct, geo)
            grid.cell_data["gamma"] = f.x.array[f.function_space.dofmap.list[cells, 0]]
            lims = panel_clim if panel_clim is not None else self._clim(grid, "gamma")
            image = self._section_image(grid, "gamma", "viridis", lims, axis, plane, overlays)
            panels.append(_Panel(image, self._math_title(title), "viridis", lims, "γ"))

        shared = None
        if len(panels) > 2 and len({p.clim for p in panels[1:]}) == 1:
            shared = ("viridis", panels[1].clim, "γ")
            panels = panels[:1] + [_Panel(p.image, p.title) for p in panels[1:]]
        return self._compose(panels, filename, ncols=ncols, width=3.4, shared_bar=shared)

    # Titles
    def _geometry_titles(self) -> List[str]:
        gam, eta = self._config.conductivity, self._config.eta
        n = len(eta.centers)
        centers = ", ".join(self._fmt_point(c) for c in eta.centers)
        return [
            f"Condutividade γ\nesfera r={self._fmt(gam.radius)}, "
            f"γ={self._fmt(gam.gamma_in)} dentro e {self._fmt(gam.gamma_out)} fora",
            f"Direção η\n{n} esfera{'s' if n != 1 else ''} r={self._fmt(eta.radius)} em {centers}",
        ]

    def _forward_titles(self, pattern: int) -> List[str]:
        g_top, g_bot = self._config.current.patterns[pattern]
        x0 = self._fmt(self._config.conductivity.center[0])
        return [
            f"Potencial na superfície\ng={g_top:+g} no topo, g={g_bot:+g} na base",
            f"Corte em x = {x0}\ncírculo: borda da esfera",
        ]

    # Image builders
    def _plotter(self, window_size=None) -> pyvista.Plotter:
        p = pyvista.Plotter(off_screen=True, window_size=window_size or self.WINDOW_SIZE)
        p.set_background(self.BG)
        return p

    @classmethod
    def _snap(cls, p: pyvista.Plotter) -> np.ndarray:
        img = p.screenshot(return_img=True)
        p.close()
        return cls._trim(img)

    @staticmethod
    def _trim(img: np.ndarray, margin: int = 12) -> np.ndarray:
        content = np.any(img < 250, axis=2)
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        if rows.size == 0:
            return img
        r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, img.shape[0])
        c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, img.shape[1])
        return img[r0:r1, c0:c1]

    def _geometry_image(self, spheres) -> np.ndarray:
        p = self._plotter()
        p.add_mesh(self._cylinder(), color="#c9d3df", opacity=0.25, smooth_shading=True)
        p.add_mesh(self._cylinder().extract_feature_edges(), color="#5b6b80", line_width=1.5)
        for mesh, color in spheres:
            p.add_mesh(mesh, color=color, smooth_shading=True, specular=0.2)
        p.view_isometric()
        return self._snap(p)

    def _surface_image(self, grid, scalar: str, cmap: str, clim) -> np.ndarray:
        p = self._plotter()
        p.add_mesh(grid.extract_surface(algorithm="dataset_surface"),
                    scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False,
                    smooth_shading=True, ambient=0.55, diffuse=0.5, specular=0.0)
        p.view_isometric()
        return self._snap(p)

    def _section_image(
        self, grid, scalar: str, cmap: str, clim,
        axis: str, plane: float, overlays: Sequence[pyvista.PolyData] = (),
        bar: Optional[str] = None,
    ) -> np.ndarray:
        idx = AXES[axis]
        origin = [0.0, 0.0, 0.0]
        origin[idx] = plane
        eye = [0.0, 0.0, 0.0]
        eye[idx] = plane + self._camera_distance()

        p = self._plotter()
        p.add_mesh(grid.clip(normal=axis, origin=origin),
                    scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False, lighting=False)
        for overlay in overlays:
            if overlay.n_points:
                p.add_mesh(overlay, color="#111111", line_width=2.5, lighting=False)
        p.camera_position = [tuple(eye), tuple(origin), (0.0, 0.0, 1.0)]
        p.enable_parallel_projection()
        p.reset_camera()
        return self._snap(p)

    def _inclusion_section_image(self, grid, scalar: str, cmap: str, clim) -> np.ndarray:
        gam = self._config.conductivity
        plane = float(gam.center[AXES["x"]])
        circle = self._section_circle(gam.center, gam.radius, AXES["x"], plane)
        return self._section_image(grid, scalar, cmap, clim, "x", plane, [circle])

    # Figure assembly
    def _compose(
        self, panels: Sequence[_Panel], filename: str,
        ncols: Optional[int] = None, width: float = 4.8,
        shared_bar: Optional[Tuple[str, Tuple[float, float], str]] = None,
    ) -> Path:
        n = len(panels)
        ncols = n if ncols is None else min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(width * ncols, width * 1.05 * nrows), layout="constrained", squeeze=False,
            )
            axes = axes.ravel()
            for ax in axes:
                ax.axis("off")
            for ax, panel in zip(axes, panels):
                ax.imshow(panel.image)
                ax.set_title(panel.title, fontsize=11)
                if panel.cmap is not None and panel.clim is not None:
                    self._colorbar(fig, ax, panel.cmap, panel.clim, panel.label, shrink=0.8)
            if shared_bar is not None:
                without_bar = [ax for ax, p in zip(axes, panels) if p.cmap is None]
                self._colorbar(fig, without_bar, *shared_bar, shrink=0.5)
            return self._save(fig, filename)

    @staticmethod
    def _colorbar(fig, ax, cmap: str, clim: Tuple[float, float], label: Optional[str], shrink: float) -> None:
        mappable = ScalarMappable(norm=Normalize(*clim), cmap=cmap)
        bar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, shrink=shrink)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=9)
        if label:
            bar.ax.set_title(label, fontsize=11, pad=6)

    @staticmethod
    def _math_title(title: str) -> str:
        match = re.fullmatch(r"γ_(\d+)", title)
        return rf"$\gamma_{{{match.group(1)}}}$" if match else title

    def _save(self, fig, filename: str) -> Path:
        out = self._output_dir / filename
        fig.savefig(str(out), dpi=self.DPI, facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", out)
        return out

    @staticmethod
    def _clim(grid, scalar: str) -> Tuple[float, float]:
        return float(grid[scalar].min()), float(grid[scalar].max())

    @staticmethod
    def _symmetric_clim(grid, scalar: str) -> Tuple[float, float]:
        m = float(np.abs(grid[scalar]).max())
        return -m, m