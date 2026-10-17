from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pyvista
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

from eit3d.config import OUTPUTS_DIR, EITConfig
from eit3d.visualization.base import BaseRenderer

logger = logging.getLogger(__name__)

AXES = {"x": 0, "y": 1, "z": 2}
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

INK = "#1d1d1f"
INK_SOFT = "#5f6368"
INK_FAINT = "#9aa0a6"
GRID = "#e8e8e8"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#c4c7cc"
GLASS = "#cfd8e3"
RIM = "#8a9099"

FONT = (
    "Inter"
    if any(f.name == "Inter" for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)

STYLE = {
    "font.family": FONT,
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "axes.labelsize": 10.5,
    "axes.edgecolor": INK_FAINT,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "ytick.left": False,
    "xtick.major.size": 3,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.cal": FONT,
    "figure.facecolor": "white",
}


@dataclass(frozen=True)
class SurfacePanel:
    grid: pyvista.UnstructuredGrid
    scalar: str
    title: str
    cmap: str = "coolwarm"
    clim: tuple[float, float] | None = None
    bar: str | None = None


@dataclass(frozen=True)
class _Panel:
    image: np.ndarray
    title: str
    cmap: str | None = None
    clim: tuple[float, float] | None = None
    label: str | None = None


class StaticRenderer(BaseRenderer):
    BG = "white"
    WINDOW_SIZE = (1100, 1100)
    DPI = 300
    CAMERA: ClassVar[list[tuple[float, float, float]]] = [
        (5.2, -6.6, 2.6),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    def __init__(self, config: EITConfig, output_dir: Path = OUTPUTS_DIR) -> None:
        super().__init__(config)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # Public figures
    def render_mesh(
        self, grid: pyvista.UnstructuredGrid, filename: str = "step1_mesh.png"
    ) -> Path:
        p = self._plotter()
        p.add_mesh(
            grid,
            color="#f4f6f9",
            show_edges=True,
            edge_color="#8fb3e3",
            line_width=0.3,
            ambient=0.6,
            diffuse=0.45,
            specular=0.0,
        )
        self._frame(p)
        mesh = self._config.mesh
        return self._compose(
            [
                _Panel(
                    self._snap(p),
                    f"Malha\nraio {self._fmt(mesh.radius)}, altura {self._fmt(mesh.height)}",
                )
            ],
            filename,
            width=6,
        )

    def render_geometry(self, filename: str = "step3_a_geometry.png") -> Path:
        gam, eta = self._config.conductivity, self._config.eta
        left, right = self._geometry_titles()
        img_gamma = self._geometry_image([(self._sphere(gam.center, gam.radius), BLUE)])
        img_eta = self._geometry_image(
            [(self._sphere(c, eta.radius), ORANGE) for c in eta.centers]
        )
        return self._compose(
            [_Panel(img_gamma, left), _Panel(img_eta, right)], filename
        )

    def render_inclusions(
        self,
        centers: Sequence[Sequence[float]],
        radius: float,
        inside: float,
        outside: float,
        filename: str,
        title: str = "Condutividade verdadeira γ⁺",
    ) -> Path:
        image = self._geometry_image([(self._sphere(c, radius), BLUE) for c in centers])
        with plt.rc_context(STYLE):
            fig = plt.figure(figsize=(7, 7.8))
            self._header(fig, title)
            ax = fig.add_axes([0.07, 0.08, 0.86, 0.82])
            ax.imshow(image)
            ax.axis("off")
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=11,
                    markerfacecolor=BLUE,
                    markeredgewidth=0,
                    label=f"esferas  γ = {self._fmt(inside)}",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=11,
                    markerfacecolor="#e3e7ec",
                    markeredgecolor=INK_FAINT,
                    label=f"cilindro  γ = {self._fmt(outside)}",
                ),
            ]
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.005),
                ncol=2,
                fontsize=10,
                labelcolor=INK_SOFT,
                handletextpad=0.4,
                columnspacing=2.2,
            )
            return self._save(fig, filename)

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
                _Panel(
                    self._surface_image(grid, scalar, "coolwarm", clim),
                    left,
                    "coolwarm",
                    clim,
                    r"$u_\gamma$",
                ),
                _Panel(
                    self._inclusion_section_image(grid, scalar, "coolwarm", clim),
                    right,
                    "coolwarm",
                    clim,
                    r"$u_\gamma$",
                ),
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
        img_geo = self._geometry_image([(self._sphere(gam.center, gam.radius), BLUE)])
        return self._compose(
            [
                _Panel(
                    img_geo,
                    f"Condutividade\nγ = {self._fmt(gam.gamma_in)} na esfera, "
                    f"{self._fmt(gam.gamma_out)} fora",
                ),
                _Panel(
                    self._surface_image(grid, scalar, "coolwarm", clim),
                    "Potencial u\nsuperfície",
                    "coolwarm",
                    clim,
                    "u",
                ),
                _Panel(
                    self._inclusion_section_image(grid, scalar, "coolwarm", clim),
                    f"Potencial u\ncorte em x = {self._fmt(gam.center[0])}",
                    "coolwarm",
                    clim,
                    "u",
                ),
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
            self._section_circle(c, self._config.eta.radius, AXES["y"], plane)
            for c in self._config.eta.centers
        ]
        return self._compose(
            [
                _Panel(
                    self._surface_image(grid, scalar, "RdBu_r", clim),
                    f"Derivada ω\nsuperfície, ∫ω ds = {integral:.1e}",
                    "RdBu_r",
                    clim,
                    "ω",
                ),
                _Panel(
                    self._section_image(
                        grid,
                        scalar,
                        "RdBu_r",
                        clim,
                        axis="y",
                        plane=plane,
                        overlays=overlays,
                    ),
                    f"Derivada ω\ncorte em y = {self._fmt(plane)}, círculos: esferas de η",
                    "RdBu_r",
                    clim,
                    "ω",
                ),
            ],
            filename,
        )

    def render_surfaces(self, panels: Sequence[SurfacePanel], filename: str) -> Path:
        images = []
        for pn in panels:
            clim = pn.clim if pn.clim is not None else self._clim(pn.grid, pn.scalar)
            images.append(
                _Panel(
                    self._surface_image(pn.grid, pn.scalar, pn.cmap, clim),
                    pn.title,
                    pn.cmap,
                    clim,
                    pn.bar if pn.bar is not None else pn.scalar,
                )
            )
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
        n_fit = len(fit_line)
        theory = t_vals * y_vals[0] / t_vals[0]

        with plt.rc_context(STYLE):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
            fig.subplots_adjust(
                left=0.08, right=0.9, top=0.78, bottom=0.14, wspace=0.32
            )
            self._header(
                fig,
                f"Teste da derivada  ·  t = {self._fmt(self._config.consistency.base)}ⁿ",
            )

            ax1.semilogy(
                ns,
                y_vals,
                color=BLUE,
                linewidth=1.6,
                marker="o",
                markersize=2.5,
                zorder=3,
            )
            ax1.plot(
                idx_min,
                y_vals[idx_min],
                "o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=1.2,
                zorder=4,
            )
            ax1.annotate(
                f"mínimo  {y_vals[idx_min]:.1e}",
                (idx_min, y_vals[idx_min]),
                xytext=(-10, -10),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=9,
                color=INK,
            )
            self._panel(ax1, "Erro por perturbação", "yₙ")
            ax1.set_xlabel("n")
            ax1.set_xlim(0, ns[-1])
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.set_ylim(y_vals.min() * 0.08, y_vals.max() * 3)
            ax1.yaxis.set_minor_locator(NullLocator())

            ax2.grid(True, axis="both")
            ax2.loglog(
                t_vals,
                theory,
                color=INK_FAINT,
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
            ax2.loglog(
                t_vals,
                y_vals,
                "o",
                color=BLUE,
                markersize=3,
                markeredgewidth=0,
                alpha=0.85,
                zorder=3,
            )
            ax2.loglog(t_vals[:n_fit], 10**fit_line, color=INK, linewidth=1.3, zorder=4)
            ax2.invert_xaxis()
            mid = n_fit // 2
            ax2.annotate(
                f"inclinação {slope:.3f}",
                (t_vals[mid], 10 ** fit_line[mid]),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9.5,
                color=INK,
                fontweight="bold",
            )
            ax2.annotate(
                "teoria: inclinação 1",
                (t_vals[-1], theory[-1]),
                xytext=(-4, -8),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=8.5,
                color=INK_SOFT,
            )
            ax2.set_ylim(min(y_vals.min(), theory.min()) * 0.08, y_vals.max() * 3)
            ax2.xaxis.set_minor_locator(NullLocator())
            ax2.yaxis.set_minor_locator(NullLocator())
            self._panel(ax2, "Taxa de convergência", "yₙ")
            ax2.set_xlabel("t")
            return self._save(fig, filename)

    def render_inversion_history(
        self,
        residuals: np.ndarray,
        errors: np.ndarray,
        step: float,
        filename: str,
        floor: float | None = None,
        title: str | None = None,
    ) -> Path:
        res = np.asarray(residuals)
        err = np.asarray(errors)
        ks = np.arange(len(res))
        with plt.rc_context(STYLE):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
            fig.subplots_adjust(
                left=0.08, right=0.92, top=0.78, bottom=0.14, wspace=0.3
            )
            self._header(
                fig,
                title if title is not None else f"Método do gradiente  ·  λ = {step:g}",
            )

            ax1.semilogy(ks, res, color=BLUE, linewidth=2, zorder=3)
            self._end_label(ax1, ks[-1], res[-1], f"{res[-1]:.3g}", BLUE, bold=True)
            self._log_axis(ax1, res if floor is None else np.append(res, floor))
            if floor is not None:
                self._floor(ax1, ks[-1], floor, above=res[-1] < floor)
                crossing = np.flatnonzero(res < floor)
                if crossing.size:
                    self._mark_crossing(ax1, int(crossing[0]), int(ks[-1]), floor)
            self._panel(ax1, "Resíduo", "‖F(γₖ) − ũ‖")

            ax2.plot(ks, err, color=ORANGE, linewidth=2, zorder=3)
            self._end_label(ax2, ks[-1], err[-1], f"{err[-1]:.3f}", ORANGE, bold=True)
            self._panel(ax2, "Erro relativo", "‖γ⁺ − γₖ‖ / ‖γ⁺‖")

            for ax in (ax1, ax2):
                ax.set_xlim(0, max(ks[-1], 1))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel("iteração k")
            return self._save(fig, filename)

    def render_step_comparison(
        self,
        residuals: dict[float, np.ndarray],
        filename: str,
        floor: float | None = None,
    ) -> Path:
        highlight = max(residuals)
        k_max = max(len(r) for r in residuals.values()) - 1
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(9, 5.4))
            fig.subplots_adjust(left=0.11, right=0.86, top=0.86, bottom=0.12)
            self._header(fig, "Resíduo para diferentes passos λ")
            values = []
            for step in sorted(residuals):
                res = np.asarray(residuals[step])
                main = step == highlight
                ax.semilogy(
                    np.arange(len(res)),
                    res,
                    color=BLUE if main else GREY,
                    linewidth=2.4 if main else 1.6,
                    zorder=3 if main else 2,
                )
                values.append(res)
            if floor is not None:
                self._floor(ax, k_max, floor)
                values.append([floor])
            self._log_axis(ax, np.concatenate(values))
            ax.set_xlim(0, k_max)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            labels = [(step, residuals[step][-1]) for step in sorted(residuals)]
            for (step, _), height in zip(
                labels, self._spread_labels(ax, [y for _, y in labels])
            ):
                main = step == highlight
                ax.text(
                    k_max,
                    height,
                    f"  λ = {step:g}",
                    transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="left",
                    va="center",
                    fontsize=9.5,
                    color=BLUE if main else INK_SOFT,
                    fontweight="bold" if main else "normal",
                )
            ax.set_xlabel("iteração k")
            ax.set_ylabel("‖F(γₖ) − ũ‖")
            return self._save(fig, filename)

    def render_conductivity_sections(
        self,
        fields: Sequence[tuple[str, object]],
        filename: str,
        spheres: Sequence[tuple[Sequence[float], float]] = (),
        axis: str = "y",
        plane: float = 0.0,
        clim: tuple[float, float] | None = None,
        ncols: int = 4,
        clims: Sequence[tuple[float, float] | None] | None = None,
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
            image = self._section_image(
                grid, "gamma", "viridis", lims, axis, plane, overlays
            )
            panels.append(_Panel(image, self._math_title(title), "viridis", lims, "γ"))

        shared = None
        if len(panels) > 2 and len({p.clim for p in panels[1:]}) == 1:
            shared = ("viridis", panels[1].clim, "γ")
            panels = panels[:1] + [_Panel(p.image, p.title) for p in panels[1:]]
        return self._compose(
            panels, filename, ncols=ncols, width=3.4, shared_bar=shared
        )

    # Titles
    def _geometry_titles(self) -> list[str]:
        gam, eta = self._config.conductivity, self._config.eta
        n = len(eta.centers)
        centers = ", ".join(self._fmt_point(c) for c in eta.centers)
        return [
            (
                f"Condutividade γ\nesfera r={self._fmt(gam.radius)}, "
                f"γ={self._fmt(gam.gamma_in)} dentro e {self._fmt(gam.gamma_out)} fora"
            ),
            f"Direção η\n{n} esfera{'s' if n != 1 else ''} r={self._fmt(eta.radius)} em {centers}",
        ]

    def _forward_titles(self, pattern: int) -> list[str]:
        g_top, g_bot = self._config.current.patterns[pattern]
        x0 = self._fmt(self._config.conductivity.center[0])
        return [
            f"Potencial na superfície\ng={g_top:+g} no topo, g={g_bot:+g} na base",
            f"Corte em x = {x0}\ncírculo: borda da esfera",
        ]

    # Line plot helpers
    @staticmethod
    def _header(fig, title: str) -> None:
        fig.text(
            0.04,
            0.95,
            title,
            fontsize=14,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )

    @staticmethod
    def _panel(ax, title: str, ylabel: str) -> None:
        ax.set_title(
            title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10
        )
        ax.set_ylabel(ylabel)

    @staticmethod
    def _end_label(
        ax, x: float, y: float, text: str, color: str, bold: bool = False
    ) -> None:
        ax.annotate(
            text,
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9.5,
            color=color,
            fontweight="bold" if bold else "normal",
            annotation_clip=False,
        )

    @staticmethod
    def _spread_labels(ax, values: Sequence[float], gap: float = 0.05) -> list[float]:
        low, high = (np.log10(v) for v in ax.get_ylim())
        heights = [(np.log10(v) - low) / (high - low) for v in values]
        order = np.argsort(heights)
        placed = list(heights)
        for prev, cur in itertools.pairwise(order):
            placed[cur] = max(placed[cur], placed[prev] + gap)
        return placed

    @staticmethod
    def _floor(ax, x_end: float, floor: float, above: bool = False) -> None:
        ax.axhline(
            floor, color=INK_FAINT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1
        )
        y, va = (floor * 1.06, "bottom") if above else (floor * 0.94, "top")
        ax.text(
            x_end,
            y,
            "resíduo da solução exata γ⁺",
            fontsize=8.5,
            color=INK_SOFT,
            ha="right",
            va=va,
        )

    @staticmethod
    def _mark_crossing(ax, k: int, k_end: int, floor: float) -> None:
        ax.fill_between(
            [k, k_end],
            ax.get_ylim()[0],
            floor,
            color=BLUE,
            alpha=0.06,
            linewidth=0,
            zorder=0,
        )
        offset, rad = ((45, 55), 0.25) if k < 0.6 * k_end else ((-90, 55), -0.25)
        ax.annotate(
            f"k = {k}",
            xy=(k, floor),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=INK,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "linewidth": 0.9,
                "connectionstyle": f"arc3,rad={rad}",
                "shrinkB": 4,
            },
        )

    @staticmethod
    def _log_axis(ax, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        ax.set_ylim(values.min() * 0.7, values.max() * 1.3)
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    # Image builders
    def _plotter(self, window_size=None) -> pyvista.Plotter:
        p = pyvista.Plotter(
            off_screen=True, window_size=window_size or self.WINDOW_SIZE
        )
        p.set_background(self.BG)
        return p

    def _frame(self, p: pyvista.Plotter) -> None:
        p.camera_position = self.CAMERA
        p.camera.view_angle = 20
        p.reset_camera()

    @classmethod
    def _snap(cls, p: pyvista.Plotter) -> np.ndarray:
        p.enable_anti_aliasing("ssaa")
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
        mesh = self._config.mesh
        p = self._plotter()
        p.add_mesh(
            self._cylinder(),
            color=GLASS,
            opacity=0.10,
            smooth_shading=True,
            specular=0.2,
        )
        for z in (-mesh.height / 2, mesh.height / 2):
            rim = pyvista.Polygon(
                center=(0.0, 0.0, z), radius=mesh.radius, n_sides=200, fill=False
            )
            p.add_mesh(rim, color=RIM, line_width=2.5)
        for sphere, color in spheres:
            p.add_mesh(
                sphere,
                color=color,
                smooth_shading=True,
                specular=0.5,
                specular_power=30,
            )
        self._frame(p)
        return self._snap(p)

    def _surface_image(self, grid, scalar: str, cmap: str, clim) -> np.ndarray:
        p = self._plotter()
        p.add_mesh(
            grid.extract_surface(algorithm="dataset_surface"),
            scalars=scalar,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            smooth_shading=True,
            ambient=0.55,
            diffuse=0.5,
            specular=0.0,
        )
        self._frame(p)
        return self._snap(p)

    def _section_image(
        self,
        grid,
        scalar: str,
        cmap: str,
        clim,
        axis: str,
        plane: float,
        overlays: Sequence[pyvista.PolyData] = (),
        bar: str | None = None,
    ) -> np.ndarray:
        idx = AXES[axis]
        origin = [0.0, 0.0, 0.0]
        origin[idx] = plane
        eye = [0.0, 0.0, 0.0]
        eye[idx] = plane + self._camera_distance()

        p = self._plotter()
        p.add_mesh(
            grid.clip(normal=axis, origin=origin),
            scalars=scalar,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            lighting=False,
        )
        for overlay in overlays:
            if overlay.n_points:
                p.add_mesh(overlay, color="#111111", line_width=2.5, lighting=False)
        p.camera_position = [tuple(eye), tuple(origin), (0.0, 0.0, 1.0)]
        p.enable_parallel_projection()
        p.reset_camera()
        return self._snap(p)

    def _inclusion_section_image(
        self, grid, scalar: str, cmap: str, clim
    ) -> np.ndarray:
        gam = self._config.conductivity
        plane = float(gam.center[AXES["x"]])
        circle = self._section_circle(gam.center, gam.radius, AXES["x"], plane)
        return self._section_image(grid, scalar, cmap, clim, "x", plane, [circle])

    # Figure assembly
    def _compose(
        self,
        panels: Sequence[_Panel],
        filename: str,
        ncols: int | None = None,
        width: float = 4.8,
        shared_bar: tuple[str, tuple[float, float], str] | None = None,
    ) -> Path:
        n = len(panels)
        ncols = n if ncols is None else min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(width * ncols, width * 1.1 * nrows),
                layout="constrained",
                squeeze=False,
            )
            axes = axes.ravel()
            for ax in axes:
                ax.axis("off")
            ratio = max(p.image.shape[0] / p.image.shape[1] for p in panels)
            for ax, panel in zip(axes, panels):
                ax.imshow(self._pad_to_ratio(panel.image, ratio))
                self._panel_title(ax, panel.title)
                if panel.cmap is not None and panel.clim is not None:
                    self._colorbar(
                        fig, ax, panel.cmap, panel.clim, panel.label, shrink=0.7
                    )
            if shared_bar is not None:
                without_bar = [ax for ax, p in zip(axes, panels) if p.cmap is None]
                self._colorbar(fig, without_bar, *shared_bar, shrink=0.5)
            return self._save(fig, filename)

    @staticmethod
    def _pad_to_ratio(img: np.ndarray, ratio: float) -> np.ndarray:
        missing = round(ratio * img.shape[1]) - img.shape[0]
        if missing <= 0:
            return img
        pad = np.full((missing, img.shape[1], img.shape[2]), 255, dtype=img.dtype)
        return np.vstack([img, pad])

    @staticmethod
    def _panel_title(ax, title: str) -> None:
        main, _, detail = title.partition("\n")
        ax.set_title(
            main,
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=18 if detail else 8,
        )
        if detail:
            ax.text(
                0.0,
                1.0,
                detail,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color=INK_SOFT,
            )

    @staticmethod
    def _colorbar(
        fig,
        ax,
        cmap: str,
        clim: tuple[float, float],
        label: str | None,
        shrink: float,
    ) -> None:
        mappable = ScalarMappable(norm=Normalize(*clim), cmap=cmap)
        bar = fig.colorbar(
            mappable, ax=ax, fraction=0.04, pad=0.02, shrink=shrink, aspect=30
        )
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8.5, colors=INK_SOFT, length=0)
        if label:
            bar.ax.set_title(label, fontsize=10, color=INK_SOFT, pad=6)

    @staticmethod
    def _math_title(title: str) -> str:
        match = re.fullmatch(r"γ_(\d+)", title)
        return "γ" + match.group(1).translate(SUBSCRIPTS) if match else title

    def _save(self, fig, filename: str) -> Path:
        out = self._output_dir / filename
        fig.savefig(str(out), dpi=self.DPI, facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", out)
        return out

    @staticmethod
    def _clim(grid, scalar: str) -> tuple[float, float]:
        return float(grid[scalar].min()), float(grid[scalar].max())

    @staticmethod
    def _symmetric_clim(grid, scalar: str) -> tuple[float, float]:
        m = float(np.abs(grid[scalar]).max())
        return -m, m
