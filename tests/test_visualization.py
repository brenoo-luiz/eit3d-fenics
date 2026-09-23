"""
Tests for the renderers.

Title tests are pure. Image tests render off-screen on a coarse mesh 
and are skipped when PyVista cannot render on this machine.
"""

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from eit3d.config import ConductivityConfig, EITConfig, EtaConfig, MeshConfig  # noqa: E402
from eit3d.visualization import (  # noqa: E402
    BaseRenderer, InteractiveRenderer, StaticRenderer, SurfacePanel,
)

# VTK 9.6 internals trigger a NumPy 2.5 DeprecationWarning
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning:vtkmodules",
)

requires_rendering = pytest.mark.skipif(
    not pyvista.system_supports_plotting(), reason="off-screen rendering unavailable",
)


# Design
def test_no_unimplemented_render_contract():
    """The base class no longer forces a render() that subclasses cannot honor."""
    assert not hasattr(BaseRenderer, "render")
    assert not hasattr(StaticRenderer, "render")


def test_renderers_share_base():
    assert issubclass(StaticRenderer, BaseRenderer)
    assert issubclass(InteractiveRenderer, BaseRenderer)


# Titles and geometry come from the config
def test_geometry_titles_follow_config(tmp_path):
    cfg = EITConfig(
        conductivity=ConductivityConfig(radius=0.5, gamma_in=3.0),
        eta=EtaConfig(centers=((0.1, 0.2, 0.0),), radius=0.15),
    )
    left, right = StaticRenderer(cfg, tmp_path)._geometry_titles()
    assert "r=0.5" in left and "γ=3" in left
    assert "1 sphere " in right and "r=0.15" in right and "(0.1, 0.2, 0)" in right


def test_forward_titles_follow_pattern(tmp_path):
    cfg = EITConfig()
    title, _ = StaticRenderer(cfg, tmp_path)._forward_titles(pattern=0)
    assert "g=+1 top" in title and "g=-1 base" in title


def test_cylinder_matches_mesh_config(tmp_path):
    cfg = EITConfig(mesh=MeshConfig(radius=0.5, height=3.0))
    xmin, xmax, _, _, zmin, zmax = StaticRenderer(cfg, tmp_path)._cylinder().bounds
    assert np.isclose(xmax - xmin, 1.0, atol=1e-2)
    assert np.isclose(zmax - zmin, 3.0, atol=1e-2)


def test_section_circle_radius():
    circle = BaseRenderer._section_circle((0.0, 0.0, 0.0), 0.5, axis=0, plane=0.3)
    radii  = np.linalg.norm(circle.points[:, 1:], axis=1)
    assert np.allclose(radii, 0.4, atol=1e-2)       # sqrt(0.5^2 - 0.3^2)
    assert np.allclose(circle.points[:, 0], 0.3)


def test_section_circle_misses_sphere():
    circle = BaseRenderer._section_circle((0.0, 0.0, 0.0), 0.5, axis=0, plane=0.7)
    assert circle.n_points == 0


# Images
def test_consistency_figure(tmp_path):
    """Matplotlib only"""
    t = 0.9 ** np.arange(60)
    y = 0.27 * t
    fit = np.polyval(np.polyfit(np.log10(t[:40]), np.log10(y[:40]), 1), np.log10(t[:40]))
    out = StaticRenderer(EITConfig(), tmp_path).render_consistency(y, t, 1.0, fit, 59)
    assert out.exists() and out.stat().st_size > 0


@requires_rendering
def test_static_figures(tmp_path, mesh_data):
    import dolfinx.plot

    mesh, _, V = mesh_data
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topo, ct, geo)
    grid["u"] = V.tabulate_dof_coordinates()[:, 2]   # u = z

    renderer = StaticRenderer(EITConfig(), tmp_path)
    outputs  = [
        renderer.render_geometry(),
        renderer.render_forward(grid, "u"),
        renderer.render_forward_overview(grid, "u"),
        renderer.render_omega(grid, "u", integral=0.0),
        renderer.render_surfaces([SurfacePanel(grid, "u", "u = z")], "surfaces.png"),
    ]
    for out in outputs:
        assert out.exists() and out.stat().st_size > 0
    assert not list(tmp_path.glob("_tmp_*")), "temporary files left behind"