import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista")

from eit3d.config import ConductivityConfig, EITConfig, EtaConfig, MeshConfig
from eit3d.visualization import (
    BaseRenderer,
    InteractiveRenderer,
    StaticRenderer,
    SurfacePanel,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning:vtkmodules",
)

requires_rendering = pytest.mark.skipif(
    not pyvista.system_supports_plotting(),
    reason="off-screen rendering unavailable",
)


# Design
def test_no_unimplemented_render_contract():
    """The base class no longer forces a render() that subclasses cannot honor."""
    assert not hasattr(BaseRenderer, "render")
    assert not hasattr(StaticRenderer, "render")


def test_renderers_share_base():
    assert issubclass(StaticRenderer, BaseRenderer)
    assert issubclass(InteractiveRenderer, BaseRenderer)


def test_geometry_titles_follow_config(tmp_path):
    cfg = EITConfig(
        conductivity=ConductivityConfig(radius=0.5, gamma_in=3.0),
        eta=EtaConfig(centers=((0.1, 0.2, 0.0),), radius=0.15),
    )
    left, right = StaticRenderer(cfg, tmp_path)._geometry_titles()
    assert "r=0.5" in left and "γ=3" in left
    assert "1 esfera " in right and "r=0.15" in right and "(0.1, 0.2, 0)" in right


def test_forward_titles_follow_pattern(tmp_path):
    cfg = EITConfig()
    title, _ = StaticRenderer(cfg, tmp_path)._forward_titles(pattern=0)
    assert "g=+1 no topo" in title and "g=-1 na base" in title


def test_cylinder_matches_mesh_config(tmp_path):
    cfg = EITConfig(mesh=MeshConfig(radius=0.5, height=3.0))
    xmin, xmax, _, _, zmin, zmax = StaticRenderer(cfg, tmp_path)._cylinder().bounds
    assert np.isclose(xmax - xmin, 1.0, atol=1e-2)
    assert np.isclose(zmax - zmin, 3.0, atol=1e-2)


def test_section_circle_radius():
    circle = BaseRenderer._section_circle((0.0, 0.0, 0.0), 0.5, axis=0, plane=0.3)
    radii = np.linalg.norm(circle.points[:, 1:], axis=1)
    assert np.allclose(radii, 0.4, atol=1e-2)
    assert np.allclose(circle.points[:, 0], 0.3)


def test_section_circle_misses_sphere():
    circle = BaseRenderer._section_circle((0.0, 0.0, 0.0), 0.5, axis=0, plane=0.7)
    assert circle.n_points == 0


# Images
def test_consistency_figure(tmp_path):
    t = 0.9 ** np.arange(60)
    y = 0.27 * t
    fit = np.polyval(
        np.polyfit(np.log10(t[:40]), np.log10(y[:40]), 1), np.log10(t[:40])
    )
    out = StaticRenderer(EITConfig(), tmp_path).render_consistency(y, t, 1.0, fit, 59)
    assert out.exists() and out.stat().st_size > 0


@requires_rendering
def test_static_figures(tmp_path, mesh_data):
    import dolfinx.plot

    _mesh, _, V = mesh_data
    topo, ct, geo = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topo, ct, geo)
    grid["u"] = V.tabulate_dof_coordinates()[:, 2]  # u = z

    renderer = StaticRenderer(EITConfig(), tmp_path)
    outputs = [
        renderer.render_geometry(),
        renderer.render_forward(grid, "u"),
        renderer.render_forward_overview(grid, "u"),
        renderer.render_omega(grid, "u", integral=0.0),
        renderer.render_surfaces([SurfacePanel(grid, "u", "u = z")], "surfaces.png"),
    ]
    for out in outputs:
        assert out.exists() and out.stat().st_size > 0
    assert not list(tmp_path.glob("_tmp_*")), "temporary files left behind"


def test_inversion_figures(tmp_path):
    k = np.arange(201)
    res = 0.01 + 0.15 * np.exp(-k / 20)
    err = 0.7 + 0.05 * np.exp(-k / 30)
    renderer = StaticRenderer(EITConfig(), tmp_path)
    outputs = [
        renderer.render_inversion_history(res, err, 1.0, "history.png", floor=0.02),
        renderer.render_inversion_history(res, err, 1.0, "history_no_floor.png"),
        renderer.render_step_comparison(
            {1.0: res, 0.1: res + 0.02}, "comparison.png", floor=0.02
        ),
    ]
    for out in outputs:
        assert out.exists() and out.stat().st_size > 0


@requires_rendering
def test_inclusions_figure(tmp_path):
    out = StaticRenderer(EITConfig(), tmp_path).render_inclusions(
        ((0.4, 0.0, 0.4), (-0.4, 0.0, -0.4)),
        0.25,
        10.0,
        1.0,
        "inclusions.png",
    )
    assert out.exists() and out.stat().st_size > 0


def _sphere_field(mesh, center, value):
    from eit3d.fields import SpheresField

    return SpheresField(mesh, (center,), 0.4, value, 1.0, name="gamma").build()


@pytest.fixture
def saved_grid(tmp_path, mesh_data):
    from eit3d.visualization.grid import TRUE_FIELD, save_conductivity

    mesh, _, _ = mesh_data
    fields = {
        TRUE_FIELD: _sphere_field(mesh, (0.3, 0.0, 0.3), 10.0),
        "k5": _sphere_field(mesh, (0.3, 0.0, 0.3), 1.5),
        "k10": _sphere_field(mesh, (0.3, 0.0, 0.3), 2.0),
    }
    path = save_conductivity(fields, tmp_path / "sub" / "gamma.vtu")
    return pyvista.read(path)


def test_conductivity_grid_follows_cell_order(mesh_data):
    import dolfinx.fem

    from eit3d.visualization.grid import conductivity_grid

    mesh, _, _ = mesh_data
    x = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    x.interpolate(lambda p: p[0])
    grid = conductivity_grid({"x": x})
    assert np.allclose(grid.cell_data["x"], grid.cell_centers().points[:, 0])


def test_conductivity_grid_rejects_non_dg0(mesh_data):
    import dolfinx.fem

    from eit3d.visualization.grid import conductivity_grid

    _, _, V = mesh_data
    with pytest.raises(ValueError, match="DG0"):
        conductivity_grid({"u": dolfinx.fem.Function(V)})


def test_saved_reconstruction_keeps_fields_in_order(saved_grid):
    from eit3d.visualization.grid import TRUE_FIELD, iteration_fields

    assert TRUE_FIELD in saved_grid.cell_data
    assert iteration_fields(saved_grid) == ["k5", "k10"]
    assert np.isclose(saved_grid.cell_data[TRUE_FIELD].max(), 10.0)


def test_viewer_needs_a_reconstruction(saved_grid):
    from eit3d.visualization import InversionViewer

    only_true = saved_grid.copy()
    for name in ("k5", "k10"):
        only_true.cell_data.remove(name)
    with pytest.raises(ValueError, match="reconstructed"):
        InversionViewer(EITConfig(), only_true, [])


def test_viewer_level_and_isosurface(saved_grid):
    from eit3d.visualization import InversionViewer

    viewer = InversionViewer(EITConfig(), saved_grid, [((0.3, 0.0, 0.3), 0.4)])
    values = viewer._points.point_data["k10"]
    low, high = np.median(values), values.max()
    assert np.isclose(viewer.level("k10", 0.0), low)
    assert np.isclose(viewer.level("k10", 1.0), high)
    surface = viewer.isosurface("k10", 0.5)
    assert surface.n_points > 0
    assert np.linalg.norm(surface.center - np.array([0.3, 0.0, 0.3])) < 0.2


@requires_rendering
def test_viewer_window_and_sliders(saved_grid):
    from eit3d.visualization import InversionViewer

    viewer = InversionViewer(EITConfig(), saved_grid, [((0.3, 0.0, 0.3), 0.4)])
    pl = viewer.plotter(off_screen=True)
    assert viewer._index == 1
    viewer._on_iteration(0.2)
    assert viewer._index == 0
    viewer._on_level(0.8)
    assert np.isclose(viewer._fraction, 0.8)
    assert "iso" in pl.renderers[1].actors
    assert "section" in pl.renderers[2].actors
    pl.close()


def test_viewer_section_keeps_the_far_half(saved_grid):
    from eit3d.visualization import InversionViewer

    viewer = InversionViewer(EITConfig(), saved_grid, [((0.3, 0.0, 0.3), 0.4)])
    section = viewer.section("k10")
    assert np.isclose(section.bounds[2], 0.0, atol=1e-8)
    assert section.bounds[3] > 0.5
    assert section.active_scalars_name == "k10"
