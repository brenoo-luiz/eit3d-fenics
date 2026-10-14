import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")

import basix.ufl
import dolfinx
import dolfinx.fem
import ufl
from mpi4py import MPI

from eit3d.config import EtaConfig, MeshConfig, SolverConfig
from eit3d.currents import cosine_currents, lateral_measure
from eit3d.fields import DirectionalField, SpheresField
from eit3d.inverse import (
    GradientMethod, InexactNewton, LinearizedOperator, data_from, project_dg0, transfer,
)
from eit3d.mesh.cylinder import CylinderMesh
from eit3d.solvers import MultiDerivativeSolver

COMM = MPI.COMM_WORLD
SOLVER = SolverConfig()
KS = range(1, 9)
CENTERS = ((0.4, 0.0, 0.4), (-0.4, 0.0, -0.4))


def integrate(expr) -> float:
    return COMM.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)


def p2_space(mesh):
    return dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", "tetrahedron", degree=2, shape=()))


@pytest.fixture(scope="module")
def fine(tmp_path_factory):
    cyl = CylinderMesh(MeshConfig(size_max=0.15, size_min=0.08), comm=COMM,
                        cache_dir=tmp_path_factory.mktemp("fine"))
    mesh, facet_tags = cyl.get()
    return mesh, facet_tags, p2_space(mesh)


@pytest.fixture(scope="module")
def problem(mesh_data, fine):
    mesh_f, tags_f, V_f = fine
    gamma_f = SpheresField(mesh_f, CENTERS, 0.25, 10.0, 1.0).build()
    data_f = data_from(mesh_f, V_f, gamma_f, cosine_currents(mesh_f, KS), lateral_measure(mesh_f, tags_f), SOLVER, COMM)

    mesh, tags, V = mesh_data
    data = [transfer(d, V) for d in data_f]
    truth = SpheresField(mesh, CENTERS, 0.25, 10.0, 1.0).build()
    method = GradientMethod(mesh, V, cosine_currents(mesh, KS), lateral_measure(mesh, tags), data, SOLVER, truth, COMM)
    newton = InexactNewton(mesh, V, cosine_currents(mesh, KS), lateral_measure(mesh, tags), data, SOLVER, truth, COMM)
    gamma0 = SpheresField(mesh, (), 0.25, 1.0, 1.0).build()
    return dict(mesh=mesh, tags=tags, V=V, method=method, newton=newton, gamma0=gamma0, truth=truth)


def test_spheres_field_values(mesh_data):
    mesh, _, _ = mesh_data
    f = SpheresField(mesh, CENTERS, 0.25, 10.0, 1.0).build()
    assert set(np.unique(f.x.array).tolist()) == {1.0, 10.0}
    with pytest.raises(ValueError):
        SpheresField(mesh, CENTERS, -1.0, 10.0, 1.0)


def test_transfer_is_exact_for_quadratics(mesh_data, fine):
    _, _, V = mesh_data
    _, _, V_f = fine
    f = dolfinx.fem.Function(V_f)
    f.interpolate(lambda x: x[0] ** 2 - 2 * x[1] * x[2] + x[2])
    exact = dolfinx.fem.Function(V)
    exact.interpolate(lambda x: x[0] ** 2 - 2 * x[1] * x[2] + x[2])
    assert np.abs(transfer(f, V).x.array - exact.x.array).max() < 1e-10


def test_project_dg0_preserves_integrals(problem):
    mesh, V = problem["mesh"], problem["V"]
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: 1 + x[0] ** 2 + x[2])
    p = project_dg0(f, problem["gamma0"].function_space)
    assert np.isclose(integrate(p * ufl.dx), integrate(f * ufl.dx), rtol=1e-12)
    sigma = DirectionalField(mesh, EtaConfig()).build()
    assert np.isclose(integrate(p * sigma * ufl.dx), integrate(f * sigma * ufl.dx), rtol=1e-12)


def test_gradient_matches_finite_difference(problem):
    method, gamma0, mesh = problem["method"], problem["gamma0"], problem["mesh"]
    sigma = DirectionalField(mesh, EtaConfig()).build()
    u0 = method.forward(gamma0)
    phi0 = 0.5 * method.residual(u0) ** 2
    dphi = integrate(method.gradient(gamma0, u0) * sigma * ufl.dx)

    errors = []
    for t in (1e-3, 1e-4):
        gamma_t = gamma0.copy()
        gamma_t.x.array[:] += t * sigma.x.array
        errors.append(abs((method.objective(gamma_t) - phi0) / t - dphi) / abs(dphi))
    assert errors[1] < 1e-3
    assert 5 < errors[0] / errors[1] < 20


def test_small_step_decreases_objective(problem):
    method, gamma0 = problem["method"], problem["gamma0"]
    u0 = method.forward(gamma0)
    gamma_1 = gamma0.copy()
    gamma_1.x.array[:] -= 0.1 * method.gradient(gamma0, u0).x.array
    assert method.objective(gamma_1) < 0.5 * method.residual(u0) ** 2


def test_true_conductivity_is_stationary_without_model_error(problem):
    mesh, tags, V, truth = problem["mesh"], problem["tags"], problem["V"], problem["truth"]
    currents = cosine_currents(mesh, KS)
    ds_g = lateral_measure(mesh, tags)
    data = data_from(mesh, V, truth, currents, ds_g, SOLVER, COMM)
    method = GradientMethod(mesh, V, currents, ds_g, data, SOLVER, truth, COMM)
    u = method.forward(truth)
    assert method.residual(u) < 1e-9
    assert np.abs(method.gradient(truth, u).x.array).max() < 1e-9
    assert method.error(truth) == 0.0


def test_run_records_history_and_snapshots(problem):
    result = problem["method"].run(problem["gamma0"], 0.1, 4, snapshot_every=2)
    assert [r.k for r in result.history] == [0, 1, 2, 3, 4]
    assert sorted(result.snapshots) == [2, 4]
    assert result.failure is None
    assert result.residuals[-1] < result.residuals[0]
    assert np.all(problem["gamma0"].x.array == 1.0)


def test_large_step_is_stopped(problem):
    result = problem["method"].run(problem["gamma0"], 100.0, 5)
    assert result.failure is not None
    assert result.failure.minimum <= 0


def test_invalid_arguments(problem):
    with pytest.raises(ValueError):
        problem["method"].run(problem["gamma0"], -1.0, 1)
    mesh, tags, V = problem["mesh"], problem["tags"], problem["V"]
    with pytest.raises(ValueError):
        GradientMethod(mesh, V, cosine_currents(mesh, KS), lateral_measure(mesh, tags), [], SOLVER)


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_inverse_figures(problem, tmp_path):
    pyvista = pytest.importorskip("pyvista")
    if not pyvista.system_supports_plotting():
        pytest.skip("off-screen rendering unavailable")
    from eit3d.config import EITConfig
    from eit3d.visualization.static import StaticRenderer

    result = problem["method"].run(problem["gamma0"], 0.1, 2, snapshot_every=1)
    renderer = StaticRenderer(EITConfig(), tmp_path)
    fields = [("true", problem["truth"])] + [(f"k={k}", g) for k, g in result.snapshots.items()]
    outputs = [
        renderer.render_inversion_history(result.residuals, result.errors, 0.1, "conv.png", floor=1e-2),
        renderer.render_conductivity_sections(fields, "sections.png", spheres=[(c, 0.25) for c in CENTERS], ncols=2),
        renderer.render_conductivity_sections(
            fields, "sections_shared.png", clims=[(1.0, 10.0)] + [(0.9, 1.1)] * (len(fields) - 1),
        ),
    ]
    for out in outputs:
        assert out.exists() and out.stat().st_size > 0


# Inexact Newton
@pytest.fixture(scope="module")
def linearized(problem):
    mesh, V, gamma0 = problem["mesh"], problem["V"], problem["gamma0"]
    u0 = problem["method"].forward(gamma0)
    op = LinearizedOperator(mesh, V, gamma0, u0, SOLVER, COMM)
    yield op, u0
    op.close()


def boundary_inner(fs, gs, mesh) -> float:
    ds = ufl.Measure("ds", domain=mesh)
    return sum(integrate(f * g * ds) for f, g in zip(fs, gs))


def test_linearized_apply_matches_derivative_solver(problem, linearized):
    op, u0 = linearized
    mesh, V, gamma0 = problem["mesh"], problem["V"], problem["gamma0"]
    sigma = DirectionalField(mesh, EtaConfig()).build()
    expected = MultiDerivativeSolver(mesh, V, gamma0, sigma, u0, SOLVER, COMM).solve_all()
    for w, w_ref in zip(op.apply(sigma), expected):
        assert np.abs(w.x.array - w_ref.x.array).max() < 1e-8


def test_linearized_adjoint_relation(problem, linearized):
    op, u0 = linearized
    mesh, V = problem["mesh"], problem["V"]
    sigma = DirectionalField(mesh, EtaConfig()).build()
    hs = []
    for j in range(len(u0)):
        h = dolfinx.fem.Function(V)
        h.interpolate(lambda x, j=j: np.cos((j + 1) * x[0]) + x[2] ** 2)
        hs.append(h)
    lhs = integrate(op.adjoint([h.x.array for h in hs]) * sigma * ufl.dx)
    centered = []
    for h in hs:
        c = h.copy()
        c.x.array[:] -= integrate(h * ufl.ds(domain=mesh)) / integrate(1.0 * ufl.ds(domain=mesh))
        centered.append(c)
    rhs = boundary_inner(centered, op.apply(sigma), mesh)
    assert abs(lhs - rhs) / abs(rhs) < 1e-8


def test_newton_decreases_residual_and_records_inner_iterations(problem):
    result = problem["newton"].run(problem["gamma0"], 1.0, 2, mu=0.9, n_inner=20, snapshot_every=1)
    assert [r.k for r in result.history] == [0, 1, 2]
    assert result.failure is None
    assert result.residuals[1] < result.residuals[0]
    assert result.residuals[2] < result.residuals[1]
    assert len(result.inner_iterations) == 2
    assert np.all((result.inner_iterations >= 1) & (result.inner_iterations <= 20))
    assert sorted(result.snapshots) == [1, 2]
    assert np.all(problem["gamma0"].x.array == 1.0)


def test_newton_inner_loop_respects_limit(problem):
    result = problem["newton"].run(problem["gamma0"], 1e-6, 1, mu=0.9, n_inner=3)
    assert list(result.inner_iterations) == [3]


def test_newton_invalid_arguments(problem):
    newton, gamma0 = problem["newton"], problem["gamma0"]
    for kwargs in (dict(step=-1.0), dict(step=1.0, mu=1.0), dict(step=1.0, mu=0.0), dict(step=1.0, n_inner=0)):
        with pytest.raises(ValueError):
            newton.run(gamma0, n_outer=1, **kwargs)


def test_newton_with_one_inner_iteration_is_a_gradient_step(problem):
    newton = problem["newton"].run(problem["gamma0"], 0.5, 1, mu=0.5, n_inner=1)
    gradient = problem["method"].run(problem["gamma0"], 0.5, 1)
    assert list(newton.inner_iterations) == [1]
    assert np.isclose(newton.residuals[1], gradient.residuals[1], rtol=1e-8)


def test_benchmark_conductivity_and_initial_guess(mesh_data):
    from eit3d.benchmark import TwoSpheres

    mesh, _, _ = mesh_data
    bench = TwoSpheres()
    assert set(np.unique(bench.conductivity(mesh).x.array).tolist()) == {1.0, 10.0}
    assert np.all(bench.initial_guess(mesh).x.array == 1.0)
    assert len(bench.currents(mesh)) == 8
    assert bench.spheres == [(c, 0.25) for c in CENTERS]