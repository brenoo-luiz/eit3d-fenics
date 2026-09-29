import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")

import dolfinx  # noqa: E402
import dolfinx.fem  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

from eit3d.config import ConductivityConfig, EtaConfig, SolverConfig  # noqa: E402
from eit3d.currents import cosine_currents, lateral_measure  # noqa: E402
from eit3d.fields.conductivity import ConductivityField, DirectionalField  # noqa: E402
from eit3d.solvers import (  # noqa: E402
    MultiAdjointSolver, MultiDerivativeSolver, MultiForwardSolver,
)

COMM   = MPI.COMM_WORLD
SOLVER = SolverConfig()
KS     = range(1, 9)


def integrate(expr) -> float:
    return COMM.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM)


@pytest.fixture(scope="module")
def setup(mesh_data):
    mesh, facet_tags, V = mesh_data
    gamma    = ConductivityField(mesh, ConductivityConfig()).build()
    sigma    = DirectionalField(mesh, EtaConfig()).build()
    ds_lat   = lateral_measure(mesh, facet_tags)
    currents = cosine_currents(mesh, KS)
    u_list   = MultiForwardSolver(mesh, V, gamma, currents, ds_lat, SOLVER, COMM).solve_all()
    w_list   = MultiDerivativeSolver(mesh, V, gamma, sigma, u_list, SOLVER, COMM).solve_all()
    return dict(mesh=mesh, V=V, gamma=gamma, sigma=sigma, ds_lat=ds_lat,
                currents=currents, u_list=u_list, w_list=w_list)


def test_currents_are_compatible(setup):
    for g in setup["currents"]:
        assert abs(integrate(g * setup["ds_lat"])) < 1e-2 * integrate(abs(g) * setup["ds_lat"])


def test_currents_are_linearly_independent(setup):
    cur, ds_lat = setup["currents"], setup["ds_lat"]
    gram = np.array([[integrate(gi * gj * ds_lat) for gj in cur] for gi in cur])
    assert np.linalg.matrix_rank(gram) == len(cur)
    assert np.linalg.cond(gram) < 1.1


def test_shared_matrix_matches_separate_solves(setup):
    mesh, V, gamma, ds_lat = setup["mesh"], setup["V"], setup["gamma"], setup["ds_lat"]
    for g, u in zip(setup["currents"][:3], setup["u_list"][:3]):
        single = MultiForwardSolver(mesh, V, gamma, [g], ds_lat, SOLVER, COMM).solve_all()[0]
        assert np.allclose(single.x.array, u.x.array, atol=1e-9)


def test_solutions_have_zero_boundary_mean(setup):
    ds = ufl.Measure("ds", domain=setup["mesh"])
    for f in setup["u_list"] + setup["w_list"]:
        assert abs(integrate(f * ds)) < 1e-10


def test_derivative_first_order_consistency(setup):
    mesh, V, gamma, sigma = setup["mesh"], setup["V"], setup["gamma"], setup["sigma"]
    ds      = ufl.Measure("ds", domain=mesh)
    norm_w  = np.sqrt(sum(integrate(w ** 2 * ds) for w in setup["w_list"]))
    gamma_t = dolfinx.fem.Function(gamma.function_space)
    diff    = dolfinx.fem.Function(V)
    t_vals  = np.array([1e-2, 1e-3, 1e-4])
    y_vals  = np.empty(len(t_vals))

    for i, t in enumerate(t_vals):
        gamma_t.x.array[:] = gamma.x.array + t * sigma.x.array
        gamma_t.x.scatter_forward()
        u_t = MultiForwardSolver(mesh, V, gamma_t, setup["currents"], setup["ds_lat"], SOLVER, COMM).solve_all()
        err = 0.0
        for u0, u1, w in zip(setup["u_list"], u_t, setup["w_list"]):
            diff.x.array[:] = (u1.x.array - u0.x.array) / t
            diff.x.scatter_forward()
            err += integrate((diff - w) ** 2 * ds)
        y_vals[i] = np.sqrt(err) / norm_w

    slope = np.polyfit(np.log10(t_vals), np.log10(y_vals), 1)[0]
    assert 0.95 < slope < 1.05


@pytest.mark.parametrize("weight", ["one", "one_plus_z2"])
def test_adjoint_relation(setup, weight):
    mesh, ds_lat = setup["mesh"], setup["ds_lat"]
    z  = ufl.SpatialCoordinate(mesh)[2]
    hs = setup["currents"] if weight == "one" else [(1 + z ** 2) * g for g in setup["currents"]]

    adj = MultiAdjointSolver(mesh, setup["V"], setup["gamma"], setup["u_list"], hs, ds_lat, SOLVER, COMM).solve()
    lhs = integrate(adj * setup["sigma"] * ufl.dx)
    rhs = sum(integrate(h * w * ds_lat) for h, w in zip(hs, setup["w_list"]))
    assert abs(lhs - rhs) / abs(rhs) < 1e-8


def test_adjoint_input_validation(setup):
    s = setup
    with pytest.raises(ValueError):
        MultiAdjointSolver(s["mesh"], s["V"], s["gamma"], s["u_list"], s["currents"][:2], s["ds_lat"], SOLVER, COMM)
    solver = MultiAdjointSolver(s["mesh"], s["V"], s["gamma"], s["u_list"], s["currents"], s["ds_lat"], SOLVER, COMM)
    with pytest.raises(RuntimeError):
        _ = solver.psis