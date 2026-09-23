"""
Numerical regression tests for the EIT 3D solvers.

Run on a coarse mesh and check the mathematical properties
verified by the scripts:

    - manufactured solution is reproduced exactly (P2 represents x^2 - y^2)
    - every solution satisfies int_dOmega u ds = 0 (H^1_diamond normalization)
    - forward solution is odd in z (symmetry of gamma and g)
    - directional derivative is first-order consistent (slope 1 in log-log)

Skipped automatically when FEniCS (dolfinx) or gmsh are not installed.
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")

import dolfinx  # noqa: E402
import dolfinx.fem  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

from eit3d.config import ConductivityConfig, EtaConfig, SolverConfig  # noqa: E402
from eit3d.fields.conductivity import ConductivityField, DirectionalField  # noqa: E402
from eit3d.solvers import (  # noqa: E402
    DerivativeSolver, ForwardSolver, NeumannSolver,
)

COMM   = MPI.COMM_WORLD
SOLVER = SolverConfig()


# Fixtures (mesh_data comes from conftest.py; base solutions are built once per module)
@pytest.fixture(scope="module")
def fields(mesh_data):
    mesh, _, _ = mesh_data
    gamma = ConductivityField(mesh, ConductivityConfig()).build()
    eta   = DirectionalField(mesh, EtaConfig()).build()
    return gamma, eta


@pytest.fixture(scope="module")
def u_gamma(mesh_data, fields):
    mesh, facet_tags, V = mesh_data
    gamma, _ = fields
    return ForwardSolver(mesh, facet_tags, V, gamma, SOLVER, COMM).solve()


@pytest.fixture(scope="module")
def omega(mesh_data, fields, u_gamma):
    mesh, _, V = mesh_data
    gamma, eta = fields
    return DerivativeSolver(mesh, V, gamma, eta, u_gamma, SOLVER, COMM).solve()


# Helpers
def integrate(expr) -> float:
    return COMM.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr)), op=MPI.SUM,
    )


def boundary_mean(u, mesh) -> float:
    ds  = ufl.Measure("ds", domain=mesh)
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    return integrate(u * ds) / integrate(one * ds)


# Tests
def test_manufactured_solution(mesh_data):
    """Pure Neumann problem with u = x^2 - y^2 is solved exactly by P2."""
    mesh, _, V = mesh_data
    x  = ufl.SpatialCoordinate(mesh)
    n  = ufl.FacetNormal(mesh)
    ue = x[0]**2 - x[1]**2
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    u_h = NeumannSolver(
        mesh, V, gamma=one, g=ufl.dot(ufl.grad(ue), n), config=SOLVER, comm=COMM,
    ).solve()

    ds  = ufl.Measure("ds", domain=mesh)
    c   = integrate(ue * ds) / integrate(one * ds)
    err = np.sqrt(integrate((u_h - (ue - c))**2 * ufl.dx))
    ref = np.sqrt(integrate((ue - c)**2 * ufl.dx))
    assert err / ref < 1e-8


def test_forward_boundary_mean_zero(mesh_data, u_gamma):
    mesh, _, _ = mesh_data
    assert abs(boundary_mean(u_gamma, mesh)) < 1e-12


def test_derivative_boundary_mean_zero(mesh_data, omega):
    mesh, _, _ = mesh_data
    assert abs(boundary_mean(omega, mesh)) < 1e-12


def test_forward_is_odd_in_z(u_gamma):
    """
    gamma is symmetric and g_top = -g_bot, so the exact solution is odd in z
    and max(u) = -min(u). The unstructured mesh is not symmetric, so the
    discrete solution is only approximately odd (the gap shrinks with h).
    """
    u_max = float(u_gamma.x.array.max())
    u_min = float(u_gamma.x.array.min())
    assert np.isclose(u_max, -u_min, rtol=5e-3)


def test_derivative_first_order_consistency(mesh_data, fields, u_gamma, omega):
    """y(t) = ||(F(gamma + t eta) - F(gamma)) / t - F'(gamma)eta|| / ||F'(gamma)eta|| = O(t)."""
    mesh, facet_tags, V = mesh_data
    gamma, eta = fields
    ds = ufl.Measure("ds", domain=mesh)

    gamma_t = dolfinx.fem.Function(gamma.function_space)
    diff    = dolfinx.fem.Function(V)
    norm_om = np.sqrt(integrate(omega**2 * ds))
    t_vals  = np.array([1e-2, 1e-3, 1e-4])
    y_vals  = np.empty(len(t_vals))

    for k, t in enumerate(t_vals):
        gamma_t.x.array[:] = gamma.x.array + t * eta.x.array
        gamma_t.x.scatter_forward()
        u_t = ForwardSolver(mesh, facet_tags, V, gamma_t, SOLVER, COMM).solve()
        diff.x.array[:] = (u_t.x.array - u_gamma.x.array) / t
        diff.x.scatter_forward()
        y_vals[k] = np.sqrt(integrate((diff - omega)**2 * ds)) / norm_om

    slope = np.polyfit(np.log10(t_vals), np.log10(y_vals), 1)[0]
    assert 0.95 < slope < 1.05