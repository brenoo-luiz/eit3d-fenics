"""Tests for the piecewise-constant sphere fields (gamma and eta)."""

import numpy as np
import pytest

from eit3d.config import ConductivityConfig, EtaConfig


def test_sphere_indicator():
    """Vectorized point-in-sphere test (the module itself imports dolfinx)."""
    pytest.importorskip("dolfinx")
    from eit3d.fields import sphere_indicator

    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask   = sphere_indicator(points, np.array([0.0, 0.0, 0.0]), 0.6)
    assert mask.tolist() == [True, True, False]


def _volume_where(field_fn, value) -> float:
    import dolfinx.fem
    import ufl
    from mpi4py import MPI

    indicator = dolfinx.fem.Function(field_fn.function_space)
    indicator.x.array[:] = np.isclose(field_fn.x.array, value).astype(float)
    return MPI.COMM_WORLD.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(indicator * ufl.dx)), op=MPI.SUM,
    )


def test_conductivity_values(mesh_data):
    from eit3d.fields import ConductivityField

    mesh, _, _ = mesh_data
    cfg   = ConductivityConfig()
    gamma = ConductivityField(mesh, cfg).build()
    values = set(np.unique(gamma.x.array).tolist())
    assert values == {cfg.gamma_in, cfg.gamma_out}


def test_conductivity_inclusion_volume(mesh_data):
    """Marked volume approximates the sphere volume (coarse mesh: 20 %)."""
    from eit3d.fields import ConductivityField

    mesh, _, _ = mesh_data
    cfg    = ConductivityConfig()
    gamma  = ConductivityField(mesh, cfg).build()
    exact  = 4.0 / 3.0 * np.pi * cfg.radius**3
    marked = _volume_where(gamma, cfg.gamma_in)
    assert abs(marked - exact) / exact < 0.2


def test_eta_marks_every_sphere(mesh_data):
    """Each eta sphere gets roughly the same marked volume."""
    from eit3d.fields import DirectionalField

    mesh, _, _ = mesh_data
    cfg   = EtaConfig()
    eta   = DirectionalField(mesh, cfg).build()
    exact = len(cfg.centers) * 4.0 / 3.0 * np.pi * cfg.radius**3
    marked = _volume_where(eta, cfg.eta_in)
    assert abs(marked - exact) / exact < 0.3


def test_fields_share_base_class():
    pytest.importorskip("dolfinx")
    from eit3d.fields import ConductivityField, DirectionalField, PiecewiseSphereField

    assert issubclass(ConductivityField, PiecewiseSphereField)
    assert issubclass(DirectionalField, PiecewiseSphereField)