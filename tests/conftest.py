"""Shared pytest fixtures."""

import pytest


@pytest.fixture(scope="session")
def mesh_data(tmp_path_factory):
    """
    Coarse cylinder mesh with its P2 space, built once per test session and
    cached in a temporary directory (the project cache/ is never touched).
    Requires FEniCS: tests using it are skipped when dolfinx/gmsh are missing.
    """
    pytest.importorskip("dolfinx")
    pytest.importorskip("gmsh")

    import basix.ufl
    import dolfinx.fem
    from mpi4py import MPI

    from eit3d.config import MeshConfig
    from eit3d.mesh.cylinder import CylinderMesh

    cylinder = CylinderMesh(
        MeshConfig(size_max=0.2, size_min=0.1),
        comm=MPI.COMM_WORLD,
        cache_dir=tmp_path_factory.mktemp("cache"),
    )
    mesh, facet_tags = cylinder.get()
    el = basix.ufl.element("Lagrange", "tetrahedron", degree=2, shape=())
    V  = dolfinx.fem.functionspace(mesh, el)
    return mesh, facet_tags, V