"""
These tests depend only on NumPy, so they run without FEniCS.
"""

import numpy as np
import pytest

from eit3d.config import (
    MeshConfig, ConductivityConfig, EtaConfig,
    CurrentConfig, SolverConfig, ConsistencyTestConfig, EITConfig,
)


class TestMeshConfig:
    def test_defaults(self):
        cfg = MeshConfig()
        assert cfg.radius == 1.0
        assert cfg.height == 2.0
        assert cfg.size_min < cfg.size_max

    def test_values_coerced_to_float(self):
        cfg = MeshConfig(radius=1, height=2)
        assert isinstance(cfg.radius, float)
        assert isinstance(cfg.height, float)

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            MeshConfig(radius=-1.0)

    def test_invalid_height(self):
        with pytest.raises(ValueError):
            MeshConfig(height=0.0)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            MeshConfig(size_min=0.1, size_max=0.05)


class TestConductivityConfig:
    def test_defaults(self):
        cfg = ConductivityConfig()
        assert cfg.gamma_in == 2.0
        assert cfg.gamma_out == 1.0
        assert cfg.gamma_min == 1.0
        assert cfg.gamma_max == 2.0

    def test_center_is_ndarray(self):
        cfg = ConductivityConfig(center=[0.1, 0.0, 0.0])
        assert isinstance(cfg.center, np.ndarray)
        assert cfg.center.dtype == float

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            ConductivityConfig(gamma_in=-1.0)

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            ConductivityConfig(radius=0.0)


class TestEtaConfig:
    def test_defaults(self):
        cfg = EtaConfig()
        assert len(cfg.centers) == 2
        assert all(isinstance(c, np.ndarray) for c in cfg.centers)

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            EtaConfig(radius=-0.1)


class TestCurrentConfig:
    def test_valid_pattern(self):
        cfg = CurrentConfig(patterns=[(1.0, -1.0)])
        assert len(cfg.patterns) == 1

    def test_existence_condition_violated(self):
        with pytest.raises(ValueError):
            CurrentConfig(patterns=[(1.0, 1.0)])


class TestSolverConfig:
    def test_defaults(self):
        cfg = SolverConfig()
        assert cfg.rtol < 1.0
        assert cfg.max_it > 0

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            SolverConfig(rtol=-1e-10)

    def test_invalid_max_it(self):
        with pytest.raises(ValueError):
            SolverConfig(max_it=0)


class TestConsistencyTestConfig:
    def test_t_values(self):
        cfg = ConsistencyTestConfig(n_iter=5, base=0.9)
        t   = cfg.t_values()
        assert len(t) == 5
        assert np.isclose(t[0], 1.0)
        assert np.isclose(t[1], 0.9)

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            ConsistencyTestConfig(base=1.5)

    def test_invalid_n_iter(self):
        with pytest.raises(ValueError):
            ConsistencyTestConfig(n_iter=0)


class TestEITConfig:
    def test_defaults(self):
        cfg = EITConfig()
        assert isinstance(cfg.mesh, MeshConfig)
        assert isinstance(cfg.conductivity, ConductivityConfig)
        assert isinstance(cfg.solver, SolverConfig)

    def test_summary_sections(self):
        s = EITConfig().summary()
        for section in ("mesh:", "conductivity:", "eta:",
                        "current:", "solver:", "consistency:"):
            assert section in s

    def test_summary_reflects_config(self):
        s = EITConfig(mesh=MeshConfig(radius=1.5)).summary()
        assert "radius=1.5" in s


class TestMeshCachePath:
    """Needs dolfinx/gmsh; skipped automatically outside the FEniCS env."""

    @pytest.fixture(autouse=True)
    def _require_fenics(self):
        pytest.importorskip("dolfinx")
        pytest.importorskip("gmsh")

    def test_int_and_float_same_path(self, tmp_path):
        from eit3d.mesh.cylinder import mesh_cache_path
        a = mesh_cache_path(MeshConfig(radius=1), tmp_path)
        b = mesh_cache_path(MeshConfig(radius=1.0), tmp_path)
        assert a == b

    def test_different_config_different_path(self, tmp_path):
        from eit3d.mesh.cylinder import mesh_cache_path
        a = mesh_cache_path(MeshConfig(size_max=0.05), tmp_path)
        b = mesh_cache_path(MeshConfig(size_max=0.06), tmp_path)
        assert a != b