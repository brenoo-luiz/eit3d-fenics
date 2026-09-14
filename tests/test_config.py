"""
Unit tests for EITConfig dataclasses.
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

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            MeshConfig(radius=-1.0)

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
        cfg = ConductivityConfig()
        assert isinstance(cfg.center, np.ndarray)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            ConductivityConfig(gamma_in=-1.0)


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


class TestEITConfig:
    def test_defaults(self):
        cfg = EITConfig()
        assert isinstance(cfg.mesh, MeshConfig)
        assert isinstance(cfg.conductivity, ConductivityConfig)
        assert isinstance(cfg.solver, SolverConfig)

    def test_summary(self):
        cfg = EITConfig()
        s   = cfg.summary()
        assert "EIT" in s
        assert "Mesh" in s
