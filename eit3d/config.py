"""
Centralized configuration for the EIT 3D project.
All physical, geometric and numerical parameters are defined here.

Every config is an immutable value object (frozen dataclass): invariants are
validated once in __post_init__ and cannot be broken afterwards, e.g.
`cfg.mesh.radius = -1` raises FrozenInstanceError. NumPy arrays are stored
read-only and lists are stored as tuples for the same reason.
To change a parameter, build a new object: dataclasses.replace(cfg, radius=2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


ROOT_DIR    = Path(__file__).parent.parent
CACHE_DIR   = ROOT_DIR / "cache"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TOP_TAG     = 1
BOTTOM_TAG  = 2
LATERAL_TAG = 3
VOLUME_TAG  = 10


def _readonly_vector(value) -> np.ndarray:
    """Return a float copy of `value` that cannot be modified in place."""
    arr = np.array(value, dtype=float)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class MeshConfig:
    radius   : float = 1.0
    height   : float = 2.0
    size_max : float = 0.05
    size_min : float = 0.02

    def __post_init__(self) -> None:
        # Coerce to float so that radius=1 and radius=1.0 hash identically
        # (the mesh cache filename is derived from these values).
        for name in ("radius", "height", "size_max", "size_min"):
            object.__setattr__(self, name, float(getattr(self, name)))

        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got: {self.radius}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got: {self.height}")
        if self.size_min >= self.size_max:
            raise ValueError("size_min must be less than size_max")


# eq=False: fields holding NumPy arrays make field-by-field `==` ambiguous
# and unhashable, so these two configs compare (and hash) by identity.
@dataclass(frozen=True, eq=False)
class ConductivityConfig:
    center   : np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    radius   : float      = 0.35
    gamma_in : float      = 2.0
    gamma_out: float      = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _readonly_vector(self.center))
        if self.gamma_in <= 0 or self.gamma_out <= 0:
            raise ValueError("Conductivities must be positive (Lax-Milgram)")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    @property
    def gamma_min(self) -> float:
        return min(self.gamma_in, self.gamma_out)

    @property
    def gamma_max(self) -> float:
        return max(self.gamma_in, self.gamma_out)


@dataclass(frozen=True, eq=False)
class EtaConfig:
    centers : Tuple[np.ndarray, ...] = (
        (0.3, 0.0, 0.0),
        (-0.3, 0.0, 0.0),
    )
    radius  : float = 0.20
    eta_in  : float = 1.0
    eta_out : float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "centers", tuple(_readonly_vector(c) for c in self.centers),
        )
        if self.radius <= 0:
            raise ValueError("radius must be positive")


@dataclass(frozen=True)
class CurrentConfig:
    patterns: Tuple[Tuple[float, float], ...] = ((1.0, -1.0),)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "patterns",
            tuple((float(g_top), float(g_bot)) for g_top, g_bot in self.patterns),
        )
        for i, (g_top, g_bot) in enumerate(self.patterns):
            if not np.isclose(g_top + g_bot, 0.0, atol=1e-10):
                raise ValueError(
                    f"Pattern {i}: g_top + g_bot = {g_top+g_bot:.2e} != 0. "
                    "Existence condition violated."
                )


@dataclass(frozen=True)
class SolverConfig:
    rtol   : float = 1e-10
    atol   : float = 1e-12
    max_it : int   = 1000

    def __post_init__(self) -> None:
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("Tolerances must be positive")
        if self.max_it <= 0:
            raise ValueError("max_it must be positive")


@dataclass(frozen=True)
class ConsistencyTestConfig:
    n_iter : int   = 50
    base   : float = 0.9

    def __post_init__(self) -> None:
        if not (0 < self.base < 1):
            raise ValueError(f"base must be in (0,1), got: {self.base}")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")

    def t_values(self) -> np.ndarray:
        return self.base ** np.arange(self.n_iter)


@dataclass(frozen=True)
class EITConfig:
    mesh        : MeshConfig             = field(default_factory=MeshConfig)
    conductivity: ConductivityConfig     = field(default_factory=ConductivityConfig)
    eta         : EtaConfig              = field(default_factory=EtaConfig)
    current     : CurrentConfig          = field(default_factory=CurrentConfig)
    solver      : SolverConfig           = field(default_factory=SolverConfig)
    consistency : ConsistencyTestConfig  = field(default_factory=ConsistencyTestConfig)

    def summary(self) -> str:
        return "\n".join([
            f"mesh:         radius={self.mesh.radius}  height={self.mesh.height}"
            f"  h=[{self.mesh.size_min}, {self.mesh.size_max}]",
            f"conductivity: gamma in [{self.conductivity.gamma_min}, "
            f"{self.conductivity.gamma_max}]  sphere r={self.conductivity.radius}",
            f"eta:          {len(self.eta.centers)} sphere(s)  r={self.eta.radius}",
            f"current:      {len(self.current.patterns)} pattern(s)",
            f"solver:       CG+HYPRE  rtol={self.solver.rtol}"
            f"  max_it={self.solver.max_it}",
            f"consistency:  {self.consistency.n_iter} iter  base={self.consistency.base}",
        ])