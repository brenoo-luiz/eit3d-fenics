from __future__ import annotations

from typing import TYPE_CHECKING

from eit3d.config import EITConfig

if TYPE_CHECKING:  # only for type checkers and IDEs, never executed
    from eit3d.pipeline.eit_pipeline import EITPipeline

__all__ = ["EITConfig", "EITPipeline"]


def __getattr__(name: str):
    if name == "EITPipeline":
        from eit3d.pipeline.eit_pipeline import EITPipeline
        return EITPipeline
    raise AttributeError(f"module 'eit3d' has no attribute {name!r}")