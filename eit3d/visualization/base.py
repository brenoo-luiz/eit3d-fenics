"""
Abstract base class for all renderers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import dolfinx


class BaseRenderer(ABC):
    """
    Abstract interface for EIT 3D renderers.

    Subclasses: StaticRenderer (PNG), InteractiveRenderer (PyVista window).
    """

    BG = "#1e1e2e"

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render and save or display the visualization."""

    def _sargs(self, title: str) -> dict:
        """Standard scalar bar arguments for PyVista."""
        return dict(
            title=title, title_font_size=20, label_font_size=15,
            color="white", position_x=0.03, position_y=0.03,
            width=0.42, height=0.06,
        )
