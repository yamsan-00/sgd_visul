"""
sgd_visualizer/__init__.py

Exposes the public API for external use.
"""

from .algorithms import vanilla_sgd, momentum_sgd
from .test_functions import FUNCTIONS
from .theme import COLORS, apply_dark_style

__all__ = [
    "vanilla_sgd",
    "momentum_sgd",
    "FUNCTIONS",
    "COLORS",
    "apply_dark_style",
]
