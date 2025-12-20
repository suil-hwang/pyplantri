# src/pyplantri/__init__.py
"""Python wrapper for plantri planar graph generator.

This package provides a Python interface to the plantri program,
which generates various types of planar graphs including triangulations
and quadrangulations.

Classes:
    Plantri: Wrapper for the plantri executable.
    PlantriError: Exception raised when plantri execution fails.
    GraphConverter: Utility for converting graph data formats.
    SQSEnumerator: Enumerator for Simple Quadrangulation on Sphere.

Example:
    >>> from pyplantri import Plantri, SQSEnumerator
    >>> plantri = Plantri()
    >>> for graph in plantri.generate_graphs(8, graph_type="triangulation"):
    ...     print(graph)
"""
from .core import (
    Plantri,
    PlantriError,
    GraphConverter,
    SQSEnumerator,
)

__version__ = "0.1.0"
__all__ = [
    "Plantri",
    "PlantriError",
    "GraphConverter",
    "SQSEnumerator",
]

