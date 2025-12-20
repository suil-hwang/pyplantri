# src/pyplantri/__init__.py
"""Python wrapper for plantri planar graph generator.

This package provides a Python interface to the plantri program,
which generates various types of planar graphs including triangulations
and quadrangulations.

Core Classes:
    Plantri: Wrapper for the plantri executable.
    PlantriError: Exception raised when plantri execution fails.
    GraphConverter: Utility for converting graph data formats.
    SQSEnumerator: Enumerator for Simple Quadrangulation on Sphere.

ILP Bridge:
    PlantriGraph: Immutable 4-regular planar multigraph dataclass.
    enumerate_plantri_graphs: List all n-vertex graphs.
    iter_plantri_graphs: Memory-efficient graph iterator.

Example:
    >>> from pyplantri import SQSEnumerator, enumerate_plantri_graphs
    >>> # Basic usage
    >>> sqs = SQSEnumerator()
    >>> for primal, dual in sqs.generate_pairs(4):
    ...     print(dual['vertex_count'])
    >>> # ILP-ready graphs
    >>> graphs = enumerate_plantri_graphs(6)
    >>> for g in graphs:
    ...     print(g.num_vertices, g.double_edges)
"""
from .core import (
    GraphConverter,
    Plantri,
    PlantriError,
    SQSEnumerator,
)
from .ilp_bridge import (
    PlantriGraph,
    enumerate_plantri_graphs,
    iter_plantri_graphs,
    load_graphs_from_cache,
    save_graphs_to_cache,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Plantri",
    "PlantriError",
    "GraphConverter",
    "SQSEnumerator",
    # ILP Bridge
    "PlantriGraph",
    "enumerate_plantri_graphs",
    "iter_plantri_graphs",
    "save_graphs_to_cache",
    "load_graphs_from_cache",
]

