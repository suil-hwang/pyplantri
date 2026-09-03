# pyplantri/__init__.py
from importlib import metadata as _metadata

from .plantri import (
    MAX_DUAL_VERTEX_COUNT,
    MIN_DUAL_VERTEX_COUNT,
    PlantriEnumerationResult,
    PlantriError,
    PrimalMinimumDegree,
    SimpleQuadrangulation,
    DualPlaneGraph,
    enumerate_simple_quadrangulation_duals,
)

try:
    __version__ = _metadata.version("pyplantri")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # FILTER-backed SQS enumeration
    "PrimalMinimumDegree",
    "PlantriError",
    "MAX_DUAL_VERTEX_COUNT",
    "MIN_DUAL_VERTEX_COUNT",
    # Candidate primal/dual plane-graph model
    "SimpleQuadrangulation",
    "DualPlaneGraph",
    # Enumeration
    "enumerate_simple_quadrangulation_duals",
    "PlantriEnumerationResult",
]
