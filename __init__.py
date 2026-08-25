# pyplantri/__init__.py
from importlib import metadata as _metadata

from .plantri import (
    BUNDLED_MAX_DUAL_VERTEX_COUNT,
    MIN_SUPPORTED_DUAL_VERTEX_COUNT,
    PlantriEnumerationResult,
    PlantriExecutableNotFoundError,
    PlantriError,
    PlantriTimeoutError,
    PrimalMinimumDegree,
    SimpleQuadrangulation,
    QuarticPlaneMap,
    enumerate_simple_quadrangulation_duals,
    iter_simple_quadrangulation_duals,
)

try:
    __version__ = _metadata.version("pyplantri")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # FILTER-backed SQS enumeration
    "PrimalMinimumDegree",
    "PlantriExecutableNotFoundError",
    "PlantriError",
    "PlantriTimeoutError",
    "BUNDLED_MAX_DUAL_VERTEX_COUNT",
    "MIN_SUPPORTED_DUAL_VERTEX_COUNT",
    # Candidate primal/dual plane-graph model
    "SimpleQuadrangulation",
    "QuarticPlaneMap",
    # Enumeration
    "iter_simple_quadrangulation_duals",
    "enumerate_simple_quadrangulation_duals",
    "PlantriEnumerationResult",
]
