# src/pyplantri/__init__.py
from importlib import metadata as _metadata

from .cache import (
    CACHE_DEFAULT_CHUNK_SIZE,
    CACHE_FORMAT_VERSION,
    CacheGraphClass,
    CacheMetadata,
    PlaneGraphCatalog,
    load_graph_catalog,
    load_graphs_from_cache,
    save_graphs_to_cache,
    validate_cache_metadata,
)
from .enumeration import (
    EnumerationTiming,
    FilteredEnumerationResult,
    enumerate_simple_quadrangulation_duals,
    enumerate_simple_quadrangulation_duals_filtered,
    enumerate_simple_quadrangulation_duals_parallel,
)
from .plane_graph import (
    FrozenEdgeMultiplicity,
    PlaneGraph,
)
from .plantri import (
    MAX_DOUBLE_CODE_DUAL_VERTEX_COUNT,
    MAX_DUAL_VERTEX_COUNT,
    MIN_DUAL_VERTEX_COUNT,
    ParsedGraphSection,
    Plantri,
    QuadrangulationDualClass,
    PlantriExecutableNotFoundError,
    PlantriError,
    QuadrangulationEnumerator,
)

try:
    __version__ = _metadata.version("pyplantri")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Plantri wrapper
    "ParsedGraphSection",
    "Plantri",
    "QuadrangulationDualClass",
    "PlantriExecutableNotFoundError",
    "PlantriError",
    "QuadrangulationEnumerator",
    "MIN_DUAL_VERTEX_COUNT",
    "MAX_DUAL_VERTEX_COUNT",
    "MAX_DOUBLE_CODE_DUAL_VERTEX_COUNT",
    # Plane Graph model
    "PlaneGraph",
    "FrozenEdgeMultiplicity",
    # Cache
    "CacheMetadata",
    "CacheGraphClass",
    "PlaneGraphCatalog",
    "CACHE_DEFAULT_CHUNK_SIZE",
    "CACHE_FORMAT_VERSION",
    "save_graphs_to_cache",
    "load_graph_catalog",
    "load_graphs_from_cache",
    "validate_cache_metadata",
    # Enumeration
    "enumerate_simple_quadrangulation_duals",
    "enumerate_simple_quadrangulation_duals_filtered",
    "enumerate_simple_quadrangulation_duals_parallel",
    "FilteredEnumerationResult",
    "EnumerationTiming",
]
