# src/pyplantri/__init__.py
from .cache import (
    CACHE_FORMAT_VERSION,
    LEGACY_CACHE_FORMAT_VERSION,
    SUPPORTED_CACHE_FORMAT_VERSIONS,
    CacheGraphClass,
    CacheMetadata,
    load_graphs_from_cache,
    save_graphs_to_cache,
    validate_cache_metadata,
)
from .converter import GraphConverter
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
    MAX_DUAL_VERTEX_COUNT,
    MIN_DUAL_VERTEX_COUNT,
    ParsedGraphSection,
    Plantri,
    QuadrangulationDualClass,
    PlantriExecutableNotFoundError,
    PlantriError,
    QuadrangulationEnumerator,
)

__version__ = "0.2.0"
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
    # Converter
    "GraphConverter",
    # Plane Graph model
    "PlaneGraph",
    "FrozenEdgeMultiplicity",
    # Cache
    "CacheMetadata",
    "CacheGraphClass",
    "CACHE_FORMAT_VERSION",
    "LEGACY_CACHE_FORMAT_VERSION",
    "SUPPORTED_CACHE_FORMAT_VERSIONS",
    "save_graphs_to_cache",
    "load_graphs_from_cache",
    "validate_cache_metadata",
    # Enumeration
    "enumerate_simple_quadrangulation_duals",
    "enumerate_simple_quadrangulation_duals_filtered",
    "enumerate_simple_quadrangulation_duals_parallel",
    "FilteredEnumerationResult",
    "EnumerationTiming",
]
