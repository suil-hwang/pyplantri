# src/pyplantri/__init__.py
from importlib import metadata as _metadata

from .cache import (
    CACHE_DEFAULT_CHUNK_SIZE,
    CACHE_FORMAT_VERSION,
    CacheGraphClass,
    CacheMetadata,
    QuarticPlaneMapCatalog,
    load_graph_catalog,
    load_graphs_from_cache,
    save_graphs_to_cache,
    validate_cache_metadata,
)
from .enumeration import (
    PlantriEnumerationResult,
    enumerate_simple_quadrangulation_duals,
)
from .plane_graph import QuarticPlaneMap
from .plantri import (
    MAX_DUAL_VERTEX_COUNT,
    MIN_DUAL_VERTEX_COUNT,
    PlanarCodeError,
    Plantri,
    QuadrangulationDualClass,
    PlantriExecutableNotFoundError,
    PlantriError,
    QuadrangulationEnumerator,
    iter_planar_code,
)

try:
    __version__ = _metadata.version("pyplantri")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    # Plantri wrapper
    "PlanarCodeError",
    "Plantri",
    "QuadrangulationDualClass",
    "PlantriExecutableNotFoundError",
    "PlantriError",
    "QuadrangulationEnumerator",
    "MIN_DUAL_VERTEX_COUNT",
    "MAX_DUAL_VERTEX_COUNT",
    "iter_planar_code",
    # Quartic plane-map model
    "QuarticPlaneMap",
    # Cache
    "CacheMetadata",
    "CacheGraphClass",
    "QuarticPlaneMapCatalog",
    "CACHE_DEFAULT_CHUNK_SIZE",
    "CACHE_FORMAT_VERSION",
    "save_graphs_to_cache",
    "load_graph_catalog",
    "load_graphs_from_cache",
    "validate_cache_metadata",
    # Enumeration
    "enumerate_simple_quadrangulation_duals",
    "PlantriEnumerationResult",
]
