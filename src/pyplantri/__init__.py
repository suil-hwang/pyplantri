# src/pyplantri/__init__.py
from importlib import metadata as _metadata

from .cache import (
    CACHE_DEFAULT_CHUNK_SIZE,
    CACHE_FORMAT_VERSION,
    CacheGraphClass,
    CacheIndexMode,
    CacheMetadata,
    CacheValidation,
    QuarticPlaneMapCache,
    open_graph_cache,
    validate_cache_metadata,
    write_graph_cache,
)
from .enumeration import (
    PlantriEnumerationResult,
    enumerate_simple_quadrangulation_duals,
    iter_simple_quadrangulation_duals,
)
from .plane_graph import MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT, QuarticPlaneMap
from .plantri_interface import (
    BUNDLED_MAX_DUAL_VERTEX_COUNT,
    BUNDLED_PLANTRI_MAX_VERTEX_COUNT,
    MIN_DUAL_VERTEX_COUNT,
    PlanarCodeError,
    Plantri,
    PlantriProvenance,
    PlantriTimeoutError,
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
    "PlantriTimeoutError",
    "PlantriProvenance",
    "QuadrangulationEnumerator",
    "MIN_DUAL_VERTEX_COUNT",
    "BUNDLED_PLANTRI_MAX_VERTEX_COUNT",
    "BUNDLED_MAX_DUAL_VERTEX_COUNT",
    "MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT",
    "iter_planar_code",
    # Quartic plane-map model
    "QuarticPlaneMap",
    # Cache
    "CacheMetadata",
    "CacheGraphClass",
    "CacheIndexMode",
    "CacheValidation",
    "QuarticPlaneMapCache",
    "CACHE_DEFAULT_CHUNK_SIZE",
    "CACHE_FORMAT_VERSION",
    "write_graph_cache",
    "open_graph_cache",
    "validate_cache_metadata",
    # Enumeration
    "iter_simple_quadrangulation_duals",
    "enumerate_simple_quadrangulation_duals",
    "PlantriEnumerationResult",
]
