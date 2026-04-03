# src/pyplantri/__init__.py
from .cache import (
    CacheMetadata,
    load_graphs_from_cache,
    save_graphs_to_cache,
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
    ParsedGraphSection,
    Plantri,
    QuadrangulationDualClass,
    PlantriExecutableNotFoundError,
    PlantriError,
    QuadrangulationEnumerator,
)

__version__ = "0.1.0"
__all__ = [
    # Plantri wrapper
    "ParsedGraphSection",
    "Plantri",
    "QuadrangulationDualClass",
    "PlantriExecutableNotFoundError",
    "PlantriError",
    "QuadrangulationEnumerator",
    # Converter
    "GraphConverter",
    # Plane Graph model
    "PlaneGraph",
    "FrozenEdgeMultiplicity",
    # Cache
    "CacheMetadata",
    "save_graphs_to_cache",
    "load_graphs_from_cache",
    # Enumeration
    "enumerate_simple_quadrangulation_duals",
    "enumerate_simple_quadrangulation_duals_filtered",
    "enumerate_simple_quadrangulation_duals_parallel",
    "FilteredEnumerationResult",
    "EnumerationTiming",
]
