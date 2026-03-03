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
    enumerate_plane_graphs,
    enumerate_plane_graphs_filtered,
    enumerate_plane_graphs_parallel,
    iter_plane_graphs,
)
from .plane_graph import (
    FrozenEdgeMultiplicity,
    PlaneGraph,
)
from .plantri import (
    Plantri,
    PlantriError,
    QuadrangulationEnumerator,
)

__version__ = "0.1.0"
__all__ = [
    # Plantri wrapper
    "Plantri",
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
    "enumerate_plane_graphs",
    "enumerate_plane_graphs_filtered",
    "enumerate_plane_graphs_parallel",
    "iter_plane_graphs",
    "FilteredEnumerationResult",
    "EnumerationTiming",
]
