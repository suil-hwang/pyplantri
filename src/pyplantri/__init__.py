# src/pyplantri/__init__.py
from .core import (
    GraphConverter,
    Plantri,
    PlantriError,
    QuadrangulationEnumerator,
)
from .plane_graph import (
    FilteredEnumerationResult,
    PlaneGraph,
    enumerate_plane_graphs,
    enumerate_plane_graphs_filtered,
    iter_plane_graphs,
    load_graphs_from_cache,
    save_graphs_to_cache,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Plantri",
    "PlantriError",
    "GraphConverter",
    "QuadrangulationEnumerator",
    # Plane Graph
    "PlaneGraph",
    "enumerate_plane_graphs",
    "enumerate_plane_graphs_filtered",
    "FilteredEnumerationResult",
    "iter_plane_graphs",
    "save_graphs_to_cache",
    "load_graphs_from_cache",
]
