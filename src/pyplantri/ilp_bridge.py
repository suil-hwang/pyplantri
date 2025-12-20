# src/pyplantri/ilp_bridge.py
"""ILP Bridge module for SQS Assignment problems.

This module provides structured graph objects and utilities for integrating
plantri-generated 4-regular planar multigraphs with ILP solvers.

All vertex indices are 0-based. Adjacency lists maintain CW (clockwise) order.

Example:
    >>> from pyplantri.ilp_bridge import enumerate_plantri_graphs, PlantriGraph
    >>> graphs = enumerate_plantri_graphs(6)
    >>> for graph in graphs:
    ...     print(f"Graph {graph.graph_id}: {graph.num_edges} edges")
"""
import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

from .core import GraphConverter, SQSEnumerator


@dataclass(frozen=True)
class PlantriGraph:
    """Immutable 4-regular planar multigraph from plantri.

    All indices are 0-based. Adjacency list neighbor order is CW (clockwise).

    Attributes:
        num_vertices: Number of vertices (= n).
        edges: Sorted edge tuples ((u, v) with u < v).
        edge_multiplicity: Mapping from edge to multiplicity (1 or 2).
        embedding: Mapping from vertex to CW-ordered neighbor tuple.
        faces: Tuple of face vertex tuples (including digons).
        graph_id: Unique identifier within the same n.

    Example:
        >>> graph = enumerate_plantri_graphs(4)[0]
        >>> graph.num_vertices
        4
        >>> graph.is_4_regular
        True
    """

    num_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    edge_multiplicity: Dict[Tuple[int, int], int]
    embedding: Dict[int, Tuple[int, ...]]
    faces: Tuple[Tuple[int, ...], ...]
    graph_id: int = 0

    @property
    def num_edges(self) -> int:
        """Total edge count including multiplicity."""
        return sum(self.edge_multiplicity.values())

    @property
    def num_simple_edges(self) -> int:
        """Unique edge count (without multiplicity)."""
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        """Face count (should be n + 2 by Euler's formula)."""
        return len(self.faces)

    @property
    def double_edges(self) -> FrozenSet[Tuple[int, int]]:
        """Set of double edges (digons)."""
        return frozenset(e for e, m in self.edge_multiplicity.items() if m == 2)

    @property
    def single_edges(self) -> FrozenSet[Tuple[int, int]]:
        """Set of single edges."""
        return frozenset(e for e, m in self.edge_multiplicity.items() if m == 1)

    @property
    def is_4_regular(self) -> bool:
        """Whether all vertices have degree 4."""
        return all(len(neighbors) == 4 for neighbors in self.embedding.values())

    @property
    def is_loop_free(self) -> bool:
        """Whether graph has no self-loops."""
        return all(v not in neighbors for v, neighbors in self.embedding.items())

    def get_neighbors_cw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CW-ordered neighbors of a vertex.

        Args:
            vertex: Vertex index (0-based).

        Returns:
            Tuple of neighbor indices in clockwise order.
        """
        return self.embedding[vertex]

    def get_neighbors_ccw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CCW-ordered neighbors of a vertex.

        Args:
            vertex: Vertex index (0-based).

        Returns:
            Tuple of neighbor indices in counter-clockwise order.
        """
        return tuple(reversed(self.embedding[vertex]))

    def get_consecutive_pairs(
        self, vertex: int, ccw: bool = False
    ) -> List[Tuple[int, int]]:
        """Gets consecutive neighbor pairs for cyclic order constraints.

        Useful for ILP assignment constraints that require checking
        consecutive pairs around a vertex.

        Args:
            vertex: Center vertex (0-based).
            ccw: If True, returns CCW pairs; otherwise CW pairs.

        Returns:
            List of (neighbor_i, neighbor_j) consecutive pairs.

        Example:
            >>> graph.get_consecutive_pairs(0)
            [(1, 2), (2, 3), (3, 0), (0, 1)]
        """
        neighbors = self.get_neighbors_ccw(vertex) if ccw else self.get_neighbors_cw(vertex)
        n = len(neighbors)
        return [(neighbors[i], neighbors[(i + 1) % n]) for i in range(n)]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validates graph invariants.

        Checks:
            - 4-regularity
            - Edge formula: s + 2d = 2n
            - Face count: n + 2
            - No self-loops

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check 4-regularity
        for v in range(self.num_vertices):
            if v not in self.embedding:
                errors.append(f"Vertex {v} missing from embedding")
                continue
            if len(self.embedding[v]) != 4:
                errors.append(
                    f"Vertex {v} has degree {len(self.embedding[v])}, expected 4"
                )

        # Check edge formula: s + 2d = 2n
        s = len(self.single_edges)
        d = len(self.double_edges)
        expected = 2 * self.num_vertices
        actual = s + 2 * d
        if actual != expected:
            errors.append(f"Edge formula: s + 2d = {actual}, expected {expected}")

        # Check face count: n + 2
        expected_faces = self.num_vertices + 2
        if self.num_faces != expected_faces:
            errors.append(f"Face count: {self.num_faces}, expected {expected_faces}")

        # Check no self-loops
        for v, neighbors in self.embedding.items():
            if v in neighbors:
                errors.append(f"Self-loop at vertex {v}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict:
        """Converts to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "num_vertices": self.num_vertices,
            "edges": list(self.edges),
            "edge_multiplicity": {
                f"{u},{v}": m for (u, v), m in self.edge_multiplicity.items()
            },
            "embedding": {str(v): list(neighbors) for v, neighbors in self.embedding.items()},
            "faces": [list(f) for f in self.faces],
            "graph_id": self.graph_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlantriGraph":
        """Creates PlantriGraph from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON.

        Returns:
            PlantriGraph instance.
        """
        return cls(
            num_vertices=data["num_vertices"],
            edges=tuple(tuple(e) for e in data["edges"]),
            edge_multiplicity={
                tuple(map(int, k.split(","))): v
                for k, v in data["edge_multiplicity"].items()
            },
            embedding={
                int(v): tuple(neighbors) for v, neighbors in data["embedding"].items()
            },
            faces=tuple(tuple(f) for f in data["faces"]),
            graph_id=data.get("graph_id", 0),
        )



def _build_plantri_graph(
    dual_data: Dict,
    graph_id: int,
) -> PlantriGraph:
    """Builds PlantriGraph from SQS dual data.

    Args:
        dual_data: Dual graph data from SQSEnumerator.
        graph_id: Unique graph identifier.

    Returns:
        PlantriGraph instance.
    """
    n = dual_data["vertex_count"]
    adj_list_1based = dual_data["adjacency_list"]

    # Convert to 0-based embedding using GraphConverter
    embedding = GraphConverter.to_zero_based_embedding(adj_list_1based)

    # Compute edge multiplicity
    edge_mult: Dict[Tuple[int, int], int] = {}
    edges_set: set = set()

    for v, neighbors in embedding.items():
        for u in neighbors:
            edge = (min(v, u), max(v, u))
            edges_set.add(edge)

    for edge in edges_set:
        u, v = edge
        count = embedding[u].count(v)
        edge_mult[edge] = count

    edges = tuple(sorted(edges_set))

    # Extract faces using GraphConverter
    faces = GraphConverter.extract_faces(embedding, edge_mult)

    return PlantriGraph(
        num_vertices=n,
        edges=edges,
        edge_multiplicity=edge_mult,
        embedding=embedding,
        faces=faces,
        graph_id=graph_id,
    )


def enumerate_plantri_graphs(
    dual_vertex_count: int,
    max_count: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
) -> List[PlantriGraph]:
    """Enumerates all n-vertex 4-regular planar multigraphs.

    Returns PlantriGraph objects suitable for ILP assignment problems.

    Args:
        dual_vertex_count: Number of vertices in Q* (Dual) = n.
        max_count: Maximum number of graphs to return (None for all).
        validate: If True, validates each graph's invariants.
        verbose: If True, prints progress information.

    Returns:
        List of PlantriGraph objects.

    Example:
        >>> graphs = enumerate_plantri_graphs(6)
        >>> len(graphs)
        9
        >>> graphs[0].num_vertices
        6
        >>> graphs[0].validate()
        (True, [])
    """
    n = dual_vertex_count

    if verbose:
        print(f"[Plantri] Enumerating {n}-vertex 4-regular planar multigraphs...")

    sqs = SQSEnumerator()
    graphs = []

    for graph_id, (_, dual_data) in enumerate(sqs.generate_pairs(n)):
        graph = _build_plantri_graph(dual_data, graph_id)

        if validate:
            is_valid, errors = graph.validate()
            if not is_valid:
                if verbose:
                    print(f"  [!] Graph {graph_id} validation failed: {errors}")
                continue

        graphs.append(graph)

        if verbose and (graph_id + 1) % 100 == 0:
            print(f"  Processed {graph_id + 1} graphs...")

        if max_count and len(graphs) >= max_count:
            break

    if verbose:
        print(f"[Plantri] Found {len(graphs)} valid graphs")

    return graphs


def iter_plantri_graphs(
    dual_vertex_count: int,
    validate: bool = True,
) -> Iterator[PlantriGraph]:
    """Memory-efficient iterator for PlantriGraph objects.

    Use this for processing large numbers of graphs without loading
    all into memory at once.

    Args:
        dual_vertex_count: Number of vertices in Q* (Dual).
        validate: If True, skips invalid graphs.

    Yields:
        PlantriGraph objects one at a time.

    Example:
        >>> for graph in iter_plantri_graphs(10):
        ...     process_graph(graph)
    """
    sqs = SQSEnumerator()

    for graph_id, (_, dual_data) in enumerate(sqs.generate_pairs(dual_vertex_count)):
        graph = _build_plantri_graph(dual_data, graph_id)

        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                continue

        yield graph


def save_graphs_to_cache(
    graphs: List[PlantriGraph],
    filepath: str,
    use_pickle: bool = False,
) -> None:
    """Saves graph list to cache file.

    Args:
        graphs: List of PlantriGraph objects.
        filepath: Output file path.
        use_pickle: If True, uses pickle format; otherwise JSON.
    """
    if use_pickle:
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(graphs, f)
    else:
        data = [g.to_dict() for g in graphs]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def load_graphs_from_cache(
    filepath: str,
    use_pickle: bool = False,
) -> List[PlantriGraph]:
    """Loads graph list from cache file.

    Args:
        filepath: Input file path.
        use_pickle: If True, reads pickle format; otherwise JSON.

    Returns:
        List of PlantriGraph objects.
    """
    if use_pickle:
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PlantriGraph.from_dict(d) for d in data]


def main() -> None:
    """CLI entry point for plantri graph enumeration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate 4-regular planar multigraphs via plantri",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m pyplantri.ilp_bridge 6
    python -m pyplantri.ilp_bridge 8 --max 5
    python -m pyplantri.ilp_bridge 6 --export cache/n6.json
    python -m pyplantri.ilp_bridge 6 --show-faces
        """,
    )
    parser.add_argument("n", type=int, help="Number of dual vertices (minimum 3)")
    parser.add_argument("--max", type=int, default=None, help="Maximum graph count")
    parser.add_argument("--export", type=str, help="Export to JSON cache file")
    parser.add_argument("--pickle", action="store_true", help="Use pickle format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--show-faces", action="store_true", help="Display face info")

    args = parser.parse_args()

    if args.n < 3:
        parser.error("n must be at least 3.")

    graphs = enumerate_plantri_graphs(
        args.n,
        max_count=args.max,
        verbose=args.verbose,
    )

    print(f"\nTotal: {len(graphs)} {args.n}-vertex 4-regular planar multigraphs")

    for g in graphs:
        print(f"\n{'='*50}")
        print(f"Graph #{g.graph_id}")
        print(f"  Vertices: {g.num_vertices}")
        print(f"  Edges: {g.num_simple_edges} unique, {g.num_edges} total")
        print(f"  Double edges: {len(g.double_edges)}")
        print(f"  Faces: {g.num_faces}")

        print("  Embedding (CW order, 0-based):")
        for v in range(g.num_vertices):
            print(f"    {v}: {list(g.embedding[v])}")

        if args.show_faces:
            print("  Faces:")
            for i, face in enumerate(g.faces):
                face_type = "digon" if len(face) == 2 else f"{len(face)}-gon"
                print(f"    F{i}: {list(face)} ({face_type})")

    if args.export:
        save_graphs_to_cache(graphs, args.export, use_pickle=args.pickle)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
