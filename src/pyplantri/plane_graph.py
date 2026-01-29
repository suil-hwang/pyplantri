# src/pyplantri/plane_graph.py
import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

from .core import GraphConverter, QuadrangulationEnumerator


@dataclass(frozen=True)
class PlaneGraph:
    """Plane graph (embedded planar graph) with fixed combinatorial embedding.

    A plane graph is a planar graph with a specific embedding on the sphere,
    represented by clockwise cyclic edge ordering at each vertex. Generated
    by plantri (Brinkmann & McKay).

    Terminology:
        - Planar graph: A graph that CAN be embedded in the plane (abstract).
        - Plane graph: A planar graph WITH a fixed embedding (concrete).

    This class represents an immutable 4-regular plane multigraph containing
    both dual (Q*) and primal (Q) topology. All indices are 0-based.

    Dual Graph (Q*):
        - 4-regular plane multigraph (allows double edges, no loops).
        - num_vertices = n (dual vertices).
        - faces = n + 2 (dual faces = primal vertices).

    Primal Graph (Q):
        - Simple quadrangulation (no loops/multi-edges, all faces are 4-gons).
        - primal_num_vertices = n + 2 (primal vertices = dual faces).
        - primal_faces = n (primal faces = dual vertices).
    """

    num_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    edge_multiplicity: Dict[Tuple[int, int], int]
    embedding: Dict[int, Tuple[int, ...]]  # CW cyclic order at each vertex
    faces: Tuple[Tuple[int, ...], ...]

    primal_num_vertices: int
    primal_embedding: Dict[int, Tuple[int, ...]]
    primal_faces: Tuple[Tuple[int, ...], ...]

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
        """Gets consecutive neighbor pairs for cyclic order constraints."""
        neighbors = self.get_neighbors_ccw(vertex) if ccw else self.get_neighbors_cw(vertex)
        n = len(neighbors)
        return [(neighbors[i], neighbors[(i + 1) % n]) for i in range(n)]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validates graph invariants."""
        errors = []

        for v in range(self.num_vertices):
            if v not in self.embedding:
                errors.append(f"Vertex {v} missing from embedding")
                continue
            if len(self.embedding[v]) != 4:
                errors.append(
                    f"Vertex {v} has degree {len(self.embedding[v])}, expected 4"
                )

        single_count = len(self.single_edges)
        double_count = len(self.double_edges)
        expected_edge_sum = 2 * self.num_vertices
        actual_edge_sum = single_count + 2 * double_count
        if actual_edge_sum != expected_edge_sum:
            errors.append(f"Edge formula: s + 2d = {actual_edge_sum}, expected {expected_edge_sum}")

        expected_faces = self.num_vertices + 2
        if self.num_faces != expected_faces:
            errors.append(f"Face count: {self.num_faces}, expected {expected_faces}")

        for v, neighbors in self.embedding.items():
            if v in neighbors:
                errors.append(f"Self-loop at vertex {v}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict:
        """Converts to dictionary for JSON serialization."""
        return {
            "num_vertices": self.num_vertices,
            "edges": list(self.edges),
            "edge_multiplicity": {
                f"{u},{v}": m for (u, v), m in self.edge_multiplicity.items()
            },
            "embedding": {
                str(v): list(neighbors) for v, neighbors in self.embedding.items()
            },
            "faces": [list(f) for f in self.faces],
            "primal_num_vertices": self.primal_num_vertices,
            "primal_embedding": {
                str(v): list(neighbors)
                for v, neighbors in self.primal_embedding.items()
            },
            "primal_faces": [list(f) for f in self.primal_faces],
            "graph_id": self.graph_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlaneGraph":
        """Creates PlaneGraph from dictionary."""
        # Parse edges as explicit 2-tuples for type safety.
        parsed_edges: List[Tuple[int, int]] = [
            (int(e[0]), int(e[1])) for e in data["edges"]
        ]
        # Parse edge_multiplicity keys as explicit 2-tuples.
        parsed_edge_multiplicity: Dict[Tuple[int, int], int] = {
            (int(parts[0]), int(parts[1])): v
            for k, v in data["edge_multiplicity"].items()
            for parts in [k.split(",")]
        }
        return cls(
            num_vertices=data["num_vertices"],
            edges=tuple(parsed_edges),
            edge_multiplicity=parsed_edge_multiplicity,
            embedding={
                int(v): tuple(neighbors) for v, neighbors in data["embedding"].items()
            },
            faces=tuple(tuple(f) for f in data["faces"]),
            primal_num_vertices=data.get("primal_num_vertices", 0),
            primal_embedding={
                int(v): tuple(neighbors)
                for v, neighbors in data.get("primal_embedding", {}).items()
            },
            primal_faces=tuple(
                tuple(f) for f in data.get("primal_faces", [])
            ),
            graph_id=data.get("graph_id", 0),
        )


def _build_plane_graph(
    primal_data: Dict,
    dual_data: Dict,
    graph_id: int,
) -> PlaneGraph:
    """Builds PlaneGraph from primal and dual data."""
    dual_vertex_count = dual_data["vertex_count"]
    dual_adj_1based = dual_data["adjacency_list"]

    embedding = GraphConverter.to_zero_based_embedding(dual_adj_1based)

    edge_multiplicity: Dict[Tuple[int, int], int] = {}
    unique_edges: set = set()

    for v, neighbors in embedding.items():
        for u in neighbors:
            edge = (min(v, u), max(v, u))
            unique_edges.add(edge)

    for edge in unique_edges:
        u, v = edge
        count = embedding[u].count(v)
        edge_multiplicity[edge] = count

    edges = tuple(sorted(unique_edges))
    faces = GraphConverter.extract_faces(embedding, edge_multiplicity)

    primal_adj_1based = primal_data["adjacency_list"]
    primal_num_vertices = primal_data["vertex_count"]
    primal_embedding = GraphConverter.to_zero_based_embedding(primal_adj_1based)
    primal_faces = GraphConverter.extract_faces(primal_embedding)

    return PlaneGraph(
        num_vertices=dual_vertex_count,
        edges=edges,
        edge_multiplicity=edge_multiplicity,
        embedding=embedding,
        faces=faces,
        primal_num_vertices=primal_num_vertices,
        primal_embedding=primal_embedding,
        primal_faces=primal_faces,
        graph_id=graph_id,
    )


def enumerate_plane_graphs(
    dual_vertex_count: int,
    max_count: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
) -> List[PlaneGraph]:
    """Enumerates all n-vertex 4-regular planar multigraphs."""
    if verbose:
        print(f"[Plantri] Enumerating {dual_vertex_count}-vertex 4-regular planar multigraphs...")

    enumerator = QuadrangulationEnumerator()
    graphs = []

    for graph_id, (primal_data, dual_data) in enumerate(enumerator.generate_pairs(dual_vertex_count)):
        graph = _build_plane_graph(primal_data, dual_data, graph_id)

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


def iter_plane_graphs(
    dual_vertex_count: int,
    validate: bool = True,
) -> Iterator[PlaneGraph]:
    """Iterates over PlaneGraph objects."""
    enumerator = QuadrangulationEnumerator()

    for graph_id, (primal_data, dual_data) in enumerate(
        enumerator.generate_pairs(dual_vertex_count)
    ):
        graph = _build_plane_graph(primal_data, dual_data, graph_id)

        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                continue

        yield graph


def save_graphs_to_cache(
    graphs: List[PlaneGraph],
    filepath: str,
    use_pickle: bool = False,
) -> None:
    """Saves graph list to cache file."""
    if use_pickle:
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(graphs, f)
    else:
        data = [graph.to_dict() for graph in graphs]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def load_graphs_from_cache(
    filepath: str,
    use_pickle: bool = False,
) -> List[PlaneGraph]:
    """Loads graph list from cache file."""
    if use_pickle:
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PlaneGraph.from_dict(d) for d in data]


def main() -> None:
    """CLI entry point for plantri graph enumeration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate 4-regular planar multigraphs via plantri",
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

    graphs = enumerate_plane_graphs(
        args.n,
        max_count=args.max,
        verbose=args.verbose,
    )

    print(f"\nTotal: {len(graphs)} {args.n}-vertex 4-regular planar multigraphs")

    for graph in graphs:
        print(f"\n{'='*50}")
        print(f"Graph #{graph.graph_id}")
        print(f"  Vertices: {graph.num_vertices}")
        print(f"  Edges: {graph.num_simple_edges} unique, {graph.num_edges} total")
        print(f"  Double edges: {len(graph.double_edges)}")
        print(f"  Faces: {graph.num_faces}")

        print("  Embedding (CW order, 0-based):")
        for v in range(graph.num_vertices):
            print(f"    {v}: {list(graph.embedding[v])}")

        if args.show_faces:
            print("  Faces:")
            for i, face in enumerate(graph.faces):
                face_type = "digon" if len(face) == 2 else f"{len(face)}-gon"
                print(f"    F{i}: {list(face)} ({face_type})")

    if args.export:
        save_graphs_to_cache(graphs, args.export, use_pickle=args.pickle)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
