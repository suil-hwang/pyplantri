from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .types import Embedding, HalfEdge

Edge = tuple[int, int]
Face = tuple[int, ...]
_STATE_FIELDS = frozenset(("twin", "graph_id"))


def _simple_twin_map(embedding: Embedding, graph_name: str) -> dict[HalfEdge, HalfEdge]:
    """Return the unique twin involution of a strict simple rotation system."""
    if type(embedding) is not tuple or any(
        type(neighbors) is not tuple
        or any(type(neighbor) is not int for neighbor in neighbors)
        for neighbors in embedding
    ):
        raise TypeError(f"{graph_name} embedding must be tuple[tuple[int, ...], ...]")

    vertex_count = len(embedding)
    half_edge_by_arc: dict[tuple[int, int], HalfEdge] = {}
    for vertex, neighbors in enumerate(embedding):
        for slot, neighbor in enumerate(neighbors):
            if not 0 <= neighbor < vertex_count:
                raise ValueError(f"{graph_name} neighbor out of range: vertex={vertex}, neighbor={neighbor}")
            if neighbor == vertex:
                raise ValueError(f"{graph_name} has a self-loop at vertex {vertex}")
            if (vertex, neighbor) in half_edge_by_arc:
                raise ValueError(f"{graph_name} has parallel edges: {vertex}->{neighbor}")
            half_edge_by_arc[vertex, neighbor] = (vertex, slot)

    twin_map: dict[HalfEdge, HalfEdge] = {}
    for (vertex, neighbor), half_edge in half_edge_by_arc.items():
        twin = half_edge_by_arc.get((neighbor, vertex))
        if twin is None:
            raise ValueError(f"{graph_name} nonreciprocal edge: {vertex}->{neighbor}")
        twin_map[half_edge] = twin

    reached: set[int] = set()
    pending = [0] if vertex_count else []
    while pending:
        vertex = pending.pop()
        if vertex in reached:
            continue
        reached.add(vertex)
        pending.extend(
            neighbor for neighbor in embedding[vertex] if neighbor not in reached
        )
    if len(reached) != vertex_count:
        raise ValueError(f"{graph_name} disconnected: reached={len(reached)}/{vertex_count}")
    return twin_map


def _right_face_half_edge_cycles(
    embedding: Embedding,
    twin_map: Mapping[HalfEdge, HalfEdge],
) -> tuple[tuple[HalfEdge, ...], ...]:
    """Return right-face orbits for exterior-view clockwise rotations."""
    visited: set[HalfEdge] = set()
    cycles: list[tuple[HalfEdge, ...]] = []
    for vertex, neighbors in enumerate(embedding):
        for slot in range(len(neighbors)):
            start = (vertex, slot)
            if start in visited:
                continue
            cycle: list[HalfEdge] = []
            half_edge = start
            while half_edge not in visited:
                visited.add(half_edge)
                cycle.append(half_edge)
                twin_vertex, twin_slot = twin_map[half_edge]
                half_edge = (
                    twin_vertex,
                    (twin_slot - 1) % len(embedding[twin_vertex]),
                )
            if half_edge != start:
                raise ValueError("right-face traversal merged distinct orbits")
            cycles.append(tuple(cycle))
    return tuple(cycles)


@dataclass(frozen=True, slots=True)
class _DerivedTopology:
    dual_embedding: Embedding
    dual_face_darts: tuple[Face, ...]
    dual_faces: tuple[Face, ...]
    dual_edge_multiplicity: Mapping[Edge, int]
    primal_embedding: Embedding
    primal_faces: tuple[Face, ...]


def _derive_topology(twin: bytes) -> _DerivedTopology:
    vertex_count = len(twin) // 4
    dual_embedding: Embedding = tuple(
        tuple(twin[dart] // 4 for dart in range(4 * vertex, 4 * vertex + 4))
        for vertex in range(vertex_count)
    )

    face_of_dart = [-1] * len(twin)
    dual_face_darts: list[Face] = []
    for start in range(len(twin)):
        if face_of_dart[start] != -1:
            continue
        face_index = len(dual_face_darts)
        cycle: list[int] = []
        dart = start
        while face_of_dart[dart] == -1:
            face_of_dart[dart] = face_index
            cycle.append(dart)
            opposite = twin[dart]
            dart = 4 * (opposite // 4) + (opposite % 4 - 1) % 4
        if dart != start:
            raise ValueError("right-face traversal merged distinct orbits")
        dual_face_darts.append(tuple(cycle))

    multiplicity: dict[Edge, int] = {}
    for dart, opposite in enumerate(twin):
        if dart > opposite:
            continue
        endpoints = dart // 4, opposite // 4
        edge = min(endpoints), max(endpoints)
        multiplicity[edge] = multiplicity.get(edge, 0) + 1

    face_darts = tuple(dual_face_darts)
    return _DerivedTopology(
        dual_embedding=dual_embedding,
        dual_face_darts=face_darts,
        dual_faces=tuple(
            tuple(dart // 4 for dart in face) for face in face_darts
        ),
        dual_edge_multiplicity=MappingProxyType(dict(sorted(multiplicity.items()))),
        primal_embedding=tuple(
            tuple(face_of_dart[twin[dart]] for dart in face)
            for face in face_darts
        ),
        primal_faces=tuple(
            tuple(
                face_of_dart[twin[dart]]
                for dart in range(4 * vertex, 4 * vertex + 4)
            )
            for vertex in range(vertex_count)
        ),
    )


@dataclass(frozen=True, slots=True)
class QuarticPlaneMap:
    """Loop-free 4-regular plane map with exterior-view clockwise rotations."""

    twin: bytes
    graph_id: int = field(default=0, compare=False)
    _topology_cache: _DerivedTopology | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate compact field types and the twin encoding."""
        if type(self.twin) is not bytes:
            raise TypeError("twin must be bytes")
        if type(self.graph_id) is not int:
            raise TypeError("graph_id must be int")

        dart_count = len(self.twin)
        if dart_count == 0 or dart_count % 4:
            raise ValueError("dart count must be a positive multiple of 4")
        if dart_count > 256:
            raise ValueError("byte-valued twin supports at most 256 darts")
        for dart, opposite in enumerate(self.twin):
            if opposite >= dart_count:
                raise ValueError(f"twin out of range: {dart}->{opposite}")
            if opposite == dart:
                raise ValueError(f"self-twin dart: {dart}")
            if self.twin[opposite] != dart:
                raise ValueError(f"twin is not involutive: {dart}->{opposite}")

    @classmethod
    def from_primal_embedding(
        cls,
        primal_embedding: Embedding,
        graph_id: int = 0,
        *,
        validate: bool = True,
    ) -> QuarticPlaneMap:
        """Dualize a strict simple exterior-CW spherical quadrangulation."""
        if type(validate) is not bool:
            raise TypeError("validate must be bool")
        primal_twins = _simple_twin_map(primal_embedding, "primal")
        vertex_count = len(primal_embedding)
        if vertex_count < 5:
            raise ValueError(f"primal vertex count must be >= 5, got {vertex_count}")

        primal_face_darts = _right_face_half_edge_cycles(
            primal_embedding,
            primal_twins,
        )
        if len(primal_face_darts) != vertex_count - 2:
            raise ValueError(f"primal face count mismatch: {len(primal_face_darts)}!={vertex_count - 2}")
        for face_index, face in enumerate(primal_face_darts):
            vertices = tuple(vertex for vertex, _ in face)
            if len(vertices) != 4 or len(set(vertices)) != 4:
                raise ValueError(f"primal face {face_index} is not a simple quadrilateral: {vertices}")
        if len(primal_face_darts) > 64:
            raise ValueError("byte-valued twin supports at most 64 dual vertices")

        dual_dart_by_primal = {
            primal_dart: 4 * face_index + slot
            for face_index, face in enumerate(primal_face_darts)
            for slot, primal_dart in enumerate(face)
        }
        twin = bytes(
            dual_dart_by_primal[primal_twins[primal_dart]]
            for face in primal_face_darts
            for primal_dart in face
        )
        plane_map = cls(twin=twin, graph_id=graph_id)
        if validate:
            valid, errors = plane_map.validate()
            if not valid:
                raise ValueError("invalid primal-induced map: " + "; ".join(errors))
        return plane_map

    def __getstate__(self) -> dict[str, Any]:
        """Return compact pickle state without derived topology."""
        return {"twin": self.twin, "graph_id": self.graph_id}

    def __setstate__(self, state: Any) -> None:
        """Restore exact compact state, validate its core, and clear the cache."""
        if type(state) is not dict:
            raise TypeError(f"QuarticPlaneMap pickle state must be dict, got {type(state).__name__}")
        actual_fields = set(state)
        if actual_fields != _STATE_FIELDS:
            raise ValueError(f"QuarticPlaneMap pickle state keys mismatch: missing={sorted(_STATE_FIELDS - actual_fields)}, extra={sorted(actual_fields - _STATE_FIELDS)}")
        object.__setattr__(self, "twin", state["twin"])
        object.__setattr__(self, "graph_id", state["graph_id"])
        object.__setattr__(self, "_topology_cache", None)
        self.__post_init__()

    def _topology(self) -> _DerivedTopology:
        """Return topology derived and cached on first access."""
        topology = self._topology_cache
        if topology is None:
            topology = _derive_topology(self.twin)
            object.__setattr__(self, "_topology_cache", topology)
        return topology

    @property
    def dual_num_vertices(self) -> int:
        """Return the number of dual vertices."""
        return len(self.twin) // 4

    @property
    def dual_embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW dual rotations."""
        return self._topology().dual_embedding

    @property
    def dual_faces(self) -> tuple[Face, ...]:
        """Return dual right-face vertex cycles."""
        return self._topology().dual_faces

    @property
    def dual_num_faces(self) -> int:
        """Return the number of dual face orbits."""
        return len(self.dual_faces)

    @property
    def dual_edge_multiplicity(self) -> Mapping[Edge, int]:
        """Return read-only multiplicities of canonical dual support edges."""
        return self._topology().dual_edge_multiplicity

    @property
    def dual_support_edges(self) -> tuple[Edge, ...]:
        """Return canonical dual support edges in lexicographic order."""
        return tuple(self.dual_edge_multiplicity)

    @property
    def double_edges(self) -> frozenset[Edge]:
        """Return dual support edges of multiplicity two."""
        return frozenset(
            edge
            for edge, multiplicity in self.dual_edge_multiplicity.items()
            if multiplicity == 2
        )

    def dual_topology_profile(self) -> tuple[int, int, tuple[int, ...]]:
        """Return support, double-edge, and descending face-size counts."""
        topology = self._topology_cache or _derive_topology(self.twin)
        multiplicities = topology.dual_edge_multiplicity.values()
        return (
            len(topology.dual_edge_multiplicity),
            sum(multiplicity == 2 for multiplicity in multiplicities),
            tuple(sorted(map(len, topology.dual_faces), reverse=True)),
        )

    @property
    def primal_num_vertices(self) -> int:
        """Return the number of implied primal vertices."""
        return self.dual_num_faces

    @property
    def primal_embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW primal rotations."""
        return self._topology().primal_embedding

    @property
    def primal_faces(self) -> tuple[Face, ...]:
        """Return implied primal face cycles indexed by dual vertex."""
        return self._topology().primal_faces

    @property
    def dual_vertex_to_primal_face(self) -> tuple[int, ...]:
        """Return the identity map from dual vertices to primal faces."""
        return tuple(range(self.dual_num_vertices))

    @property
    def primal_vertex_to_dual_face(self) -> tuple[int, ...]:
        """Return the identity map from primal vertices to dual faces."""
        return tuple(range(self.primal_num_vertices))

    @staticmethod
    def vertex(dart: int) -> int:
        """Return the source vertex of a dart."""
        return dart // 4

    @staticmethod
    def next_at_vertex(dart: int) -> int:
        """Return the next dart in its exterior-view-CW vertex rotation."""
        vertex, slot = divmod(dart, 4)
        return 4 * vertex + (slot + 1) % 4

    @staticmethod
    def prev_at_vertex(dart: int) -> int:
        """Return the previous dart in its exterior-view-CW vertex rotation."""
        vertex, slot = divmod(dart, 4)
        return 4 * vertex + (slot - 1) % 4

    def right_face_next(self, dart: int) -> int:
        """Return the next dart along the face on a dart's right."""
        return self.prev_at_vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the dual vertex across a dart's edge."""
        return self.vertex(self.twin[dart])

    def validate(self) -> tuple[bool, list[str]]:
        """Certify the SQS quartic-map and implied-primal invariants."""
        errors: list[str] = []
        vertex_count = self.dual_num_vertices
        if vertex_count < 3:
            errors.append(f"dual_num_vertices must be >= 3, got {vertex_count}")
        if self.graph_id < 0:
            errors.append(f"graph_id must be non-negative, got {self.graph_id}")

        # Keep catalogue audits from populating lazy topology caches.
        topology = self._topology_cache or _derive_topology(self.twin)
        for dart, opposite in enumerate(self.twin):
            if dart // 4 == opposite // 4:
                errors.append(f"dual loop at dart pair {dart}<->{opposite}")
                break

        reached: set[int] = set()
        pending = [0]
        while pending:
            vertex = pending.pop()
            if vertex in reached:
                continue
            reached.add(vertex)
            pending.extend(
                neighbor
                for neighbor in topology.dual_embedding[vertex]
                if neighbor not in reached
            )
        if len(reached) != vertex_count:
            errors.append(f"dual disconnected: reached={len(reached)}/{vertex_count}")

        expected_face_count = vertex_count + 2
        if len(topology.dual_faces) != expected_face_count:
            errors.append(f"dual face count/Euler mismatch: {len(topology.dual_faces)}!={expected_face_count}")
        for face_index, face in enumerate(topology.dual_faces):
            if len(set(face)) != len(face):
                errors.append(f"dual face {face_index} repeats a vertex: {face}")

        for edge, multiplicity in topology.dual_edge_multiplicity.items():
            if multiplicity not in (1, 2):
                errors.append(f"dual edge {edge} has unsupported multiplicity {multiplicity}")

        digons = Counter(
            tuple(sorted(face))
            for face in topology.dual_faces
            if len(face) == 2 and face[0] != face[1]
        )
        expected_digons = Counter(
            {
                edge: 1
                for edge, multiplicity in topology.dual_edge_multiplicity.items()
                if multiplicity == 2
            }
        )
        if digons != expected_digons:
            errors.append("dual digon/double-edge mismatch")

        # A spherical quartic dual has quadrilateral primal faces; only simplicity remains.
        try:
            _simple_twin_map(topology.primal_embedding, "implied primal")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

        return not errors, errors
