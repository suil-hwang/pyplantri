# src/pyplantri/plane_graph.py
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .types import Embedding, FaceCycle, HalfEdge, SupportEdge

_PICKLE_STATE_FIELDS = frozenset(("twin", "graph_id"))


@dataclass(frozen=True, slots=True)
class _DerivedTopology:
    dual_embedding: Embedding
    dual_faces: tuple[FaceCycle, ...]
    dual_edge_multiplicity: Mapping[SupportEdge, int]
    primal_embedding: Embedding
    primal_faces: tuple[FaceCycle, ...]


@dataclass(frozen=True, slots=True)
class QuarticPlaneMap:
    """Compact 4-regular rotation-system record with exterior-view-CW rotations."""

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
        if type(primal_embedding) is not tuple or any(
            type(neighbors) is not tuple
            or any(type(neighbor) is not int for neighbor in neighbors)
            for neighbors in primal_embedding
        ):
            raise TypeError("primal embedding must be tuple[tuple[int, ...], ...]")

        vertex_count = len(primal_embedding)
        # Simplicity makes each directed endpoint pair identify one half-edge.
        half_edge_by_directed_edge: dict[tuple[int, int], HalfEdge] = {}
        for vertex, neighbors in enumerate(primal_embedding):
            for slot, neighbor in enumerate(neighbors):
                if not 0 <= neighbor < vertex_count:
                    raise ValueError(f"primal neighbor out of range: vertex={vertex}, neighbor={neighbor}")
                if neighbor == vertex:
                    raise ValueError(f"primal has a self-loop at vertex {vertex}")
                if (vertex, neighbor) in half_edge_by_directed_edge:
                    raise ValueError(f"primal has parallel edges: {vertex}->{neighbor}")
                half_edge_by_directed_edge[vertex, neighbor] = (vertex, slot)

        # Reciprocity turns edge reversal into the fixed-point-free involution alpha.
        for vertex, neighbor in half_edge_by_directed_edge:
            if (neighbor, vertex) not in half_edge_by_directed_edge:
                raise ValueError(f"primal nonreciprocal edge: {vertex}->{neighbor}")

        # Connectivity is independent of the later Euler and face-shape checks.
        reached_vertices: set[int] = set()
        pending_vertices = [0] if vertex_count else []
        while pending_vertices:
            vertex = pending_vertices.pop()
            if vertex in reached_vertices:
                continue
            reached_vertices.add(vertex)
            pending_vertices.extend(primal_embedding[vertex])
        if len(reached_vertices) != vertex_count:
            raise ValueError(f"primal disconnected: reached={len(reached_vertices)}/{vertex_count}")
        if vertex_count < 5:
            raise ValueError(f"primal vertex count must be >= 5, got {vertex_count}")

        # Right-face orbits of phi = sigma^-1 composed with alpha become dual vertices.
        dual_dart_by_half_edge: dict[HalfEdge, int] = {}
        face_count = 0
        # Retain only the first bad face while counting every orbit for Euler's formula.
        first_invalid_face: tuple[int, tuple[int, ...]] | None = None
        for vertex, neighbors in enumerate(primal_embedding):
            for slot in range(len(neighbors)):
                start_half_edge = (vertex, slot)
                if start_half_edge in dual_dart_by_half_edge:
                    continue
                face_vertices: list[int] = []
                half_edge = start_half_edge
                while half_edge not in dual_dart_by_half_edge:
                    dual_dart_by_half_edge[half_edge] = len(dual_dart_by_half_edge)
                    half_edge_vertex, half_edge_slot = half_edge
                    face_vertices.append(half_edge_vertex)
                    twin_vertex, twin_slot = half_edge_by_directed_edge[
                        primal_embedding[half_edge_vertex][half_edge_slot],
                        half_edge_vertex,
                    ]
                    half_edge = (
                        twin_vertex,
                        (twin_slot - 1) % len(primal_embedding[twin_vertex]),
                    )
                face_cycle = tuple(face_vertices)
                if first_invalid_face is None and (
                    len(face_cycle) != 4 or len(set(face_cycle)) != 4
                ):
                    first_invalid_face = face_count, face_cycle
                face_count += 1

        expected_face_count = vertex_count - 2
        if face_count != expected_face_count:
            raise ValueError(f"primal face count mismatch: {face_count}!={expected_face_count}")
        if first_invalid_face is not None:
            face_index, face_cycle = first_invalid_face
            raise ValueError(f"primal face {face_index} is not a simple quadrilateral: {face_cycle}")
        if face_count > 64:
            raise ValueError("byte-valued twin supports at most 64 dual vertices")

        # Four darts per face make discovery order exactly d = 4f + i.
        dual_twin = bytearray(4 * face_count)
        for (vertex, slot), dual_dart in dual_dart_by_half_edge.items():
            twin_half_edge = half_edge_by_directed_edge[
                primal_embedding[vertex][slot], vertex
            ]
            dual_twin[dual_dart] = dual_dart_by_half_edge[twin_half_edge]

        plane_map = cls(twin=bytes(dual_twin), graph_id=graph_id)
        # The optional full audit enforces target-class invariants beyond dualization.
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
        if actual_fields != _PICKLE_STATE_FIELDS:
            raise ValueError(f"QuarticPlaneMap pickle state keys mismatch: missing={sorted(_PICKLE_STATE_FIELDS - actual_fields)}, extra={sorted(actual_fields - _PICKLE_STATE_FIELDS)}")
        object.__setattr__(self, "twin", state["twin"])
        object.__setattr__(self, "graph_id", state["graph_id"])
        object.__setattr__(self, "_topology_cache", None)
        self.__post_init__()

    def _compute_derived_topology(self) -> _DerivedTopology:
        """Compute immutable dual and primal topology views without caching."""
        twin = self.twin
        # The implicit quartic rotation groups darts 4v, ..., 4v+3 at dual vertex v.
        dual_vertex_count = len(twin) // 4
        dual_embedding: Embedding = tuple(
            tuple(twin[dart] // 4 for dart in range(4 * vertex, 4 * vertex + 4))
            for vertex in range(dual_vertex_count)
        )

        # Right faces are the orbits of phi = sigma^-1 composed with alpha.
        dual_face_index_by_dart = [-1] * len(twin)
        dual_face_dart_orbits: list[tuple[int, ...]] = []
        for start_dart in range(len(twin)):
            if dual_face_index_by_dart[start_dart] != -1:
                continue
            dual_face_index = len(dual_face_dart_orbits)
            face_dart_orbit: list[int] = []
            dart = start_dart
            while dual_face_index_by_dart[dart] == -1:
                dual_face_index_by_dart[dart] = dual_face_index
                face_dart_orbit.append(dart)
                twin_dart = twin[dart]
                dart = 4 * (twin_dart // 4) + (twin_dart % 4 - 1) % 4
            if dart != start_dart:
                raise ValueError("right-face traversal merged distinct orbits")
            dual_face_dart_orbits.append(tuple(face_dart_orbit))

        # Count each alpha-paired edge once after normalizing its endpoint order.
        dual_edge_multiplicity: dict[SupportEdge, int] = {}
        for dart, twin_dart in enumerate(twin):
            if dart > twin_dart:
                continue
            endpoints = dart // 4, twin_dart // 4
            support_edge = min(endpoints), max(endpoints)
            dual_edge_multiplicity[support_edge] = (
                dual_edge_multiplicity.get(support_edge, 0) + 1
            )

        # Dual face orbits become primal vertices; dual vertices become primal faces.
        return _DerivedTopology(
            dual_embedding=dual_embedding,
            dual_faces=tuple(
                tuple(dart // 4 for dart in face_darts)
                for face_darts in dual_face_dart_orbits
            ),
            dual_edge_multiplicity=MappingProxyType(
                dict(sorted(dual_edge_multiplicity.items()))
            ),
            primal_embedding=tuple(
                tuple(dual_face_index_by_dart[twin[dart]] for dart in face_darts)
                for face_darts in dual_face_dart_orbits
            ),
            primal_faces=tuple(
                tuple(
                    dual_face_index_by_dart[twin[dart]]
                    for dart in range(4 * vertex, 4 * vertex + 4)
                )
                for vertex in range(dual_vertex_count)
            ),
        )

    def _derived_topology(self) -> _DerivedTopology:
        """Return dual and primal topology views, deriving and caching them on first access."""
        topology = self._topology_cache
        if topology is None:
            topology = self._compute_derived_topology()
            object.__setattr__(self, "_topology_cache", topology)
        return topology

    @property
    def dual_num_vertices(self) -> int:
        """Return the number of dual vertices."""
        return len(self.twin) // 4

    @property
    def dual_embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW dual rotations."""
        return self._derived_topology().dual_embedding

    @property
    def dual_faces(self) -> tuple[FaceCycle, ...]:
        """Return dual right-face vertex cycles."""
        return self._derived_topology().dual_faces

    @property
    def dual_num_faces(self) -> int:
        """Return the number of dual face orbits."""
        return len(self.dual_faces)

    @property
    def dual_edge_multiplicity(self) -> Mapping[SupportEdge, int]:
        """Return multiplicities of normalized dual support edges."""
        return self._derived_topology().dual_edge_multiplicity

    @property
    def dual_support_edges(self) -> tuple[SupportEdge, ...]:
        """Return normalized dual support edges in lexicographic order."""
        return tuple(self.dual_edge_multiplicity)

    @property
    def double_edges(self) -> frozenset[SupportEdge]:
        """Return dual support edges of multiplicity two."""
        return frozenset(
            edge
            for edge, multiplicity in self.dual_edge_multiplicity.items()
            if multiplicity == 2
        )

    def dual_topology_profile(self) -> tuple[int, int, tuple[int, ...]]:
        """Return support-edge count, double-edge count, and descending face sizes."""
        topology = self._topology_cache or self._compute_derived_topology()
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
        return self._derived_topology().primal_embedding

    @property
    def primal_faces(self) -> tuple[FaceCycle, ...]:
        """Return implied primal face cycles indexed by dual vertex."""
        return self._derived_topology().primal_faces

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
        """Return the source vertex of a valid dart."""
        return dart // 4

    @staticmethod
    def next_at_vertex(dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        vertex, slot = divmod(dart, 4)
        return 4 * vertex + (slot + 1) % 4

    @staticmethod
    def prev_at_vertex(dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        vertex, slot = divmod(dart, 4)
        return 4 * vertex + (slot - 1) % 4

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on a dart's right."""
        return self.prev_at_vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])

    def validate(self) -> tuple[bool, list[str]]:
        """Validate SQS topology invariants of the dual and implied primal."""
        errors: list[str] = []
        dual_vertex_count = self.dual_num_vertices
        if dual_vertex_count < 3:
            errors.append(f"dual_num_vertices must be >= 3, got {dual_vertex_count}")
        if self.graph_id < 0:
            errors.append(f"graph_id must be non-negative, got {self.graph_id}")

        # Keep catalogue audits from populating lazy topology caches.
        topology = self._topology_cache or self._compute_derived_topology()
        for dart, twin_dart in enumerate(self.twin):
            if dart // 4 == twin_dart // 4:
                errors.append(f"dual loop at dart pair {dart}<->{twin_dart}")
                break

        reached_vertices: set[int] = set()
        pending_vertices = [0]
        while pending_vertices:
            vertex = pending_vertices.pop()
            if vertex in reached_vertices:
                continue
            reached_vertices.add(vertex)
            pending_vertices.extend(
                neighbor
                for neighbor in topology.dual_embedding[vertex]
                if neighbor not in reached_vertices
            )
        if len(reached_vertices) != dual_vertex_count:
            errors.append(f"dual disconnected: reached={len(reached_vertices)}/{dual_vertex_count}")

        expected_dual_face_count = dual_vertex_count + 2
        if len(topology.dual_faces) != expected_dual_face_count:
            errors.append(f"dual face count/Euler mismatch: {len(topology.dual_faces)}!={expected_dual_face_count}")
        for face_index, face in enumerate(topology.dual_faces):
            if len(set(face)) != len(face):
                errors.append(f"dual face {face_index} repeats a vertex: {face}")

        for edge, multiplicity in topology.dual_edge_multiplicity.items():
            if multiplicity not in (1, 2):
                errors.append(f"dual edge {edge} has unsupported multiplicity {multiplicity}")

        digon_counts = Counter(
            tuple(sorted(face))
            for face in topology.dual_faces
            if len(face) == 2 and face[0] != face[1]
        )
        expected_digon_counts = Counter(
            {
                edge: 1
                for edge, multiplicity in topology.dual_edge_multiplicity.items()
                if multiplicity == 2
            }
        )
        if digon_counts != expected_digon_counts:
            errors.append("dual digon/double-edge mismatch")

        # Derived primal type, range, reciprocity, and connectivity are algebraic.
        for primal_vertex, neighbors in enumerate(topology.primal_embedding):
            seen_neighbors: set[int] = set()
            for neighbor in neighbors:
                if neighbor == primal_vertex:
                    errors.append(f"implied primal has a self-loop at vertex {primal_vertex}")
                    break
                if neighbor in seen_neighbors:
                    errors.append(f"implied primal has parallel edges: {primal_vertex}->{neighbor}")
                    break
                seen_neighbors.add(neighbor)
            else:
                continue
            break

        return not errors, errors
