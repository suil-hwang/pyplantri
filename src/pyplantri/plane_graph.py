# src/pyplantri/plane_graph.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .types import Embedding, FaceCycle, SupportEdge

_PICKLE_STATE_FIELDS = frozenset(("twin", "graph_id"))
_VERTEX_SHIFT = 2
_DARTS_PER_VERTEX = 1 << _VERTEX_SHIFT
_SLOT_MASK = _DARTS_PER_VERTEX - 1


@dataclass(frozen=True, slots=True)
class _DerivedTopology:
    dual_embedding: Embedding
    dual_faces: tuple[FaceCycle, ...]
    dual_edge_multiplicity: Mapping[SupportEdge, int]
    primal_embedding: Embedding
    primal_faces: tuple[FaceCycle, ...]


@dataclass(frozen=True, slots=True)
class QuarticPlaneMap:
    """Byte-encoded quartic rotation-system core."""

    twin: bytes
    graph_id: int = field(default=0, compare=False)
    _topology_cache: _DerivedTopology | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate compact record fields and the twin encoding."""
        if type(self.twin) is not bytes:
            raise TypeError("twin must be bytes")
        if type(self.graph_id) is not int:
            raise TypeError("graph_id must be int")
        if self.graph_id < 0:
            raise ValueError(f"graph_id must be non-negative, got {self.graph_id}")

        dart_count = len(self.twin)
        if dart_count == 0 or dart_count % _DARTS_PER_VERTEX:
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
        audit: bool = True,
    ) -> QuarticPlaneMap:
        """Dualize a supported simple quadrangulation and optionally audit it."""
        if type(audit) is not bool:
            raise TypeError("audit must be bool")
        if type(primal_embedding) is not tuple or any(
            type(neighbors) is not tuple
            or any(type(neighbor) is not int for neighbor in neighbors)
            for neighbors in primal_embedding
        ):
            raise TypeError("primal embedding must be tuple[tuple[int, ...], ...]")

        vertex_count = len(primal_embedding)
        # Flatten (vertex, slot) half-edges into row-major primal dart indices.
        first_dart = [0] * (vertex_count + 1)
        source: list[int] = []
        target: list[int] = []
        dart_by_directed_edge: dict[int, int] = {}
        for vertex, neighbors in enumerate(primal_embedding):
            for neighbor in neighbors:
                if not 0 <= neighbor < vertex_count:
                    raise ValueError(f"primal neighbor out of range: vertex={vertex}, neighbor={neighbor}")
                if neighbor == vertex:
                    raise ValueError(f"primal has a self-loop at vertex {vertex}")
                key = vertex * vertex_count + neighbor
                if key in dart_by_directed_edge:
                    raise ValueError(f"primal has parallel edges: {vertex}->{neighbor}")
                dart_by_directed_edge[key] = len(target)
                source.append(vertex)
                target.append(neighbor)
            first_dart[vertex + 1] = len(target)

        # Reciprocity turns edge reversal into the fixed-point-free involution alpha.
        reverse = [-1] * len(target)
        for dart, neighbor in enumerate(target):
            vertex = source[dart]
            opposite = dart_by_directed_edge.get(neighbor * vertex_count + vertex)
            if opposite is None:
                raise ValueError(f"primal nonreciprocal edge: {vertex}->{neighbor}")
            reverse[dart] = opposite

        # Connectivity is independent of the later Euler and face-shape checks.
        reached = bytearray(vertex_count)
        reached_count = 0
        pending_vertices = [0] if vertex_count else []
        while pending_vertices:
            vertex = pending_vertices.pop()
            if reached[vertex]:
                continue
            reached[vertex] = 1
            reached_count += 1
            pending_vertices.extend(primal_embedding[vertex])
        if reached_count != vertex_count:
            raise ValueError(f"primal disconnected: reached={reached_count}/{vertex_count}")
        if vertex_count < 5:
            raise ValueError(f"primal vertex count must be >= 5, got {vertex_count}")

        # Right-face orbits of phi = sigma^-1 composed with alpha become dual vertices.
        dual_dart_by_primal_dart = [-1] * len(target)
        next_dual_dart = 0
        face_count = 0
        # Retain only the first bad face while counting every orbit for Euler's formula.
        first_invalid_face: tuple[int, tuple[int, ...]] | None = None
        for start_dart in range(len(target)):
            if dual_dart_by_primal_dart[start_dart] != -1:
                continue
            face_vertices: list[int] = []
            dart = start_dart
            while dual_dart_by_primal_dart[dart] == -1:
                dual_dart_by_primal_dart[dart] = next_dual_dart
                next_dual_dart += 1
                face_vertices.append(source[dart])
                opposite = reverse[dart]
                target_vertex = target[dart]
                first_target_dart = first_dart[target_vertex]
                dart = (
                    opposite - 1
                    if opposite != first_target_dart
                    else first_dart[target_vertex + 1] - 1
                )
            face_cycle = tuple(face_vertices)
            if first_invalid_face is None and (len(face_cycle) != _DARTS_PER_VERTEX or len(set(face_cycle)) != _DARTS_PER_VERTEX):
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
        dual_twin = bytearray(_DARTS_PER_VERTEX * face_count)
        for dart, opposite in enumerate(reverse):
            dual_twin[dual_dart_by_primal_dart[dart]] = (dual_dart_by_primal_dart[opposite])

        plane_map = cls(twin=bytes(dual_twin), graph_id=graph_id)
        # The optional topology audit enforces target-class invariants beyond dualization.
        if audit:
            valid, errors = plane_map.audit_sqs_topology()
            if not valid:
                raise ValueError("invalid primal-induced map: " + "; ".join(errors))
        return plane_map

    @classmethod
    def _from_plantri_embedding(
        cls,
        primal_embedding: Embedding,
        graph_id: int,
    ) -> QuarticPlaneMap:
        """Convert one plantri record through the mandatory audited boundary."""
        return cls.from_primal_embedding(
            primal_embedding,
            graph_id=graph_id,
            audit=True,
        )

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

    def _compute_dual_edge_multiplicities(self) -> dict[SupportEdge, int]:
        """Count normalized support-edge multiplicities without caching."""
        twin = self.twin
        dual_vertex_count = len(twin) >> _VERTEX_SHIFT
        multiplicity_by_key: dict[int, int] = {}
        for dart, twin_dart in enumerate(twin):
            if dart > twin_dart:
                continue
            # dart <= twin_dart makes the base-n key endpoint-normalized.
            key = ((dart >> _VERTEX_SHIFT) * dual_vertex_count + (twin_dart >> _VERTEX_SHIFT))
            multiplicity_by_key[key] = multiplicity_by_key.get(key, 0) + 1
        return {
            divmod(key, dual_vertex_count): multiplicity
            for key, multiplicity in sorted(multiplicity_by_key.items())
        }

    def _compute_derived_topology(self) -> _DerivedTopology:
        """Compute immutable dual and primal topology views without caching."""
        twin = self.twin
        # The implicit quartic rotation groups darts 4v, ..., 4v+3 at dual vertex v.
        dual_embedding: Embedding = tuple(
            tuple(
                twin[dart] >> _VERTEX_SHIFT
                for dart in range(base, base + _DARTS_PER_VERTEX)
            )
            for base in range(0, len(twin), _DARTS_PER_VERTEX)
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
                dart = ((twin_dart & ~_SLOT_MASK) | ((twin_dart - 1) & _SLOT_MASK))
            dual_face_dart_orbits.append(tuple(face_dart_orbit))

        # Count each alpha-paired edge once after normalizing its endpoint order.
        dual_edge_multiplicity = self._compute_dual_edge_multiplicities()

        # Dual face orbits become primal vertices; dual vertices become primal faces.
        return _DerivedTopology(
            dual_embedding=dual_embedding,
            dual_faces=tuple(
                tuple(dart >> _VERTEX_SHIFT for dart in face_darts)
                for face_darts in dual_face_dart_orbits
            ),
            dual_edge_multiplicity=MappingProxyType(dual_edge_multiplicity),
            primal_embedding=tuple(
                tuple(dual_face_index_by_dart[twin[dart]] for dart in face_darts)
                for face_darts in dual_face_dart_orbits
            ),
            primal_faces=tuple(
                tuple(
                    dual_face_index_by_dart[twin[dart]]
                    for dart in range(base, base + _DARTS_PER_VERTEX)
                )
                for base in range(0, len(twin), _DARTS_PER_VERTEX)
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
        return len(self.twin) >> _VERTEX_SHIFT

    @property
    def dual_embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW dual rotations."""
        topology = self._topology_cache
        if topology is not None:
            return topology.dual_embedding
        twin = self.twin
        return tuple(
            tuple(
                opposite >> _VERTEX_SHIFT
                for opposite in twin[base : base + _DARTS_PER_VERTEX]
            )
            for base in range(0, len(twin), _DARTS_PER_VERTEX)
        )

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

    def _dual_edge_cardinality_profile(self) -> tuple[int, int]:
        """Return support- and double-edge counts without retaining topology."""
        topology = self._topology_cache
        multiplicities = (
            topology.dual_edge_multiplicity
            if topology is not None
            else self._compute_dual_edge_multiplicities()
        )
        return (
            len(multiplicities),
            sum(multiplicity == 2 for multiplicity in multiplicities.values()),
        )

    def dual_topology_profile(self) -> tuple[int, int, tuple[int, ...]]:
        """Return support-edge count, double-edge count, and descending face sizes."""
        topology = self._topology_cache
        if topology is not None:
            multiplicities = topology.dual_edge_multiplicity
            return (
                len(multiplicities),
                sum(value == 2 for value in multiplicities.values()),
                tuple(sorted(map(len, topology.dual_faces), reverse=True)),
            )

        twin = self.twin
        dual_vertex_count = len(twin) >> _VERTEX_SHIFT
        visited = bytearray(len(twin))
        multiplicity_by_key: dict[int, int] = {}
        face_sizes: list[int] = []
        for start_dart in range(len(twin)):
            if visited[start_dart]:
                continue
            dart = start_dart
            face_size = 0
            while not visited[dart]:
                visited[dart] = 1
                opposite = twin[dart]
                if dart < opposite:
                    u = dart >> _VERTEX_SHIFT
                    v = opposite >> _VERTEX_SHIFT
                    if u > v:
                        u, v = v, u
                    key = u * dual_vertex_count + v
                    multiplicity_by_key[key] = multiplicity_by_key.get(key, 0) + 1
                dart = ((opposite & ~_SLOT_MASK) | ((opposite - 1) & _SLOT_MASK))
                face_size += 1
            face_sizes.append(face_size)

        face_sizes.sort(reverse=True)
        return (
            len(multiplicity_by_key),
            sum(value == 2 for value in multiplicity_by_key.values()),
            tuple(face_sizes),
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
    def dual_vertex_to_primal_face(self) -> range:
        """Return the identity map from dual vertices to primal faces."""
        return range(self.dual_num_vertices)

    @property
    def primal_vertex_to_dual_face(self) -> range:
        """Return the identity map from primal vertices to dual faces."""
        return range(self.primal_num_vertices)

    @staticmethod
    def vertex(dart: int) -> int:
        """Return the source vertex of a valid dart."""
        return dart >> _VERTEX_SHIFT

    @staticmethod
    def next_at_vertex(dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        return (dart & ~_SLOT_MASK) | ((dart + 1) & _SLOT_MASK)

    @staticmethod
    def prev_at_vertex(dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        return (dart & ~_SLOT_MASK) | ((dart - 1) & _SLOT_MASK)

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on a dart's right."""
        return self.prev_at_vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])

    def audit_sqs_topology(self) -> tuple[bool, list[str]]:
        """Audit supported SQS topology invariants of the dual and implied primal."""
        errors: list[str] = []
        dual_vertex_count = self.dual_num_vertices
        if dual_vertex_count < 3:
            errors.append(f"dual_num_vertices must be >= 3, got {dual_vertex_count}")

        # Keep catalogue audits from populating lazy topology caches.
        topology = self._topology_cache or self._compute_derived_topology()
        for dart, twin_dart in enumerate(self.twin):
            if dart >> _VERTEX_SHIFT == twin_dart >> _VERTEX_SHIFT:
                errors.append(f"dual loop at dart pair {dart}<->{twin_dart}")
                break

        reached = bytearray(dual_vertex_count)
        reached[0] = 1
        reached_count = 1
        pending_vertices = [0]
        while pending_vertices:
            vertex = pending_vertices.pop()
            base = vertex << _VERTEX_SHIFT
            for dart in range(base, base + _DARTS_PER_VERTEX):
                neighbor = self.twin[dart] >> _VERTEX_SHIFT
                if not reached[neighbor]:
                    reached[neighbor] = 1
                    reached_count += 1
                    pending_vertices.append(neighbor)
        if reached_count != dual_vertex_count:
            errors.append(f"dual disconnected: reached={reached_count}/{dual_vertex_count}")

        expected_dual_face_count = dual_vertex_count + 2
        if len(topology.dual_faces) != expected_dual_face_count:
            errors.append(f"dual face count/Euler mismatch: {len(topology.dual_faces)}!={expected_dual_face_count}")
        for face_index, face in enumerate(topology.dual_faces):
            if len(set(face)) != len(face):
                errors.append(f"dual face {face_index} repeats a vertex: {face}")

        for edge, multiplicity in topology.dual_edge_multiplicity.items():
            if multiplicity not in (1, 2):
                errors.append(f"dual edge {edge} has unsupported multiplicity {multiplicity}")

        digons = sorted(
            tuple(sorted(face))
            for face in topology.dual_faces
            if len(face) == 2 and face[0] != face[1]
        )
        double_edges = sorted(
            edge
            for edge, multiplicity in topology.dual_edge_multiplicity.items()
            if multiplicity == 2
        )
        if digons != double_edges:
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
