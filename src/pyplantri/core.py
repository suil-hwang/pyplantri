# src/pyplantri/core.py
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix


# Constant for ASCII character parsing optimization.
_ORD_A = ord('a')


def _find_plantri_exe() -> Path:
    """Finds the plantri executable path."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"
    
    # Package bin folder (installed).
    pkg_bin = Path(__file__).parent / "bin" / exe_name
    if pkg_bin.exists():
        return pkg_bin
    
    # scikit-build-core build folder (editable install, dev mode).
    # Path: src/pyplantri/core.py -> src -> project_root.
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    if build_dir.exists():
        for tag_dir in build_dir.iterdir():
            if tag_dir.is_dir():
                # Release folder (Visual Studio build).
                release_exe = tag_dir / "Release" / exe_name
                if release_exe.exists():
                    return release_exe
                # MinGW/Unix build.
                direct_exe = tag_dir / exe_name
                if direct_exe.exists():
                    return direct_exe
    
    # Default path for error messages.
    return Path(__file__).parent / "bin" / exe_name


_PLANTRI_EXE = _find_plantri_exe()


class PlantriError(Exception):
    """Plantri execution failure.

    Attributes:
        message: Explanation of the error.
    """


class Plantri:
    """Wrapper for the plantri executable."""

    def __init__(self, executable: Optional[Path] = None) -> None:
        """Initializes Plantri with the executable path."""
        self.executable = Path(executable) if executable else _PLANTRI_EXE
        if not self.executable.exists():
            raise FileNotFoundError(
                f"plantri executable not found: {self.executable}\n"
                "Run 'pip install -e .' first."
            )

    def run(
        self,
        n_vertices: int,
        options: Optional[List[str]] = None,
        output_format: Literal["planar_code", "ascii", "adjacency"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = [str(self.executable)]

        if options:
            cmd.extend(options)

        # Set output format flag.
        if output_format == "ascii":
            if "-a" not in (options or []):
                cmd.append("-a")
        elif output_format == "adjacency":
            if "-A" not in (options or []):
                cmd.append("-A")

        cmd.append(str(n_vertices))

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise PlantriError(f"plantri execution failed: {error_msg}")
        except FileNotFoundError:
            raise PlantriError(
                f"plantri executable not found: {self.executable}"
            )

    def count(
        self,
        n_vertices: int,
        graph_type: Literal["triangulation", "quadrangulation"] = "triangulation",
        connectivity: int = 3,
        timeout: float = 3600.0,
    ) -> int:
        """Counts graphs without generating output."""
        options = ["-u"]  # No stdout output (count only via stderr).

        if graph_type == "quadrangulation":
            options.append("-q")

        options.append(f"-c{connectivity}")

        try:
            result = subprocess.run(
                [str(self.executable)] + options + [str(n_vertices)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Check exit code
            if result.returncode != 0:
                raise PlantriError(
                    f"plantri exited with code {result.returncode}.\n"
                    f"stderr: {result.stderr}\nstdout: {result.stdout}"
                )

            # Parse count from stderr (e.g., "1 graphs written to stdout" or "1 quadrangulations generated").
            for line in result.stderr.split("\n"):
                match = re.search(r"(\d+)\s+(?:graph|triangulation|quadrangulation)", line.lower())
                if match:
                    return int(match.group(1))

            # If we reach here, parsing failed
            raise PlantriError(
                f"Could not parse graph count from plantri output.\n"
                f"stderr: {result.stderr}\nstdout: {result.stdout}"
            )

        except subprocess.TimeoutExpired as e:
            raise PlantriError(
                f"plantri execution timed out after {timeout} seconds "
                f"for n={n_vertices}, graph_type={graph_type}"
            ) from e

        except FileNotFoundError as e:
            raise PlantriError(
                f"plantri executable not found at {self.executable}"
            ) from e

        except PlantriError:
            # Re-raise our own errors
            raise

        except Exception as e:
            # Catch-all for unexpected errors
            raise PlantriError(
                f"Unexpected error during plantri execution: {type(e).__name__}: {e}"
            ) from e

    def generate_graphs(
        self,
        n_vertices: int,
        graph_type: Literal["triangulation", "quadrangulation", "cubic"] = "triangulation",
        connectivity: Optional[int] = None,
        dual: bool = False,
        minimum_degree: Optional[int] = None,
        bipartite: bool = False,
    ) -> Iterator[str]:
        """Generates plane graphs as an iterator."""
        options = ["-a"]  # ASCII output.

        if graph_type == "quadrangulation":
            options.append("-q")
        elif graph_type == "cubic":
            options.append("-b")

        if connectivity is not None:
            options.append(f"-c{connectivity}")
        if dual:
            options.append("-d")
        if minimum_degree is not None:
            options.append(f"-m{minimum_degree}")
        if bipartite:
            options.append("-bp")

        output = self.run(n_vertices, options, output_format="ascii")

        # Parse ASCII output.
        current_graph_lines: List[str] = []
        for line in output.decode(errors="replace").split("\n"):
            line = line.rstrip()
            if line.startswith("Graph"):
                if current_graph_lines:
                    yield "\n".join(current_graph_lines)
                current_graph_lines = [line]
            elif line:
                current_graph_lines.append(line)

        if current_graph_lines:
            yield "\n".join(current_graph_lines)

    def generate_planar_code(
        self,
        n_vertices: int,
        graph_type: Literal["triangulation", "quadrangulation"] = "triangulation",
        connectivity: Optional[int] = None,
        dual: bool = False,
    ) -> bytes:
        """Generates graphs in planar_code binary format."""
        options = []

        if graph_type == "quadrangulation":
            options.append("-q")
        if connectivity is not None:
            options.append(f"-c{connectivity}")
        if dual:
            options.append("-d")

        return self.run(n_vertices, options, output_format="planar_code")


class GraphConverter:
    """Utility for converting plantri output to various formats."""

    @staticmethod
    def parse_ascii_to_edge_map(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[Tuple[int, int], int]:
        """Converts ASCII output to edge multiplicity map."""
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        if len(parts) < 2:
            return {}

        adjacency_str = parts[1]
        vertex_neighbor_lists = adjacency_str.split(",")

        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        index_offset = 0 if is_one_based else 1

        for vertex_idx, neighbors_str in enumerate(vertex_neighbor_lists):
            source_vertex = vertex_idx + (1 if is_one_based else 0)
            for char in neighbors_str:
                target_vertex = ord(char) - _ORD_A + 1 - index_offset
                # Canonical edge key (u < v) - count one direction only.
                if source_vertex < target_vertex:
                    edge_multiplicity[(source_vertex, target_vertex)] += 1

        return dict(edge_multiplicity)

    @staticmethod
    def parse_ascii_to_adjacency_list(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[int, List[int]]:
        """Converts ASCII output to adjacency list."""
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        if len(parts) < 2:
            return {}

        adjacency_str = parts[1]
        vertex_neighbor_lists = adjacency_str.split(",")

        adjacency_list: Dict[int, List[int]] = {}
        index_offset = 0 if is_one_based else 1

        for vertex_idx, neighbors_str in enumerate(vertex_neighbor_lists):
            vertex = vertex_idx + (1 if is_one_based else 0)
            adjacency_list[vertex] = [
                ord(c) - _ORD_A + 1 - index_offset for c in neighbors_str
            ]

        return adjacency_list

    @staticmethod
    def adjacency_to_edge_multiplicity(
        adjacency_list: Dict[int, List[int]],
        is_one_based: bool = True
    ) -> Dict[Tuple[int, int], int]:
        """Converts adjacency list to edge multiplicity map."""
        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        index_offset = 1 if is_one_based else 0

        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                u = min(vertex - index_offset, neighbor - index_offset)
                v = max(vertex - index_offset, neighbor - index_offset)
                edge_multiplicity[(u, v)] += 1

        # Divide by 2 since edges are counted from both endpoints.
        return {k: v // 2 for k, v in edge_multiplicity.items()}

    @staticmethod
    def to_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> np.ndarray:
        """Converts edge multiplicity map to dense adjacency matrix."""
        if not edge_multiplicity:
            return np.zeros((vertex_count, vertex_count), dtype=np.int32)

        # Vectorized indexing for performance.
        edges = list(edge_multiplicity.keys())
        sources = np.array([e[0] for e in edges], dtype=np.intp)
        targets = np.array([e[1] for e in edges], dtype=np.intp)
        multiplicities = np.array(list(edge_multiplicity.values()), dtype=np.int32)

        adjacency_matrix = np.zeros((vertex_count, vertex_count), dtype=np.int32)
        # Assign symmetric matrix at once using fancy indexing.
        adjacency_matrix[sources, targets] = multiplicities
        adjacency_matrix[targets, sources] = multiplicities
        return adjacency_matrix

    @staticmethod
    def to_sparse_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> csr_matrix:
        """Converts edge multiplicity map to CSR sparse matrix."""
        if not edge_multiplicity:
            return csr_matrix((vertex_count, vertex_count), dtype=np.int32)

        # Prepare COO format data (add both directions for symmetric matrix).
        edges = list(edge_multiplicity.keys())
        multiplicities = list(edge_multiplicity.values())

        row = [e[0] for e in edges] + [e[1] for e in edges]
        col = [e[1] for e in edges] + [e[0] for e in edges]
        data = multiplicities + multiplicities

        return csr_matrix(
            (data, (row, col)),
            shape=(vertex_count, vertex_count),
            dtype=np.int32
        )

    @staticmethod
    def to_gurobi_start_dict(
        edge_multiplicity: Dict[Tuple[int, int], int],
        var_prefix: str = "x"
    ) -> Dict[str, int]:
        """Creates a Gurobi MIP start dictionary."""
        return {
            f"{var_prefix}[{source},{target}]": multiplicity
            for (source, target), multiplicity in edge_multiplicity.items()
        }

    @staticmethod
    def get_graph_stats(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> dict:
        """Computes graph statistics from edge multiplicity map."""
        degrees = [0] * vertex_count
        single_edge_count = 0
        double_edge_count = 0
        total_edge_count = 0

        for (source, target), multiplicity in edge_multiplicity.items():
            degrees[source] += multiplicity
            degrees[target] += multiplicity
            total_edge_count += multiplicity
            if multiplicity == 1:
                single_edge_count += 1
            elif multiplicity == 2:
                double_edge_count += 1

        is_regular = len(set(degrees)) == 1 if degrees else False
        regularity = degrees[0] if is_regular and degrees else None

        return {
            "vertex_count": vertex_count,
            "edge_count": total_edge_count,
            "single_edge_count": single_edge_count,
            "double_edge_count": double_edge_count,
            "degrees": degrees,
            "is_regular": is_regular,
            "regularity": regularity,
        }

    @staticmethod
    def to_zero_based_embedding(
        adjacency_list: Dict[int, List[int]],
    ) -> Dict[int, Tuple[int, ...]]:
        """Converts 1-based adjacency list to 0-based embedding."""
        embedding: Dict[int, Tuple[int, ...]] = {}
        for vertex, neighbors in adjacency_list.items():
            vertex_idx = vertex - 1
            neighbor_tuple = tuple(u - 1 for u in neighbors)
            embedding[vertex_idx] = neighbor_tuple
        return embedding


    @staticmethod
    def extract_faces(
        embedding: Dict[int, Tuple[int, ...]],
        edge_multiplicity: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> Tuple[Tuple[int, ...], ...]:
        """Extracts faces from a 0-based embedding."""
        if not embedding:
            return tuple()

        # Infer multiplicities when not provided so digons are still recovered.
        effective_multiplicity: Dict[Tuple[int, int], int]
        if edge_multiplicity is None:
            raw_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for v, neighbors in embedding.items():
                for u in neighbors:
                    edge = (min(v, u), max(v, u))
                    raw_counts[edge] += 1
            effective_multiplicity = {
                edge: count // 2 for edge, count in raw_counts.items()
            }
        else:
            effective_multiplicity = edge_multiplicity

        visited_half_edges: Set[Tuple[int, int]] = set()  # (vertex, local slot index)
        faces: List[Tuple[int, ...]] = []

        total_half_edges = sum(len(neighbors) for neighbors in embedding.values())
        max_face_steps = max(4, total_half_edges)

        for start_v in embedding:
            for start_i in range(len(embedding[start_v])):
                if (start_v, start_i) in visited_half_edges:
                    continue

                face: List[int] = []
                current_v = start_v
                current_i = start_i
                steps = 0

                while True:
                    half_edge = (current_v, current_i)
                    if half_edge in visited_half_edges:
                        break

                    visited_half_edges.add(half_edge)
                    face.append(current_v)

                    next_v = embedding[current_v][current_i]
                    next_neighbors = embedding.get(next_v)
                    if not next_neighbors:
                        break

                    # Ambiguous for parallel edges, but deterministic.
                    try:
                        incoming_idx = next_neighbors.index(current_v)
                    except ValueError:
                        break

                    next_next_v = next_neighbors[(incoming_idx - 1) % len(next_neighbors)]
                    try:
                        next_i = embedding[next_v].index(next_next_v)
                    except ValueError:
                        break

                    current_v, current_i = next_v, next_i

                    if current_v == start_v and current_i == start_i:
                        break

                    steps += 1
                    if steps > max_face_steps:
                        break

                if len(face) >= 2:
                    faces.append(tuple(face))

        # Drop malformed non-digon loops (repeated vertices) from ambiguous fallback.
        cleaned_faces: List[Tuple[int, ...]] = []
        for face in faces:
            if len(face) == 2 or len(set(face)) == len(face):
                cleaned_faces.append(face)
        faces = cleaned_faces

        # Add digon faces for parallel edges.
        for (u, v), multiplicity in effective_multiplicity.items():
            if multiplicity >= 2:
                digon = (u, v)
                if digon not in faces and (v, u) not in faces:
                    faces.append(digon)

        return tuple(faces)

    @staticmethod
    def extract_faces_with_twins(
        embedding: Dict[int, Tuple[int, ...]],
        twin_map: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[Tuple[int, ...], ...]:
        """Extract faces accurately using position-based half-edge traversal."""
        visited: set = set()
        faces: List[Tuple[int, ...]] = []

        # Calculate max iterations for infinite loop detection
        if not embedding:
            return tuple(faces)

        max_iterations = len(embedding) * max(
            len(neighbors) for neighbors in embedding.values()
        )

        for v in sorted(embedding.keys()):
            deg_v = len(embedding[v])
            for i in range(deg_v):
                if (v, i) in visited:
                    continue

                face: List[int] = []
                curr_v, curr_i = v, i
                iterations = 0

                while (curr_v, curr_i) not in visited:
                    iterations += 1
                    if iterations > max_iterations:
                        raise RuntimeError(
                            f"Face traversal exceeded {max_iterations} iterations. "
                            f"Possible infinite loop or invalid twin_map. "
                            f"Current face: {face}"
                        )

                    visited.add((curr_v, curr_i))
                    face.append(curr_v)

                    # Twin
                    if (curr_v, curr_i) not in twin_map:
                        raise ValueError(
                            f"Half-edge ({curr_v}, {curr_i}) not found in twin_map. "
                            f"Embedding may be invalid or twin_map incomplete. "
                            f"Face so far: {face}"
                        )
                    twin_v, twin_i = twin_map[(curr_v, curr_i)]

                    # Predecessor in CW order
                    deg = len(embedding[twin_v])
                    curr_v = twin_v
                    curr_i = (twin_i - 1) % deg

                if len(face) >= 2:
                    faces.append(tuple(face))

        return tuple(faces)

    @staticmethod
    def is_4_regular(adjacency_list: Dict[int, List[int]]) -> bool:
        """Checks if graph is 4-regular."""
        if not adjacency_list:
            return False
        return all(len(neighbors) == 4 for neighbors in adjacency_list.values())

    @staticmethod
    def is_loop_free(adjacency_list: Dict[int, List[int]]) -> bool:
        """Checks if graph has no self-loops."""
        return all(v not in neighbors for v, neighbors in adjacency_list.items())


class QuadrangulationEnumerator:
    """Enumerates 4-regular plane multigraphs (duals of simple quadrangulations)."""

    OPTIONS = ["-q", "-c2", "-m2", "-T"]

    def __init__(self) -> None:
        """Initializes the SQS enumerator with a Plantri instance."""
        self._plantri = Plantri()

    def generate_pairs(
        self,
        dual_vertex_count: int
    ) -> Iterator[Tuple[Dict, Dict]]:
        """Generates primal (Q) and dual (Q*) graph pairs."""
        for line in self.iter_raw(dual_vertex_count):
            primal, dual = self.parse_double_code(line)
            if primal["adjacency_list"] and dual["adjacency_list"]:
                yield primal, dual

    def count(self, dual_vertex_count: int) -> int:
        """Counts the number of non-isomorphic SQS structures."""
        return sum(1 for _ in self.iter_raw(dual_vertex_count))

    def iter_raw(self, dual_vertex_count: int) -> Iterator[str]:
        """Generates raw double_code output lines."""
        # Euler's formula for plane graphs: V - E + F = 2
        # For quadrangulations: primal_vertices = dual_vertices + 2
        primal_vertex_count = dual_vertex_count + 2
        output = self._plantri.run(primal_vertex_count, self.OPTIONS)

        for line in output.decode(errors="replace").split("\n"):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                yield stripped

    @staticmethod
    def parse_double_code(double_code_line: str) -> Tuple[Dict, Dict]:
        """Parses plantri double_code output to adjacency lists with twin maps."""
        parts = double_code_line.split()
        if len(parts) < 2:
            empty: Dict = {"vertex_count": 0, "adjacency_list": {}, "twin_map": {}}
            return empty, empty

        # Parse primal graph.
        primal_vertex_count = int(parts[0])

        # Collect primal vertex edge lists.
        idx = 1
        primal_edge_lists = []
        while idx < len(parts) and not parts[idx][0].isdigit():
            primal_edge_lists.append(parts[idx])
            idx += 1

        # Parse dual graph.
        if idx >= len(parts):
            empty = {"vertex_count": 0, "adjacency_list": {}, "twin_map": {}}
            return empty, empty

        dual_vertex_count = int(parts[idx])
        idx += 1

        dual_edge_lists = parts[idx:]

        # Build adjacency lists AND twin maps from edge name mappings.
        primal_adj, primal_twins = (
            QuadrangulationEnumerator._build_adjacency_and_twins(primal_edge_lists)
        )
        dual_adj, dual_twins = (
            QuadrangulationEnumerator._build_adjacency_and_twins(dual_edge_lists)
        )

        primal_data = {
            "vertex_count": primal_vertex_count,
            "adjacency_list": primal_adj,
            "twin_map": primal_twins,
        }
        dual_data = {
            "vertex_count": dual_vertex_count,
            "adjacency_list": dual_adj,
            "twin_map": dual_twins,
        }

        return primal_data, dual_data

    @staticmethod
    def _build_adjacency_and_twins(
        edge_lists: List[str],
    ) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], Tuple[int, int]]]:
        """Builds adjacency list AND half-edge twin mapping from edge lists."""
        # Collect (vertex, position) pairs where each edge name appears.
        edge_name_to_half_edges: Dict[str, List[Tuple[int, int]]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            for pos, edge_name in enumerate(edges_str):
                slots = edge_name_to_half_edges.get(edge_name)
                if slots is None:
                    edge_name_to_half_edges[edge_name] = [(vertex_idx, pos)]
                else:
                    slots.append((vertex_idx, pos))

        # Build adjacency list.
        adjacency: Dict[int, List[int]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            neighbors: List[int] = []
            for edge_name in edges_str:
                half_edges = edge_name_to_half_edges.get(edge_name)
                if half_edges is None or len(half_edges) != 2:
                    raise ValueError(
                        f"Edge '{edge_name}' appears "
                        f"{0 if half_edges is None else len(half_edges)} times "
                        f"(expected 2). Invalid plantri output or corrupted data."
                    )
                (v1, _), (v2, _) = half_edges
                if v1 == v2 == vertex_idx:
                    neighbors.append(vertex_idx)  # Loop edge
                else:
                    neighbors.append(v2 if v1 == vertex_idx else v1)
            adjacency[vertex_idx] = neighbors

        # Twin mapping: match two half-edges sharing the same edge name.
        twin_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for half_edges in edge_name_to_half_edges.values():
            if len(half_edges) == 2:
                twin_map[half_edges[0]] = half_edges[1]
                twin_map[half_edges[1]] = half_edges[0]

        return adjacency, twin_map
