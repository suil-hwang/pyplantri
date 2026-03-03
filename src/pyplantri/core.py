# src/pyplantri/core.py
import os
import re
import subprocess
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple, Union, cast

import numpy as np
from scipy.sparse import csr_matrix


# Constant for ASCII character parsing optimization.
_ORD_A = ord('a')
EdgeLabel = Union[str, int]
GraphType = Literal["triangulation", "quadrangulation", "cubic"]
GraphClass = Literal[
    "triangulation",
    "quadrangulation",
    "cubic",
    "eulerian",
    "eulerian_triangulation",
    "bipartite_plane",
]


def _is_ascii_digit_byte(value: int) -> bool:
    """Returns True if value is an ASCII digit byte ('0'..'9')."""
    return 48 <= value <= 57


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
    _COUNT_INCOMPATIBLE_OUTPUT_OPTIONS = frozenset({"-a", "-g", "-s", "-E", "-T", "-u"})
    _GRAPH_CLASS_OPTIONS: Dict[str, Tuple[str, ...]] = {
        # Simple triangulations.
        "triangulation": tuple(),
        # Simple quadrangulations.
        "quadrangulation": ("-q",),
        # Cubic plane graphs are duals of triangulations.
        "cubic": ("-d",),
        # plantri -b : Eulerian triangulations.
        "eulerian": ("-b",),
        # plantri -bp : simple bipartite plane graphs (separate class dispatch).
        "bipartite_plane": ("-bp",),
    }
    _GRAPH_CLASS_ALIASES: Dict[str, str] = {
        "eulerian_triangulation": "eulerian",
    }

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
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format=output_format,
        )

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode(errors="replace") if e.stderr else str(e)
            raise PlantriError(f"plantri execution failed: {error_msg}") from e
        except FileNotFoundError as e:
            raise PlantriError(
                f"plantri executable not found: {self.executable}"
            ) from e

    def _build_command(
        self,
        n_vertices: int,
        *,
        options: Optional[List[str]],
        output_format: Literal["planar_code", "ascii"],
    ) -> List[str]:
        """Builds a plantri command line for the given options."""
        if output_format not in ("planar_code", "ascii"):
            raise ValueError(
                f"Unsupported output_format={output_format!r}. "
                "Use 'planar_code' or 'ascii'. "
                "Note: plantri '-A' is an Apollonian-generation option, "
                "not an adjacency output format."
            )

        cmd = [str(self.executable)]
        if options:
            cmd.extend(options)

        # Set output format flag.
        if output_format == "ascii" and "-a" not in (options or []):
            cmd.append("-a")

        cmd.append(str(n_vertices))
        return cmd

    def iter_stdout_lines(
        self,
        n_vertices: int,
        options: Optional[List[str]] = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> Iterator[bytes]:
        """Streams non-empty stdout lines from plantri without buffering all output."""
        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format=output_format,
        )

        with tempfile.TemporaryFile() as stderr_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriError(
                    f"plantri executable not found: {self.executable}"
                ) from e

            if proc.stdout is None:
                proc.kill()
                proc.wait()
                raise PlantriError("Failed to capture plantri stdout stream.")

            fully_consumed = False
            try:
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if line:
                        yield line
                fully_consumed = True
            finally:
                proc.stdout.close()

                if fully_consumed:
                    return_code = proc.wait()
                else:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    return_code = proc.returncode if proc.returncode is not None else 0

                stderr_file.seek(0)
                stderr_text = stderr_file.read().decode("utf-8", errors="replace")
                if fully_consumed and return_code != 0:
                    stderr_excerpt = stderr_text[:4000]
                    raise PlantriError(
                        f"plantri execution failed (exit code {return_code}): "
                        f"{stderr_excerpt}"
                    )

    def _normalize_graph_class(self, graph_class: str) -> str:
        """Validate and normalize graph class alias names."""
        canonical = self._GRAPH_CLASS_ALIASES.get(graph_class, graph_class)
        if canonical not in self._GRAPH_CLASS_OPTIONS:
            allowed = ", ".join(
                sorted(self._GRAPH_CLASS_OPTIONS.keys() | self._GRAPH_CLASS_ALIASES.keys())
            )
            raise ValueError(
                f"Unsupported graph_class={graph_class!r}. "
                f"Allowed values: {allowed}."
            )
        return canonical

    def _resolve_graph_class(
        self,
        *,
        graph_class: Optional[GraphClass],
        graph_type: GraphType,
        bipartite: bool,
    ) -> str:
        """Resolve legacy graph_type/bipartite into a canonical graph_class."""
        if graph_class is not None:
            if bipartite:
                raise ValueError(
                    "bipartite=True cannot be combined with graph_class. "
                    "Use graph_class='bipartite_plane' instead."
                )
            return self._normalize_graph_class(graph_class)

        if graph_type == "triangulation":
            if bipartite:
                warnings.warn(
                    "bipartite=True is deprecated. "
                    "Use graph_class='bipartite_plane'.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                return "bipartite_plane"
            return "triangulation"

        if graph_type == "quadrangulation":
            if bipartite:
                raise ValueError(
                    "graph_type='quadrangulation' is incompatible with bipartite=True "
                    "(plantri: -q and -bp are incompatible). "
                    "Use graph_class to choose a valid class."
                )
            return "quadrangulation"

        if graph_type == "cubic":
            if bipartite:
                raise ValueError(
                    "Ambiguous legacy options: graph_type='cubic' with bipartite=True. "
                    "Use graph_class='eulerian' with dual=True for bipartite cubic output, "
                    "or graph_class='bipartite_plane' for general bipartite plane graphs."
                )
            warnings.warn(
                "graph_type='cubic' now means cubic graphs (triangulation dual, -d). "
                "To get plantri -b behavior, use graph_class='eulerian'.",
                DeprecationWarning,
                stacklevel=3,
            )
            return "cubic"

        raise ValueError(f"Unsupported graph_type={graph_type!r}.")

    def _build_generation_options(
        self,
        *,
        graph_class: str,
        connectivity: Optional[int],
        dual: bool,
        minimum_degree: Optional[int],
    ) -> List[str]:
        """Build generation options from canonical graph class and constraints."""
        options = list(self._GRAPH_CLASS_OPTIONS[graph_class])

        if connectivity is not None:
            options.append(f"-c{connectivity}")
        if minimum_degree is not None:
            options.append(f"-m{minimum_degree}")
        if dual and "-d" not in options:
            options.append("-d")

        return options

    def count(
        self,
        n_vertices: int,
        graph_type: GraphType = "triangulation",
        connectivity: Optional[int] = 3,
        dual: bool = False,
        minimum_degree: Optional[int] = None,
        bipartite: bool = False,
        timeout: float = 3600.0,
        *,
        graph_class: Optional[GraphClass] = None,
    ) -> int:
        """Count output objects for a selected plantri graph class.

        Args:
            graph_type: Legacy API selector. Kept for compatibility.
                - ``"triangulation"``
                - ``"quadrangulation"``
                - ``"cubic"`` (corrected to mean triangulation dual: ``-d``)
            graph_class: Recommended explicit class selector:
                - ``"triangulation"``
                - ``"quadrangulation"``
                - ``"cubic"`` (triangulation dual)
                - ``"eulerian"`` / ``"eulerian_triangulation"`` (plantri ``-b``)
                - ``"bipartite_plane"`` (plantri ``-bp``)
            bipartite: Legacy flag. Deprecated; use ``graph_class='bipartite_plane'``.
        """
        resolved_graph_class = self._resolve_graph_class(
            graph_class=graph_class,
            graph_type=graph_type,
            bipartite=bipartite,
        )
        options = self._build_generation_options(
            graph_class=resolved_graph_class,
            connectivity=connectivity,
            dual=dual,
            minimum_degree=minimum_degree,
        )

        return self.count_from_options(
            n_vertices,
            options=options,
            timeout=timeout,
        )

    def count_from_options(
        self,
        n_vertices: int,
        options: Optional[List[str]] = None,
        timeout: float = 3600.0,
    ) -> int:
        """Counts graphs with arbitrary generation options via plantri ``-u``."""
        normalized_options = [
            opt
            for opt in (options or [])
            if opt not in self._COUNT_INCOMPATIBLE_OUTPUT_OPTIONS
        ]

        try:
            result = subprocess.run(
                [str(self.executable)] + normalized_options + ["-u", str(n_vertices)],
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
                match = re.search(
                    r"(\d+)\s+.*\b(?:graph|triangulation|quadrangulation)s?\b",
                    line.lower(),
                )
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
                f"for n={n_vertices}, options={normalized_options}"
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
        graph_type: GraphType = "triangulation",
        connectivity: Optional[int] = None,
        dual: bool = False,
        minimum_degree: Optional[int] = None,
        bipartite: bool = False,
        *,
        graph_class: Optional[GraphClass] = None,
    ) -> Iterator[str]:
        """Generate ASCII graphs for a selected plantri graph class.

        ``graph_class`` is the recommended selector; ``graph_type`` and
        ``bipartite`` are legacy compatibility parameters.
        """
        resolved_graph_class = self._resolve_graph_class(
            graph_class=graph_class,
            graph_type=graph_type,
            bipartite=bipartite,
        )
        options = ["-a"]
        options.extend(
            self._build_generation_options(
                graph_class=resolved_graph_class,
                connectivity=connectivity,
                dual=dual,
                minimum_degree=minimum_degree,
            )
        )

        # Parse text output without UTF-8 assumptions. plantri emits byte labels.
        # Decode per line with latin-1 only when yielding strings.
        current_graph_lines: List[bytes] = []
        for line in self.iter_stdout_lines(
            n_vertices,
            options,
            output_format="ascii",
        ):

            # Legacy format with "Graph ..." headers.
            if line.startswith(b"Graph"):
                if current_graph_lines:
                    yield "\n".join(
                        chunk.decode("latin-1") for chunk in current_graph_lines
                    )
                current_graph_lines = [line]
                continue

            # Standard plantri -a line: one graph per line, starts with vertex count.
            if _is_ascii_digit_byte(line[0]) and not current_graph_lines:
                yield line.decode("latin-1")
                continue

            # Continuation lines for legacy block formats.
            if current_graph_lines:
                current_graph_lines.append(line)

        if current_graph_lines:
            yield "\n".join(chunk.decode("latin-1") for chunk in current_graph_lines)

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
        is_one_based: Optional[bool] = None,
        *,
        output_one_based: bool = False,
    ) -> Dict[Tuple[int, int], int]:
        """Convert adjacency list to edge multiplicity map.

        Args:
            adjacency_list: Vertex -> neighbor list adjacency map.
            is_one_based: Whether input vertex indices are 1-based.
                When ``None`` (default), infer indexing from observed vertex
                labels: presence of ``0`` means 0-based, otherwise presence of
                ``1`` means 1-based.
            output_one_based: Whether output edge keys should be 1-based.
                Historical default behavior is 0-based output regardless of
                ``is_one_based``.
        """
        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        if is_one_based is None:
            observed_indices: Set[int] = set(int(v) for v in adjacency_list.keys())
            for neighbors in adjacency_list.values():
                observed_indices.update(int(u) for u in neighbors)

            if not observed_indices:
                is_one_based = False
            elif min(observed_indices) < 0:
                raise ValueError(
                    "adjacency_list contains negative vertex indices, cannot infer index base."
                )
            elif 0 in observed_indices:
                is_one_based = False
            elif 1 in observed_indices:
                is_one_based = True
            else:
                raise ValueError(
                    "Cannot infer index base from adjacency_list. "
                    "Pass is_one_based=True or is_one_based=False explicitly."
                )

        input_offset = 1 if is_one_based else 0
        output_offset = 1 if output_one_based else 0

        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                u = min(vertex - input_offset, neighbor - input_offset)
                v = max(vertex - input_offset, neighbor - input_offset)
                if output_offset:
                    u += output_offset
                    v += output_offset
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
        primal_vertex_count = dual_vertex_count + 2
        return self._plantri.count_from_options(
            primal_vertex_count,
            options=self.OPTIONS,
        )

    def iter_raw(self, dual_vertex_count: int) -> Iterator[bytes]:
        """Generates raw double_code output lines as bytes."""
        # Euler's formula for plane graphs: V - E + F = 2
        # For quadrangulations: primal_vertices = dual_vertices + 2
        primal_vertex_count = dual_vertex_count + 2
        for line in self._plantri.iter_stdout_lines(
            primal_vertex_count,
            self.OPTIONS,
        ):
            if _is_ascii_digit_byte(line[0]):
                yield line

    @staticmethod
    def parse_double_code(double_code_line: Union[str, bytes]) -> Tuple[Dict, Dict]:
        """Parses plantri double_code output to adjacency lists with twin maps."""
        parts = double_code_line.split()
        if len(parts) < 2:
            empty: Dict = {
                "vertex_count": 0,
                "adjacency_list": {},
                "twin_map": {},
                "edge_label_pairs": {},
            }
            return empty, empty

        # Parse first graph section.
        first_vertex_count = int(parts[0])

        # Collect first graph edge lists.
        idx = 1
        first_edge_lists: List[Union[str, bytes]] = []
        while idx < len(parts):
            head = parts[idx][0]
            if isinstance(head, str):
                is_vertex_count = head.isdigit()
            else:
                is_vertex_count = _is_ascii_digit_byte(head)
            if is_vertex_count:
                break
            first_edge_lists.append(parts[idx])
            idx += 1

        # Parse second graph section.
        if idx >= len(parts):
            empty = {
                "vertex_count": 0,
                "adjacency_list": {},
                "twin_map": {},
                "edge_label_pairs": {},
            }
            return empty, empty

        second_vertex_count = int(parts[idx])
        idx += 1

        second_edge_lists = cast(List[Union[str, bytes]], list(parts[idx:]))

        # Build adjacency lists AND twin maps from edge name mappings.
        first_adj, first_twins, first_edge_label_pairs = (
            QuadrangulationEnumerator._build_adjacency_and_twins(first_edge_lists)
        )
        second_adj, second_twins, second_edge_label_pairs = (
            QuadrangulationEnumerator._build_adjacency_and_twins(second_edge_lists)
        )

        first_data = {
            "vertex_count": first_vertex_count,
            "adjacency_list": first_adj,
            "twin_map": first_twins,
            "edge_label_pairs": first_edge_label_pairs,
        }
        second_data = {
            "vertex_count": second_vertex_count,
            "adjacency_list": second_adj,
            "twin_map": second_twins,
            "edge_label_pairs": second_edge_label_pairs,
        }

        # Determine primal/dual orientation robustly.
        #
        # plantri double_code output order depends on -d:
        # - without -d: primal first, dual second
        # - with -d:    dual first, primal second
        #
        # Prefer strict quadrangulation consistency checks:
        #   1) dual is 4-regular
        #   2) |V_primal| = |V_dual| + 2
        # Fall back to degree-only discrimination when relation is unavailable.
        first_is_4_regular = GraphConverter.is_4_regular(first_adj)
        second_is_4_regular = GraphConverter.is_4_regular(second_adj)

        as_is_is_quadrangulation_pair = (
            second_is_4_regular
            and first_vertex_count == second_vertex_count + 2
        )
        swapped_is_quadrangulation_pair = (
            first_is_4_regular
            and second_vertex_count == first_vertex_count + 2
        )

        swap_sections = False
        if swapped_is_quadrangulation_pair and not as_is_is_quadrangulation_pair:
            swap_sections = True
        elif as_is_is_quadrangulation_pair and not swapped_is_quadrangulation_pair:
            swap_sections = False
        elif first_is_4_regular and not second_is_4_regular:
            swap_sections = True
        elif second_is_4_regular and not first_is_4_regular:
            swap_sections = False

        if swap_sections:
            primal_data = second_data
            dual_data = first_data
        else:
            primal_data = first_data
            dual_data = second_data

        return primal_data, dual_data

    @staticmethod
    def _format_edge_name_for_error(edge_name: Union[str, int]) -> str:
        """Formats an edge label for stable, readable error messages."""
        if isinstance(edge_name, str):
            return edge_name
        if 32 <= edge_name <= 126:
            return chr(edge_name)
        return f"0x{edge_name:02x}"

    @staticmethod
    def _build_adjacency_and_twins(
        edge_lists: List[Union[str, bytes]],
    ) -> Tuple[
        Dict[int, List[int]],
        Dict[Tuple[int, int], Tuple[int, int]],
        Dict[EdgeLabel, Tuple[Tuple[int, int], Tuple[int, int]]],
    ]:
        """Build adjacency, twin map, and edge-label/half-edge pairs."""
        # Collect (vertex, position) pairs where each edge name appears.
        edge_name_to_half_edges: Dict[EdgeLabel, List[Tuple[int, int]]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            if isinstance(edges_str, bytes):
                edge_iter: Iterator[EdgeLabel] = iter(edges_str)
            else:
                edge_iter = iter(edges_str)
            for pos, edge_name in enumerate(edge_iter):
                slots = edge_name_to_half_edges.get(edge_name)
                if slots is None:
                    edge_name_to_half_edges[edge_name] = [(vertex_idx, pos)]
                else:
                    slots.append((vertex_idx, pos))

        # Build adjacency list.
        adjacency: Dict[int, List[int]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            neighbors: List[int] = []
            if isinstance(edges_str, bytes):
                edge_iter = iter(edges_str)
            else:
                edge_iter = iter(edges_str)
            for edge_name in edge_iter:
                half_edges = edge_name_to_half_edges.get(edge_name)
                if half_edges is None or len(half_edges) != 2:
                    edge_name_str = QuadrangulationEnumerator._format_edge_name_for_error(
                        edge_name
                    )
                    raise ValueError(
                        f"Edge '{edge_name_str}' appears "
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
        edge_label_pairs: Dict[EdgeLabel, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
        for half_edges in edge_name_to_half_edges.values():
            if len(half_edges) == 2:
                twin_map[half_edges[0]] = half_edges[1]
                twin_map[half_edges[1]] = half_edges[0]
        for edge_name, half_edges in edge_name_to_half_edges.items():
            if len(half_edges) == 2:
                edge_label_pairs[edge_name] = (half_edges[0], half_edges[1])

        return adjacency, twin_map, edge_label_pairs
