# src/pyplantri/plantri.py
import os
import re
import subprocess
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Dict, Iterator, List, Literal, Optional, Set, Tuple, Union, cast

from .converter import GraphConverter
from .types import EdgeLabel, EdgeLabelPairs, HalfEdge


@dataclass(frozen=True, slots=True)
class ParsedGraphSection:
    """One graph section parsed from plantri -T double_code output.

    All vertex indices are 1-based, matching plantri's native output convention.
    Use GraphConverter.to_zero_based_embedding() to convert to 0-based.
    """

    vertex_count: int
    adjacency_list: Dict[int, List[int]]
    twin_map: Dict[HalfEdge, HalfEdge]
    edge_label_pairs: EdgeLabelPairs


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

    path_exe = which(exe_name)
    if path_exe is not None:
        return Path(path_exe)

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
                "Run 'pip install -e .' first or ensure 'plantri' is available on PATH."
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


class QuadrangulationEnumerator:
    """Enumerates 4-regular plane multigraphs (duals of simple quadrangulations)."""

    OPTIONS = ["-q", "-c2", "-m2", "-T"]

    def __init__(self) -> None:
        """Initializes the SQS enumerator with a Plantri instance."""
        self._plantri = Plantri()

    def generate_pairs(
        self,
        dual_vertex_count: int
    ) -> Iterator[Tuple[ParsedGraphSection, ParsedGraphSection]]:
        """Generates primal (Q) and dual (Q*) graph pairs."""
        for line in self.iter_raw(dual_vertex_count):
            primal, dual = self.parse_double_code(line)
            if primal.adjacency_list and dual.adjacency_list:
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
    def parse_double_code(
        double_code_line: Union[str, bytes],
    ) -> Tuple[ParsedGraphSection, ParsedGraphSection]:
        """Parses plantri double_code output to adjacency lists with twin maps."""
        parts = double_code_line.split()
        if len(parts) < 2:
            empty = ParsedGraphSection(
                vertex_count=0,
                adjacency_list={},
                twin_map={},
                edge_label_pairs={},
            )
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
            empty = ParsedGraphSection(
                vertex_count=0,
                adjacency_list={},
                twin_map={},
                edge_label_pairs={},
            )
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

        first_data = ParsedGraphSection(
            vertex_count=first_vertex_count,
            adjacency_list=first_adj,
            twin_map=first_twins,
            edge_label_pairs=first_edge_label_pairs,
        )
        second_data = ParsedGraphSection(
            vertex_count=second_vertex_count,
            adjacency_list=second_adj,
            twin_map=second_twins,
            edge_label_pairs=second_edge_label_pairs,
        )

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
