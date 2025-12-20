# src/pyplantri/core.py
"""Core module for pyplantri - Python wrapper for plantri.

This module provides classes for generating and converting planar graphs
using the plantri executable.
"""

import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix


# =============================================================================
# Module Constants (Simple)
# =============================================================================

_ORD_A = ord('a')  # 97, 문자열 파싱 최적화용 상수


# =============================================================================
# Private Helpers
# =============================================================================

def _find_plantri_exe() -> Path:
    """Finds the plantri executable path.

    Searches in the following order:
        1. Package bin folder (installed package)
        2. scikit-build-core build folder (editable install)
        3. Default path (for error messages)

    Returns:
        Path to the plantri executable.
    """
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"
    
    # 1. 패키지 내 bin 폴더 (설치된 경우)
    pkg_bin = Path(__file__).parent / "bin" / exe_name
    if pkg_bin.exists():
        return pkg_bin
    
    # 2. scikit-build-core 빌드 폴더 (editable 설치, 개발 모드)
    # src/pyplantri/core.py -> src -> project_root
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    if build_dir.exists():
        for tag_dir in build_dir.iterdir():
            if tag_dir.is_dir():
                # Release 폴더 (Visual Studio 빌드)
                release_exe = tag_dir / "Release" / exe_name
                if release_exe.exists():
                    return release_exe
                # MinGW/Unix 빌드
                direct_exe = tag_dir / exe_name
                if direct_exe.exists():
                    return direct_exe
    
    # 3. 기본 경로 반환 (에러 메시지용)
    return Path(__file__).parent / "bin" / exe_name



# =============================================================================
# Module Constants (Runtime)
# =============================================================================

_PLANTRI_EXE = _find_plantri_exe()


# =============================================================================
# Exceptions
# =============================================================================

class PlantriError(Exception):
    """Exception raised when plantri execution fails.

    Attributes:
        message: Explanation of the error.
    """


# =============================================================================
# Core Classes
# =============================================================================

class Plantri:
    """Wrapper class for the plantri executable.

    Provides a Python interface to run plantri and generate various
    types of planar graphs.

    Attributes:
        executable: Path to the plantri executable.

    Example:
        >>> plantri = Plantri()
        >>> output = plantri.run(8, options=["-q", "-a"])
        >>> for graph in plantri.generate_graphs(8, graph_type="quadrangulation"):
        ...     print(graph)
    """

    def __init__(self, executable: Optional[Path] = None):
        """Initializes Plantri with the executable path.

        Args:
            executable: Path to plantri executable. If None, uses the
                bundled binary from the package.

        Raises:
            FileNotFoundError: If the plantri executable is not found.
        """
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
        """Runs plantri with the given parameters.

        Args:
            n_vertices: Number of vertices in the graph.
            options: Additional command-line options (e.g., ["-q", "-c3"]).
            output_format: Output format type.
                - "planar_code": Binary planar code (default).
                - "ascii": Human-readable ASCII format.
                - "adjacency": Adjacency list format.

        Returns:
            Raw binary output from plantri.

        Raises:
            PlantriError: If plantri execution fails.
        """
        cmd = [str(self.executable)]

        if options:
            cmd.extend(options)

        # 출력 형식 설정
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
    ) -> int:
        """Counts the number of graphs without generating output.

        Args:
            n_vertices: Number of vertices in the graph.
            graph_type: Type of graph to count.
            connectivity: Minimum connectivity requirement.

        Returns:
            Number of generated graphs.

        Example:
            >>> plantri = Plantri()
            >>> plantri.count(8, "triangulation", 3)
            14
        """
        options = ["-u"]  # stdout 출력 안 함 (개수만 stderr로)

        if graph_type == "quadrangulation":
            options.append("-q")

        options.append(f"-c{connectivity}")

        try:
            result = subprocess.run(
                [str(self.executable)] + options + [str(n_vertices)],
                capture_output=True,
                text=True,
            )
            # stderr에서 개수 파싱 (예: "1 graphs written to stdout")
            for line in result.stderr.split("\n"):
                match = re.search(r"(\d+)\s+graph", line.lower())
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return 0

    def generate_graphs(
        self,
        n_vertices: int,
        graph_type: Literal["triangulation", "quadrangulation", "cubic"] = "triangulation",
        connectivity: Optional[int] = None,
        dual: bool = False,
        minimum_degree: Optional[int] = None,
        bipartite: bool = False,
    ) -> Iterator[str]:
        """Generates planar graphs as an iterator.

        Args:
            n_vertices: Number of vertices in the graph.
            graph_type: Type of graph to generate.
                - "triangulation": Triangulated graph (default).
                - "quadrangulation": Quadrangulated graph.
                - "cubic": 3-regular graph.
            connectivity: Minimum connectivity (1, 2, 3, 4, or 5).
            dual: If True, outputs dual graph.
            minimum_degree: Minimum vertex degree.
            bipartite: If True, only generates bipartite graphs.

        Yields:
            ASCII representation of each graph.
        """
        options = ["-a"]  # ASCII 출력

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

        # ASCII 출력 파싱
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
        """Generates graphs in planar_code binary format.

        Args:
            n_vertices: Number of vertices in the graph.
            graph_type: Type of graph to generate.
            connectivity: Minimum connectivity requirement.
            dual: If True, generates dual graph.

        Returns:
            Binary planar_code data.
        """
        options = []

        if graph_type == "quadrangulation":
            options.append("-q")
        if connectivity is not None:
            options.append(f"-c{connectivity}")
        if dual:
            options.append("-d")

        return self.run(n_vertices, options, output_format="planar_code")


class GraphConverter:
    """Utility class for converting plantri output to various formats.

    Provides static methods for converting graph data to formats suitable
    for ILP solvers, NumPy arrays, and SciPy sparse matrices.

    Example:
        >>> edge_map = GraphConverter.parse_ascii_to_edge_map("5 b,acc,bbded,cc,c")
        >>> adj_matrix = GraphConverter.to_adjacency_matrix(edge_map, 5)
    """

    @staticmethod
    def parse_ascii_to_edge_map(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[Tuple[int, int], int]:
        """Converts ASCII output to edge multiplicity map.

        Parses plantri's ASCII output to a {(u, v): multiplicity} dictionary.
        For multigraphs, multiplicity > 1 when multiple edges exist between vertices.

        Args:
            ascii_line: Plantri ASCII output string (e.g., "5 b,acc,bbded,cc,c").
            is_one_based: If True, vertex numbering starts from 1.

        Returns:
            Dictionary mapping canonical edges (u < v) to their multiplicities.

        Example:
            >>> GraphConverter.parse_ascii_to_edge_map("4 bcd,acd,abd,abc")
            {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
        """
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
                # Canonical edge key (u < v) - 한 방향만 카운트
                if source_vertex < target_vertex:
                    edge_multiplicity[(source_vertex, target_vertex)] += 1

        return dict(edge_multiplicity)

    @staticmethod
    def parse_ascii_to_adjacency_list(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[int, List[int]]:
        """Converts ASCII output to adjacency list.

        Args:
            ascii_line: Plantri ASCII output string.
            is_one_based: If True, vertex numbering starts from 1.

        Returns:
            Dictionary mapping each vertex to its list of neighbors.
            Multi-edges appear multiple times in the neighbor list.
        """
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
        """Converts adjacency list to edge multiplicity map.

        Args:
            adjacency_list: Dictionary mapping vertices to neighbor lists.
            is_one_based: If True, vertex numbering starts from 1.

        Returns:
            Dictionary mapping canonical edges (u < v) to multiplicities.
        """
        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        offset = 1 if is_one_based else 0

        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                u = min(vertex - offset, neighbor - offset)
                v = max(vertex - offset, neighbor - offset)
                edge_multiplicity[(u, v)] += 1

        # 양쪽에서 카운트되므로 2로 나눔
        return {k: v // 2 for k, v in edge_multiplicity.items()}

    @staticmethod
    def to_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ):
        """Converts edge multiplicity map to dense adjacency matrix.

        Args:
            edge_multiplicity: Dictionary mapping edges to multiplicities.
            vertex_count: Number of vertices in the graph.

        Returns:
            numpy.ndarray: Symmetric adjacency matrix of shape (V, V).
        """
        if not edge_multiplicity:
            return np.zeros((vertex_count, vertex_count), dtype=np.int32)

        # 벡터화된 인덱싱으로 성능 개선
        edges = list(edge_multiplicity.keys())
        sources = np.array([e[0] for e in edges], dtype=np.intp)
        targets = np.array([e[1] for e in edges], dtype=np.intp)
        multiplicities = np.array(list(edge_multiplicity.values()), dtype=np.int32)

        adjacency_matrix = np.zeros((vertex_count, vertex_count), dtype=np.int32)
        # 대칭 행렬 한 번에 할당 (fancy indexing)
        adjacency_matrix[sources, targets] = multiplicities
        adjacency_matrix[targets, sources] = multiplicities
        return adjacency_matrix

    @staticmethod
    def to_sparse_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> csr_matrix:
        """Converts edge multiplicity map to CSR sparse matrix.

        Planar graphs satisfy E <= 3V - 6, making sparse matrices
        memory-efficient. Compatible with scipy.sparse.csgraph algorithms.

        Args:
            edge_multiplicity: Dictionary mapping edges to multiplicities.
            vertex_count: Number of vertices in the graph.

        Returns:
            Symmetric CSR sparse adjacency matrix.

        Example:
            >>> edge_map = {(0, 1): 2, (1, 2): 1}
            >>> sparse_mat = GraphConverter.to_sparse_adjacency_matrix(edge_map, 3)
            >>> sparse_mat.toarray()
            array([[0, 2, 0],
                   [2, 0, 1],
                   [0, 1, 0]], dtype=int32)
        """
        if not edge_multiplicity:
            return csr_matrix((vertex_count, vertex_count), dtype=np.int32)

        # COO 형식으로 데이터 준비 (대칭 행렬이므로 양방향 추가)
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
        """Creates a Gurobi MIP start dictionary.

        Args:
            edge_multiplicity: Dictionary mapping edges to multiplicities.
            var_prefix: Variable name prefix (default: "x").

        Returns:
            Dictionary mapping variable names to values.

        Example:
            >>> edge_multiplicity = {(0, 1): 1, (0, 2): 2}
            >>> GraphConverter.to_gurobi_start_dict(edge_multiplicity)
            {'x[0,1]': 1, 'x[0,2]': 2}
        """
        return {
            f"{var_prefix}[{source},{target}]": multiplicity
            for (source, target), multiplicity in edge_multiplicity.items()
        }

    @staticmethod
    def get_graph_stats(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> dict:
        """Computes graph statistics from edge multiplicity map.

        Args:
            edge_multiplicity: Dictionary mapping edges to multiplicities.
            vertex_count: Number of vertices in the graph.

        Returns:
            Dictionary containing:
                - vertex_count: Number of vertices.
                - edge_count: Total edges (including multi-edges).
                - single_edge_count: Number of single edges.
                - double_edge_count: Number of double edges (digons).
                - degrees: List of vertex degrees.
                - is_regular: Whether graph is regular.
                - regularity: Degree if regular, else None.
        """
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


class SQSEnumerator:
    """Enumerator for Simple Quadrangulation on Sphere (SQS).

    Based on: Samuel Peltier et al. (2021).
    Uses plantri options: -q -c2 -m2 -T (double_code).

    Properties:
        Q (Primal): Simple quadrangulation (no loops/multi-edges).
        Q* (Dual): 4-regular planar multigraph (no loops, double edges allowed).

    Adjacency list neighbor order is clockwise (CW).

    Attributes:
        OPTIONS: Plantri command-line options for SQS enumeration.

    Example:
        >>> sqs = SQSEnumerator()
        >>> for primal, dual in sqs.generate_pairs(4):
        ...     print(f"Q: {primal['vertex_count']} vertices")
        ...     print(f"Q*: {dual['vertex_count']} vertices")
    """

    # plantri 옵션: Simple Quadrangulation, 2-connected, min degree 2, double_code
    OPTIONS = ["-q", "-c2", "-m2", "-T"]

    def __init__(self):
        self._plantri = Plantri()

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def generate_pairs(
        self,
        dual_vertex_count: int
    ) -> Iterator[Tuple[Dict, Dict]]:
        """Generates primal (Q) and dual (Q*) graph pairs.

        Args:
            dual_vertex_count: Number of vertices in Q* (minimum 3).

        Yields:
            Tuple of (primal_data, dual_data) where each contains:
                - vertex_count: Number of vertices.
                - adjacency_list: {vertex: [neighbors]} in CW order.
        """
        for line in self.iter_raw(dual_vertex_count):
            primal, dual = self.parse_double_code(line)
            if primal["adjacency_list"] and dual["adjacency_list"]:
                yield primal, dual

    def count(self, dual_vertex_count: int) -> int:
        """Counts the number of non-isomorphic SQS structures.

        Args:
            dual_vertex_count: Number of vertices in Q* (dual).

        Returns:
            Number of non-isomorphic simple quadrangulations.
        """
        return sum(1 for _ in self.iter_raw(dual_vertex_count))

    def iter_raw(self, dual_vertex_count: int) -> Iterator[str]:
        """Generates raw double_code output lines.

        Memory-efficient generator for raw plantri output.

        Args:
            dual_vertex_count: Number of vertices in Q* (minimum 3).

        Yields:
            Raw double_code output line for each graph.
        """
        primal_vertex_count = dual_vertex_count + 2  # Euler 공식
        output = self._plantri.run(primal_vertex_count, self.OPTIONS)

        for line in output.decode(errors="replace").split("\n"):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                yield stripped

    # -------------------------------------------------------------------------
    # Static Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def parse_double_code(double_code_line: str) -> Tuple[Dict, Dict]:
        """Parses plantri double_code output to adjacency lists.

        Format: "6 ABCD BAEF CAGB DHEC EFDG GHFA 4 AEB BFCA CGDE DHGF"
            - Edge names: A, B, C, ... (uppercase letters).
            - First number: primal vertex count.
            - Following: edges around each vertex (space-separated, CW order).
            - Second number: dual vertex count.
            - Following: edges around each dual vertex (CW order).

        Args:
            double_code_line: Plantri double_code format string.

        Returns:
            Tuple of (primal_data, dual_data) where each contains:
                - vertex_count: Number of vertices.
                - adjacency_list: {vertex: [neighbors]} in CW order.
        """
        parts = double_code_line.strip().split()
        if len(parts) < 2:
            empty: Dict = {"vertex_count": 0, "adjacency_list": {}}
            return empty, empty

        # 첫 번째 그래프 (primal) 파싱
        primal_vertex_count = int(parts[0])

        # primal 정점들의 에지 리스트 수집
        idx = 1
        primal_edge_lists = []
        while idx < len(parts) and not parts[idx].isdigit():
            primal_edge_lists.append(parts[idx])
            idx += 1

        # 두 번째 그래프 (dual) 파싱
        if idx >= len(parts):
            empty = {"vertex_count": 0, "adjacency_list": {}}
            return empty, empty

        dual_vertex_count = int(parts[idx])
        idx += 1

        dual_edge_lists = []
        while idx < len(parts):
            dual_edge_lists.append(parts[idx])
            idx += 1

        # 에지 이름으로 연결된 정점 찾기
        primal_adj = SQSEnumerator._build_adjacency_from_edge_lists(primal_edge_lists)
        dual_adj = SQSEnumerator._build_adjacency_from_edge_lists(dual_edge_lists)

        primal_data = {
            "vertex_count": primal_vertex_count,
            "adjacency_list": primal_adj
        }
        dual_data = {
            "vertex_count": dual_vertex_count,
            "adjacency_list": dual_adj
        }

        return primal_data, dual_data

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_adjacency_from_edge_lists(
        edge_lists: List[str]
    ) -> Dict[int, List[int]]:
        """
        에지 리스트에서 인접 리스트 구성 (CW 순서 유지)

        edge_lists: ["ABCD", "BAEF", "CAGB", ...]
        각 문자열은 해당 정점(1-indexed) 주변의 에지들 (CW 순서)
        같은 에지 이름이 두 정점에서 나타나면 그 두 정점이 연결됨
        """
        # 각 에지가 어떤 정점들에서 나타나는지 추적
        edge_to_vertices: Dict[str, List[int]] = defaultdict(list)

        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            for edge_name in edges_str:
                edge_to_vertices[edge_name].append(vertex_idx)

        # 루프 에지 사전 계산 (Counter로 O(n) 처리)
        loop_edges: set[str] = {
            edge_name
            for edge_name, vertices in edge_to_vertices.items()
            if len(vertices) == 2 and vertices[0] == vertices[1]
        }

        # 인접 리스트 구성
        adjacency_list: Dict[int, List[int]] = defaultdict(list)

        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            for edge_name in edges_str:
                vertices = edge_to_vertices[edge_name]
                if edge_name in loop_edges:
                    # 루프 에지: 자기 자신 추가
                    adjacency_list[vertex_idx].append(vertex_idx)
                else:
                    # 일반 에지: 다른 정점만 추가
                    for other_vertex in vertices:
                        if other_vertex != vertex_idx:
                            adjacency_list[vertex_idx].append(other_vertex)

        return dict(adjacency_list)
