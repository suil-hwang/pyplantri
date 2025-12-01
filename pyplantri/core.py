"""
pyplantri - Python wrapper for plantri planar graph generator

plantri는 평면 그래프(triangulation, quadrangulation 등)를 생성하는 C 프로그램입니다.
이 모듈은 plantri를 subprocess로 호출하여 Python에서 쉽게 사용할 수 있게 합니다.
"""

import subprocess
import os
import re
from pathlib import Path
from typing import Iterator, Optional, List, Literal, Union, Dict, Tuple
from collections import defaultdict

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# plantri 실행 파일 경로
_BIN_DIR = Path(__file__).parent / "bin"
_PLANTRI_EXE = _BIN_DIR / ("plantri.exe" if os.name == "nt" else "plantri")


class PlantriError(Exception):
    """Plantri 실행 관련 에러"""
    pass


class Plantri:
    """
    Plantri wrapper class

    plantri 실행 파일을 감싸서 Python에서 쉽게 사용할 수 있게 합니다.

    Examples:
        >>> plantri = Plantri()
        >>> output = plantri.run(8, options=["-q", "-a"])

        >>> for graph in plantri.generate_graphs(8, graph_type="quadrangulation"):
        ...     print(graph)
    """

    def __init__(self, executable: Optional[Path] = None):
        """
        Args:
            executable: plantri 실행 파일 경로 (기본값: 패키지 내장 바이너리)
        """
        self.executable = Path(executable) if executable else _PLANTRI_EXE
        if not self.executable.exists():
            raise FileNotFoundError(
                f"plantri 실행 파일을 찾을 수 없습니다: {self.executable}\n"
                "'pip install -e .' 를 먼저 실행하세요."
            )

    def run(
        self,
        n_vertices: int,
        options: Optional[List[str]] = None,
        output_format: Literal["planar_code", "ascii", "adjacency"] = "planar_code",
    ) -> bytes:
        """
        plantri 실행

        Args:
            n_vertices: 정점 수
            options: 추가 옵션 리스트 (예: ["-q", "-c3"])
            output_format: 출력 형식
                - "planar_code": 바이너리 planar code (기본값)
                - "ascii": 사람이 읽을 수 있는 ASCII 형식
                - "adjacency": 인접 리스트 형식

        Returns:
            plantri 출력 (바이너리)

        Raises:
            PlantriError: plantri 실행 실패 시
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise PlantriError(f"plantri 실행 실패: {error_msg}")
        except FileNotFoundError:
            raise PlantriError(f"plantri 실행 파일을 찾을 수 없습니다: {self.executable}")

    def generate_graphs(
        self,
        n_vertices: int,
        graph_type: Literal["triangulation", "quadrangulation", "cubic"] = "triangulation",
        connectivity: Optional[int] = None,
        dual: bool = False,
        minimum_degree: Optional[int] = None,
        bipartite: bool = False,
    ) -> Iterator[str]:
        """
        그래프 생성 (iterator)

        Args:
            n_vertices: 정점 수
            graph_type: 그래프 유형
                - "triangulation": 삼각분할 (기본값)
                - "quadrangulation": 사각분할
                - "cubic": 3-정규 그래프
            connectivity: 연결성 하한 (1, 2, 3, 4, 5)
            dual: True이면 dual 그래프 출력
            minimum_degree: 최소 차수
            bipartite: True이면 이분 그래프만

        Yields:
            각 그래프의 ASCII 표현
        """
        options = ["-a"]  # ASCII 출력

        # 그래프 유형
        if graph_type == "quadrangulation":
            options.append("-q")
        elif graph_type == "cubic":
            options.append("-b")

        # 연결성
        if connectivity is not None:
            options.append(f"-c{connectivity}")

        # Dual
        if dual:
            options.append("-d")

        # 최소 차수
        if minimum_degree is not None:
            options.append(f"-m{minimum_degree}")

        # 이분 그래프
        if bipartite:
            options.append("-bp")

        output = self.run(n_vertices, options, output_format="ascii")

        # ASCII 출력 파싱
        current_graph_lines = []
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
        """
        planar_code 형식으로 그래프 생성

        Args:
            n_vertices: 정점 수
            graph_type: 그래프 유형
            connectivity: 연결성
            dual: dual 그래프 여부

        Returns:
            planar_code 바이너리 데이터
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
    """
    plantri 출력을 ILP/수치해석용 포맷으로 변환하는 헬퍼

    ILP(Integer Linear Programming) 솔버에서 Warm Start나 검증에 활용할 수 있도록
    plantri ASCII 출력을 다양한 데이터 구조로 변환합니다.

    Examples:
        >>> converter = GraphConverter()
        >>> edge_map = converter.parse_ascii_to_edge_map("5 b,acc,bbded,cc,c")
        >>> print(edge_map)
        {(0, 1): 1, (0, 2): 2, ...}

        >>> adj_matrix = converter.to_adjacency_matrix(edge_map, 5)
    """

    @staticmethod
    def parse_ascii_to_edge_map(
        ascii_line: str,
        one_based: bool = False
    ) -> Dict[Tuple[int, int], int]:
        """
        ASCII 출력을 Edge Multiplicity Map으로 변환

        plantri의 ASCII 출력을 (u, v) -> count 형태의 딕셔너리로 변환합니다.
        Multigraph에서 두 정점 사이에 여러 간선이 있을 경우 count > 1입니다.

        Args:
            ascii_line: plantri ASCII 출력 문자열 (예: "5 b,acc,bbded,cc,c")
            one_based: True면 정점 번호가 1부터 시작, False면 0부터 시작 (ILP는 보통 0)

        Returns:
            dict: {(u, v): count} 형태. u < v 보장 (canonical edge).

        Examples:
            >>> GraphConverter.parse_ascii_to_edge_map("4 bcd,acd,abd,abc")
            {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 3): 1}
        """
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        n = int(parts[0])
        if len(parts) < 2:
            return {}

        adj_str = parts[1]
        vertex_lists = adj_str.split(",")

        edge_map: Dict[Tuple[int, int], int] = defaultdict(int)
        offset = 0 if one_based else 1  # plantri 'a'는 1번째 정점을 의미

        for u_idx, neighbors_str in enumerate(vertex_lists):
            u = u_idx + (1 if one_based else 0)
            for char in neighbors_str:
                v_raw = ord(char) - ord('a') + 1
                v = v_raw - offset

                # Canonical edge key (u < v) - 한 방향만 카운트
                # plantri는 양방향 모두 나열하므로 u < v인 경우만 카운트
                if u < v:
                    edge_map[(u, v)] += 1
                # u > v인 경우는 이미 반대 방향에서 카운트됨, 스킵

        return dict(edge_map)

    @staticmethod
    def parse_ascii_to_adjacency_list(
        ascii_line: str,
        one_based: bool = False
    ) -> Dict[int, List[int]]:
        """
        ASCII 출력을 인접 리스트로 변환 (중복 포함)

        Args:
            ascii_line: plantri ASCII 출력 문자열
            one_based: True면 정점 번호가 1부터 시작

        Returns:
            dict: {vertex: [neighbors]} 형태. 중복 간선은 리스트에 여러 번 등장.
        """
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        if len(parts) < 2:
            return {}

        adj_str = parts[1]
        vertex_lists = adj_str.split(",")

        adjacency: Dict[int, List[int]] = {}
        offset = 0 if one_based else 1

        for u_idx, neighbors_str in enumerate(vertex_lists):
            u = u_idx + (1 if one_based else 0)
            neighbors = [ord(c) - ord('a') + 1 - offset for c in neighbors_str]
            adjacency[u] = neighbors

        return adjacency

    @staticmethod
    def to_adjacency_matrix(
        edge_map: Dict[Tuple[int, int], int],
        n_vertices: int
    ):
        """
        Edge map을 인접 행렬(numpy)로 변환

        Args:
            edge_map: {(u, v): count} 형태의 edge map
            n_vertices: 정점 수

        Returns:
            numpy.ndarray: n_vertices x n_vertices 인접 행렬

        Raises:
            ImportError: numpy가 설치되지 않은 경우
        """
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy가 필요합니다. 'pip install numpy'로 설치하세요."
            )

        adj = np.zeros((n_vertices, n_vertices), dtype=int)
        for (u, v), count in edge_map.items():
            adj[u, v] = count
            adj[v, u] = count
        return adj

    @staticmethod
    def get_graph_stats(edge_map: Dict[Tuple[int, int], int], n_vertices: int) -> dict:
        """
        그래프 통계 정보 반환

        Args:
            edge_map: {(u, v): count} 형태의 edge map
            n_vertices: 정점 수

        Returns:
            dict: 그래프 통계 정보
                - n_vertices: 정점 수
                - n_edges: 총 간선 수 (중복 포함)
                - single_edges: 단일 간선 수
                - double_edges: 이중 간선 수 (digon)
                - degrees: 각 정점의 차수 리스트
                - is_regular: 정규 그래프 여부
                - regularity: 정규 그래프일 경우 차수
        """
        degrees = [0] * n_vertices
        single_edges = 0
        double_edges = 0
        total_edges = 0

        for (u, v), count in edge_map.items():
            degrees[u] += count
            degrees[v] += count
            total_edges += count
            if count == 1:
                single_edges += 1
            elif count == 2:
                double_edges += 1

        is_regular = len(set(degrees)) == 1 if degrees else False
        regularity = degrees[0] if is_regular and degrees else None

        return {
            "n_vertices": n_vertices,
            "n_edges": total_edges,
            "single_edges": single_edges,
            "double_edges": double_edges,
            "degrees": degrees,
            "is_regular": is_regular,
            "regularity": regularity,
        }

    @staticmethod
    def to_gurobi_start_dict(
        edge_map: Dict[Tuple[int, int], int],
        var_prefix: str = "x"
    ) -> Dict[str, int]:
        """
        Gurobi MIP Start용 딕셔너리 생성

        Args:
            edge_map: {(u, v): count} 형태의 edge map
            var_prefix: 변수 이름 prefix (기본값: "x")

        Returns:
            dict: {"x[0,1]": 1, "x[0,2]": 2, ...} 형태의 딕셔너리

        Examples:
            >>> edge_map = {(0, 1): 1, (0, 2): 2}
            >>> GraphConverter.to_gurobi_start_dict(edge_map)
            {'x[0,1]': 1, 'x[0,2]': 2}
        """
        return {
            f"{var_prefix}[{u},{v}]": count
            for (u, v), count in edge_map.items()
        }


def generate_triangulations(
    n_vertices: int,
    connectivity: int = 3,
    dual: bool = False,
) -> Iterator[str]:
    """
    Triangulation(삼각분할) 생성

    삼각분할은 모든 면이 삼각형인 평면 그래프입니다.

    Args:
        n_vertices: 정점 수
        connectivity: 연결성 (기본값 3)
        dual: True이면 cubic dual 그래프 출력

    Yields:
        각 triangulation의 ASCII 표현

    Examples:
        >>> for graph in generate_triangulations(6):
        ...     print(graph)
    """
    plantri = Plantri()
    yield from plantri.generate_graphs(
        n_vertices,
        graph_type="triangulation",
        connectivity=connectivity,
        dual=dual,
    )


def generate_quadrangulations(
    n_vertices: int,
    connectivity: int = 3,
    dual: bool = False,
) -> Iterator[str]:
    """
    Quadrangulation(사각분할) 생성

    사각분할은 모든 면이 사각형인 평면 그래프입니다.

    Args:
        n_vertices: 정점 수 (짝수여야 함)
        connectivity: 연결성 (기본값 3)
        dual: True이면 4-regular dual 그래프 출력

    Yields:
        각 quadrangulation의 ASCII 표현

    Examples:
        >>> for graph in generate_quadrangulations(8):
        ...     print(graph)

        >>> # 4-regular dual graphs
        >>> for graph in generate_quadrangulations(8, dual=True):
        ...     print(graph)
    """
    plantri = Plantri()
    yield from plantri.generate_graphs(
        n_vertices,
        graph_type="quadrangulation",
        connectivity=connectivity,
        dual=dual,
    )


def count_graphs(
    n_vertices: int,
    graph_type: Literal["triangulation", "quadrangulation"] = "triangulation",
    connectivity: int = 3,
) -> int:
    """
    그래프 개수 세기 (출력 없이)

    Args:
        n_vertices: 정점 수
        graph_type: 그래프 유형
        connectivity: 연결성

    Returns:
        생성된 그래프의 개수

    Examples:
        >>> count_graphs(8, "triangulation", 3)
        14
        >>> count_graphs(8, "quadrangulation", 3)
        1
    """
    plantri = Plantri()
    options = ["-u"]  # stdout 출력 안 함 (개수만 stderr로)

    if graph_type == "quadrangulation":
        options.append("-q")

    options.append(f"-c{connectivity}")

    try:
        result = subprocess.run(
            [str(plantri.executable)] + options + [str(n_vertices)],
            capture_output=True,
            text=True,
        )
        # stderr에서 개수 파싱
        # 예: "1 graphs written to stdout"
        for line in result.stderr.split("\n"):
            match = re.search(r"(\d+)\s+graph", line.lower())
            if match:
                return int(match.group(1))
    except Exception:
        pass
    return 0


def list_graphs(
    n_vertices: int,
    graph_type: Literal["triangulation", "quadrangulation"] = "triangulation",
    connectivity: int = 3,
    dual: bool = False,
) -> List[str]:
    """
    그래프 목록 반환 (리스트)

    iterator 대신 전체 리스트가 필요할 때 사용합니다.

    Args:
        n_vertices: 정점 수
        graph_type: 그래프 유형
        connectivity: 연결성
        dual: dual 그래프 여부

    Returns:
        그래프 ASCII 표현의 리스트

    Examples:
        >>> graphs = list_graphs(6, "triangulation")
        >>> len(graphs)
        1
    """
    plantri = Plantri()
    return list(plantri.generate_graphs(
        n_vertices,
        graph_type=graph_type,
        connectivity=connectivity,
        dual=dual,
    ))
