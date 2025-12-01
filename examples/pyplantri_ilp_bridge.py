"""
pyplantri_ilp_bridge.py
Plantri로 생성한 그래프를 Gurobi 초기 해(MIP Start) 형식으로 변환

Simple Quadrangulation (SQS)의 Dual Graph인 4-regular planar multigraph를
ILP 솔버의 Warm Start로 활용할 수 있는 형태로 출력합니다.

논문 참조: Samuel Peltier et al. (2021)
Dual Q*의 성질:
1. Planar graph (평면 그래프)
2. Multigraph (double edge 허용)
3. Loop 없음 (Q에 degree 1 vertex가 없으므로)
4. 4-regular (모든 vertex가 정확히 degree 4)

plantri 옵션:
- `-q -c2 -m2`: Simple quadrangulation, 2-connected, min degree 2
  → Dual: 4-edge-connected quartic multigraph (double edge 허용)

Usage:
    python pyplantri_ilp_bridge.py 6          # Q*: 6정점 (Q: 8정점)
    python pyplantri_ilp_bridge.py 6 --show-matrix
    python pyplantri_ilp_bridge.py 6 --max-graphs 5
    python pyplantri_ilp_bridge.py 6 --export output.json
"""
import argparse
from pyplantri import Plantri
from pyplantri.core import GraphConverter


def generate_sqs_dual_graphs(
    n_dual_vertices: int,
    show_matrix: bool = False,
    max_graphs: int = None
):
    """
    SQS Dual Graph (4-regular planar multigraph, loop-free) 생성

    plantri 옵션: -q -c2 -m2
    - -q: Simple quadrangulation (primal에 multi-edge 없음)
    - -c2: 2-connected
    - -m2: minimum degree 2
    → Dual: 4-edge-connected quartic multigraph (double edge 허용, loop 없음)

    Args:
        n_dual_vertices: Q*의 정점 수
        show_matrix: True면 인접 행렬도 출력
        max_graphs: 출력할 최대 그래프 수 (None이면 전체)
    """
    plantri = Plantri()

    # -q: Simple Quadrangulation (loop-free dual 보장)
    # -c2: 2-connected
    # -m2: minimum degree 2
    # -d: Dual (4-regular graph 생성)
    # -a: ASCII 출력
    options = ["-q", "-c2", "-m2", "-d", "-a"]

    # Q*의 정점 수 = Q의 정점 수 - 2 (Euler 공식)
    # 따라서 Q의 정점 수 = n_dual_vertices + 2
    plantri_input_n = n_dual_vertices + 2

    try:
        output = plantri.run(plantri_input_n, options)
    except Exception as e:
        print(f"Plantri 실행 오류: {e}")
        return None

    print(f"{'='*60}")
    print(f"SQS Dual Graph Generation (Loop-free 4-regular Multigraph)")
    print(f"{'='*60}")
    print(f"논문: Samuel Peltier et al. (2021)")
    print(f"plantri 옵션: -q -c2 -m2")
    print(f"")
    print(f"Q (Primal):  {plantri_input_n} vertices (Simple Quadrangulation)")
    print(f"Q* (Dual):   {n_dual_vertices} vertices (4-regular planar multigraph)")

    count = 0
    graphs_data = []

    for line in output.decode(errors="replace").split("\n"):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        # 1. Edge Map으로 파싱 (0-based index)
        edge_map = GraphConverter.parse_ascii_to_edge_map(line, one_based=False)
        if not edge_map:
            continue

        # 2. 통계 정보 및 검증
        stats = GraphConverter.get_graph_stats(edge_map, n_dual_vertices)

        # 4-regular 검증
        if not stats["is_regular"] or stats["regularity"] != 4:
            continue

        # Loop 검증 (Simple Quadrangulation의 dual이므로 loop가 없어야 함)
        adj_list = GraphConverter.parse_ascii_to_adjacency_list(line, one_based=False)
        has_loop = any(v in neighbors for v, neighbors in adj_list.items())
        if has_loop:
            print(f"[WARNING] Unexpected loop detected: {line}")
            continue

        count += 1
        graphs_data.append((edge_map, stats, line))

        if max_graphs and count >= max_graphs:
            break

    print(f"\n총 {len(graphs_data)}개의 loop-free 4-regular planar multigraph 생성됨")

    for idx, (edge_map, stats, raw_line) in enumerate(graphs_data, 1):
        print(f"\n{'#'*60}")
        print(f"# Solution Candidate #{idx}")
        print(f"{'#'*60}")
        print(f"Raw: {raw_line}")

        # 3. 그래프 통계
        print(f"\n[Graph Statistics]")
        print(f"  Vertices: {stats['n_vertices']}")
        print(f"  Total Edges: {stats['n_edges']}")
        print(f"  Single Edges: {stats['single_edges']}")
        print(f"  Double Edges (Digons): {stats['double_edges']}")
        print(f"  Regularity: {stats['regularity']}-regular")
        print(f"  Loop-free: Yes")

        # 4. Gurobi 스타일 변수 출력
        print(f"\n[Gurobi Variable Hints]")
        gurobi_dict = GraphConverter.to_gurobi_start_dict(edge_map)
        for var_name, value in sorted(gurobi_dict.items()):
            print(f"  {var_name}.Start = {value}")

        # 5. 인접 행렬 (옵션)
        if show_matrix:
            try:
                adj_matrix = GraphConverter.to_adjacency_matrix(edge_map, n_dual_vertices)
                print(f"\n[Adjacency Matrix]")
                print(adj_matrix)
            except ImportError as e:
                print(f"\n[Adjacency Matrix] {e}")

    return graphs_data


def export_to_json(graphs_data: list, filename: str):
    """
    그래프 데이터를 JSON 파일로 내보내기

    Args:
        graphs_data: generate_sqs_dual_graphs()의 반환값
        filename: 출력 파일명
    """
    import json

    export_data = []
    for edge_map, stats, raw_line in graphs_data:
        # JSON에서 tuple key를 지원하지 않으므로 리스트로 변환
        edge_list = [
            {"edge": [u, v], "multiplicity": count}
            for (u, v), count in edge_map.items()
        ]
        export_data.append({
            "raw": raw_line,
            "stats": stats,
            "edges": edge_list,
            "gurobi_vars": GraphConverter.to_gurobi_start_dict(edge_map),
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(export_data)} graphs to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="SQS Dual Graph를 ILP Warm Start 형식으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            논문 기반 성질 (Samuel Peltier et al., 2021):
            Q* (Dual)는 다음 성질을 만족:
            1. Planar graph
            2. Multigraph (double edge 허용)
            3. Loop 없음
            4. 4-regular

            plantri 옵션: -q -c2 -m2
            → Dual: 4-edge-connected quartic multigraph

            정점 수 관계:
            Q (Primal): n + 2 vertices
            Q* (Dual):  n vertices

            예제:
            python pyplantri_ilp_bridge.py 6              # Q*: 6정점, Q: 8정점
            python pyplantri_ilp_bridge.py 6 --show-matrix
            python pyplantri_ilp_bridge.py 8 --max-graphs 5
            python pyplantri_ilp_bridge.py 6 --export output.json
            """,
    )
    parser.add_argument(
        "n",
        type=int,
        help="Q* (Dual)의 정점 수 (최소 3)"
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="인접 행렬 출력 (numpy 필요)"
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="출력할 최대 그래프 수"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="FILE",
        help="JSON 파일로 내보내기"
    )

    args = parser.parse_args()

    # Q*의 정점 수 검증: 최소 3 이상
    if args.n < 3:
        parser.error("n은 최소 3 이상이어야 합니다.")

    graphs_data = generate_sqs_dual_graphs(
        args.n,
        show_matrix=args.show_matrix,
        max_graphs=args.max_graphs
    )

    if args.export and graphs_data:
        export_to_json(graphs_data, args.export)

    print("\n완료!")


if __name__ == "__main__":
    main()

# python examples/pyplantri_ilp_bridge.py 6          # Q*: 6정점, Q: 8정점
# python examples/pyplantri_ilp_bridge.py 6 --show-matrix
# python examples/pyplantri_ilp_bridge.py 6 --max-graphs 5
# python examples/pyplantri_ilp_bridge.py 6 --export output.json