# src/pyplantri/ilp_bridge.py
import argparse
import json
from typing import List, Dict, Tuple, Optional
from .core import SQSEnumerator, GraphConverter


def enumerate_sqs_graphs(
    dual_vertex_count: int,
    should_show_matrix: bool = False,
    max_graph_count: Optional[int] = None
) -> List[Tuple]:
    """
    SQS (Simple Quadrangulation on Sphere) 및 Dual Graph 열거

    plantri 옵션: -q -c2 -m2 -T
    - -q: Simple quadrangulation (primal에 multi-edge 없음)
    - -c2: 2-connected
    - -m2: minimum degree 2
    - -T: double_code 출력 (CW 순서)
    → Dual: 4-edge-connected quartic multigraph (double edge 허용, loop 없음)
    
    출력되는 인접 리스트의 이웃 순서는 CW (시계방향) 순서입니다.

    Args:
        dual_vertex_count: Q*의 정점 수
        should_show_matrix: True면 인접 행렬도 출력
        max_graph_count: 출력할 최대 그래프 수 (None이면 전체)
        
    Returns:
        [(edge_multiplicity, stats, primal_data, dual_data), ...] 리스트
    """
    primal_vertex_count = dual_vertex_count + 2  # Euler 공식

    print(f"{'='*60}")
    print(f"SQS Enumeration (Primal & Dual Graphs)")
    print(f"{'='*60}")
    print(f"논문: Samuel Peltier et al. (2021)")
    print(f"plantri 옵션: -q -c2 -m2 -T (double_code)")
    print(f"")
    print(f"Q (Primal):  {primal_vertex_count} vertices (Simple Quadrangulation)")
    print(f"Q* (Dual):   {dual_vertex_count} vertices (4-regular planar multigraph)")

    sqs = SQSEnumerator()
    graph_count = 0
    graphs_data = []

    for primal_data, dual_data in sqs.generate_pairs(dual_vertex_count):
        adjacency_list = dual_data["adjacency_list"]
        
        # 인접 리스트에서 edge_multiplicity 생성 (0-based)
        edge_multiplicity = GraphConverter.adjacency_to_edge_multiplicity(
            adjacency_list, is_one_based=True
        )

        # 통계 정보 계산
        stats = GraphConverter.get_graph_stats(edge_multiplicity, dual_vertex_count)

        # 4-regular 검증
        if not stats["is_regular"] or stats["regularity"] != 4:
            continue

        # Loop 검증
        has_loop = any(vertex in neighbors for vertex, neighbors in adjacency_list.items())
        if has_loop:
            print(f"[WARNING] Unexpected loop detected")
            continue

        graph_count += 1
        graphs_data.append((edge_multiplicity, stats, primal_data, dual_data))

        if max_graph_count and graph_count >= max_graph_count:
            break

    print(f"\n총 {len(graphs_data)}개의 비동형 SQS 열거됨")

    for graph_idx, (edge_multiplicity, stats, primal_data, dual_data) in enumerate(graphs_data, 1):
        print(f"\n{'#'*60}")
        print(f"# Solution Candidate #{graph_idx}")
        print(f"{'#'*60}")

        # Primal 그래프 정보
        primal_adj = primal_data["adjacency_list"]
        primal_edge_multiplicity = GraphConverter.adjacency_to_edge_multiplicity(
            primal_adj, is_one_based=True
        )
        primal_stats = GraphConverter.get_graph_stats(
            primal_edge_multiplicity, primal_data["vertex_count"]
        )

        print(f"\n[Q (Primal) - Simple Quadrangulation]")
        print(f"  Vertices: {primal_stats['vertex_count']}")
        print(f"  Edges: {primal_stats['edge_count']}")
        print(f"  Adjacency List (CW order, 1-based):")
        for v in sorted(primal_adj.keys()):
            neighbors = primal_adj[v]
            print(f"    {v}: {neighbors}")

        # Dual 그래프 정보
        print(f"\n[Q* (Dual) - 4-regular Planar Multigraph]")
        print(f"  Vertices: {stats['vertex_count']}")
        print(f"  Total Edges: {stats['edge_count']}")
        print(f"  Single Edges: {stats['single_edge_count']}")
        print(f"  Double Edges (Digons): {stats['double_edge_count']}")
        print(f"  Regularity: {stats['regularity']}-regular")
        print(f"  Loop-free: Yes")
        dual_adj = dual_data["adjacency_list"]
        print(f"  Adjacency List (CW order, 1-based):")
        for v in sorted(dual_adj.keys()):
            neighbors = dual_adj[v]
            print(f"    {v}: {neighbors}")

        # Gurobi 스타일 변수 출력 (Dual용)
        print(f"\n[Gurobi Variable Hints (Dual)]")
        gurobi_dict = GraphConverter.to_gurobi_start_dict(edge_multiplicity)
        for var_name, value in sorted(gurobi_dict.items()):
            print(f"  {var_name}.Start = {value}")

        # 인접 행렬 (옵션)
        if should_show_matrix:
            try:
                print(f"\n[Primal Adjacency Matrix]")
                primal_matrix = GraphConverter.to_adjacency_matrix(
                    primal_edge_multiplicity, primal_data["vertex_count"]
                )
                print(primal_matrix)

                print(f"\n[Dual Adjacency Matrix]")
                dual_matrix = GraphConverter.to_adjacency_matrix(edge_multiplicity, dual_vertex_count)
                print(dual_matrix)
            except ImportError as e:
                print(f"\n[Adjacency Matrix] {e}")

    return graphs_data


def export_to_json(graphs_data: List[Tuple], output_filename: str):
    """
    그래프 데이터를 JSON 파일로 내보내기

    Args:
        graphs_data: generate_sqs_dual_graphs()의 반환값
        output_filename: 출력 파일명
    """
    export_data = []
    for edge_multiplicity, stats, primal_data, dual_data in graphs_data:
        # JSON에서 tuple key를 지원하지 않으므로 리스트로 변환
        edge_list = [
            {"edge": [source, target], "multiplicity": multiplicity}
            for (source, target), multiplicity in edge_multiplicity.items()
        ]
        export_data.append({
            "stats": stats,
            "edges": edge_list,
            "gurobi_vars": GraphConverter.to_gurobi_start_dict(edge_multiplicity),
            "primal": primal_data,
            "dual": dual_data,
        })

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(export_data)} graphs to {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="SQS (Primal & Dual) 열거 및 ILP Warm Start 형식 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            논문 기반 성질 (Samuel Peltier et al., 2021):
            Q* (Dual)는 다음 성질을 만족:
            1. Planar graph
            2. Multigraph (double edge 허용)
            3. Loop 없음
            4. 4-regular

            plantri 옵션: -q -c2 -m2 -T (double_code)
            → Dual: 4-edge-connected quartic multigraph

            정점 수 관계:
            Q (Primal): n + 2 vertices
            Q* (Dual):  n vertices

            예제:
            python -m pyplantri.ilp_bridge 6              # Q*: 6정점, Q: 8정점
            python -m pyplantri.ilp_bridge 6 --show-matrix
            python -m pyplantri.ilp_bridge 8 --max-graphs 5
            python -m pyplantri.ilp_bridge 6 --export output.json
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
    dual_vertex_count = args.n
    if dual_vertex_count < 3:
        parser.error("n은 최소 3 이상이어야 합니다.")

    graphs_data = enumerate_sqs_graphs(
        dual_vertex_count,
        should_show_matrix=args.show_matrix,
        max_graph_count=args.max_graphs
    )

    if args.export and graphs_data:
        export_to_json(graphs_data, args.export)

    print("\n완료!")


if __name__ == "__main__":
    main()
