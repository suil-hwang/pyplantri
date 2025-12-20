# pyplantri/example.py
import argparse
from collections import Counter
from .core import SQSEnumerator


def print_graph_info(graph_data: dict, title: str = "Graph"):
    """그래프 정보 출력 (인접 리스트는 CW 순서)"""
    adjacency_list = graph_data["adjacency_list"]
    vertex_count = graph_data["vertex_count"]
    edge_count = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"정점 수: {vertex_count}")
    print(f"간선 수: {edge_count}")
    print(f"\n인접 리스트 (CW 순서):")
    for vertex in sorted(adjacency_list.keys()):
        neighbors = " ".join(map(str, adjacency_list[vertex]))
        degree = len(adjacency_list[vertex])
        print(f"  {vertex}: [{neighbors}] (차수: {degree})")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Quadrangulation (Q) 및 4-regular planar multigraph (Q*) 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                논문 기반 성질 (Samuel Peltier et al., 2021):
                - Q (Primal): Simple quadrangulation (loop/multi-edge 없음)
                - Q* (Dual): 4-regular planar multigraph (loop 없음, double edge 허용)

                plantri 옵션: -q -c2 -m2 -T (double_code)
                → Dual: 4-edge-connected quartic multigraph

                정점 수 관계:
                Q* (Dual)의 정점 수 = n
                Q (Primal)의 정점 수 = n + 2 (Euler 공식)
                """
    )
    parser.add_argument(
        "n",
        type=int,
        help="Q*(Dual)의 정점 수 (최소 3)"
    )
    args = parser.parse_args()

    dual_vertex_count = args.n
    if dual_vertex_count < 3:
        parser.error("n은 최소 3 이상이어야 합니다.")

    primal_vertex_count = dual_vertex_count + 2  # Euler 공식

    print("=" * 60)
    print("pyplantri - Simple Quadrangulation (SQS) Generator")
    print("=" * 60)
    print("\n논문 기반 성질 (Samuel Peltier et al., 2021):")
    print("  Q:  Simple quadrangulation (loop/multi-edge 없음)")
    print("  Q*: 4-regular planar multigraph (loop 없음, double edge 허용)")
    print(f"\nplantri 옵션: -q -c2 -m2 -T (double_code)")
    print(f"관계: Q*({dual_vertex_count}정점) <- Q({primal_vertex_count}정점)")

    print("\n생성 중...")

    # SQSEnumerator를 사용하여 생성
    sqs = SQSEnumerator()
    graph_pairs = list(sqs.generate_pairs(dual_vertex_count))

    if not graph_pairs:
        print(f"n={dual_vertex_count}에 대한 simple quadrangulation이 없습니다.")
        return

    # 개수 출력 및 진행 여부 확인
    print(f"\n구성 가능한 비동형 구조: {len(graph_pairs)}개")
    print(f"  Q* (4-regular multigraph): {dual_vertex_count}정점")
    print(f"  Q  (Simple Quadrangulation): {primal_vertex_count}정점")

    response = input("\n구조를 확인하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("종료합니다.")
        return

    # 각 그래프 쌍 출력
    for graph_idx, (primal_data, dual_data) in enumerate(graph_pairs, 1):
        print(f"\n{'#'*60}")
        print(f"# 구조 {graph_idx}/{len(graph_pairs)}")
        print(f"{'#'*60}")

        # Q* (4-regular multigraph) 먼저 출력
        print_graph_info(dual_data, f"Q* (4-Regular Planar Multigraph, {dual_vertex_count} vertices)")

        # Q (Simple Quadrangulation)
        print_graph_info(primal_data, f"Q (Simple Quadrangulation, {primal_vertex_count} vertices)")

        adjacency_list = dual_data["adjacency_list"]
        degrees = [len(neighbors) for neighbors in adjacency_list.values()]
        is_4_regular = degrees and all(degree == 4 for degree in degrees)
        if is_4_regular:
            print("  [OK] 모든 정점의 차수가 4 (4-regular)")

        # Loop 체크
        has_loop = any(vertex in neighbors for vertex, neighbors in adjacency_list.items())
        if not has_loop:
            print("  [OK] Loop 없음")

        # Double edge 체크
        has_double_edge = False
        for vertex, neighbors in adjacency_list.items():
            neighbor_count = Counter(neighbors)
            if any(count > 1 for count in neighbor_count.values()):
                has_double_edge = True
                break
        if has_double_edge:
            print("  [INFO] Double edge 존재 (multigraph)")

        # 사용자 입력 대기
        if graph_idx < len(graph_pairs):
            response = input("\n다음 구조? (Enter/q): ")
            if response.lower() == 'q':
                break

    print("\n완료!")


if __name__ == "__main__":
    main()

# Usage:
# python src/pyplantri/example.py 4