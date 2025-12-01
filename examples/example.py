"""
pyplantri 사용 예제

Simple Quadrangulation (SQS)과 그 Dual인 4-regular planar multigraph (Q*)를 생성합니다.

논문 참조: Samuel Peltier et al. (2021)
- Q (Primal): Simple quadrangulation (no loops, no multi-edges)
- Q* (Dual): 4-regular planar multigraph (no loops, multi-edges allowed)

Dual Q*의 성질:
1. Planar graph (평면 그래프)
2. Multigraph (double edge 허용)
3. Loop 없음 (Q에 degree 1 vertex가 없으므로)
4. 4-regular (모든 vertex가 정확히 degree 4)

plantri 옵션:
- `-q -c2 -m2`: Simple quadrangulation, 2-connected, min degree 2
  → Dual: 4-edge-connected quartic multigraph (double edge 허용)
"""

import argparse
from pyplantri import Plantri


def parse_plantri_ascii(ascii_line: str) -> dict:
    """
    plantri ASCII 출력을 파싱하여 인접 리스트로 변환

    plantri 출력 형식: "8 bcd,aef,afg,age,bdh,bhc,chd,egf"
    - 첫 번째 숫자는 정점 수
    - 각 정점의 이웃을 알파벳으로 표시 (a=1, b=2, ...)
    - 쉼표로 구분

    Returns:
        {"n_vertices": int, "adjacency": {vertex: [neighbors]}}
    """
    if ascii_line.startswith("Graph"):
        return {"n_vertices": 0, "adjacency": {}}

    parts = ascii_line.strip().split()
    if len(parts) < 2:
        return {"n_vertices": 0, "adjacency": {}}

    n_vertices = int(parts[0])
    adj_str = parts[1]

    adjacency = {}
    vertex_lists = adj_str.split(",")

    for i, neighbors_str in enumerate(vertex_lists, start=1):
        neighbors = [ord(c) - ord('a') + 1 for c in neighbors_str]
        adjacency[i] = neighbors

    return {"n_vertices": n_vertices, "adjacency": adjacency}


def print_graph_info(graph_data: dict, title: str = "Graph"):
    """그래프 정보 출력"""
    adj = graph_data["adjacency"]
    n_vertices = graph_data["n_vertices"]
    n_edges = sum(len(neighbors) for neighbors in adj.values()) // 2

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"정점 수: {n_vertices}")
    print(f"간선 수: {n_edges}")
    print(f"\n인접 리스트:")
    for v in sorted(adj.keys()):
        neighbors = " ".join(map(str, adj[v]))
        degree = len(adj[v])
        print(f"  {v}: [{neighbors}] (차수: {degree})")


def generate_simple_quadrangulation(n_dual: int, dual: bool = False):
    """
    Simple Quadrangulation 생성 (-q -c2 -m2 옵션 사용)

    plantri 옵션 설명:
    - -q: Simple quadrangulation (primal에 multi-edge 없음)
    - -c2: 2-connected
    - -m2: minimum degree 2
    → Dual: 4-edge-connected quartic multigraph (double edge 허용, loop 없음)

    Args:
        n_dual: Q*의 정점 수 (최소 2)
        dual: True이면 Q* (4-regular planar multigraph) 출력

    Yields:
        plantri ASCII 출력 라인
    """
    plantri = Plantri()
    # -q: Simple quadrangulation
    # -c2: 2-connected
    # -m2: minimum degree 2
    # -a: ASCII output
    options = ["-q", "-c2", "-m2", "-a"]

    if dual:
        options.append("-d")

    # Q*의 정점 수 = Q의 정점 수 - 2 (Euler 공식)
    # 따라서 Q의 정점 수 = n_dual + 2
    n_primal = n_dual + 2
    output = plantri.run(n_primal, options)

    for line in output.decode(errors="replace").split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            yield line


def main():
    parser = argparse.ArgumentParser(
        description="Simple Quadrangulation (Q) 및 4-regular planar multigraph (Q*) 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                논문 기반 성질 (Samuel Peltier et al., 2021):
                - Q (Primal): Simple quadrangulation (loop/multi-edge 없음)
                - Q* (Dual): 4-regular planar multigraph (loop 없음, double edge 허용)

                plantri 옵션: -q -c2 -m2
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

    n_dual = args.n
    if n_dual < 3:
        parser.error("n은 최소 3 이상이어야 합니다.")

    n_primal = n_dual + 2  # Euler 공식: Q의 정점 수

    print("=" * 60)
    print("pyplantri - Simple Quadrangulation (SQS) Generator")
    print("=" * 60)
    print("\n논문 기반 성질 (Samuel Peltier et al., 2021):")
    print("  Q:  Simple quadrangulation (loop/multi-edge 없음)")
    print("  Q*: 4-regular planar multigraph (loop 없음, double edge 허용)")
    print(f"\nplantri 옵션: -q -c2 -m2")
    print(f"관계: Q*({n_dual}정점) <- Q({n_primal}정점)")

    print(f"\n생성 중...")

    # Q (primal) 생성
    primal_graphs = list(generate_simple_quadrangulation(n_dual, dual=False))

    if not primal_graphs:
        print(f"n={n_dual}에 대한 simple quadrangulation이 없습니다.")
        return

    # Q* (dual) 생성
    dual_graphs = list(generate_simple_quadrangulation(n_dual, dual=True))

    # 개수 출력 및 진행 여부 확인
    print(f"\n구성 가능한 비동형 구조:")
    print(f"  Q* (4-regular multigraph, {n_dual}정점): {len(dual_graphs)}개")
    print(f"  Q  (Simple Quadrangulation, {n_primal}정점): {len(primal_graphs)}개")

    response = input("\n구조를 확인하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("종료합니다.")
        return

    # 각 그래프 쌍 출력
    for i, (primal_ascii, dual_ascii) in enumerate(zip(primal_graphs, dual_graphs), 1):
        print(f"\n{'#'*60}")
        print(f"# 구조 {i}/{len(dual_graphs)}")
        print(f"{'#'*60}")

        # Q* (4-regular multigraph) 먼저 출력
        dual_data = parse_plantri_ascii(dual_ascii)
        print_graph_info(dual_data, f"Q* (4-Regular Planar Multigraph, {n_dual} vertices)")

        # Q (Simple Quadrangulation)
        primal_data = parse_plantri_ascii(primal_ascii)
        print_graph_info(primal_data, f"Q (Simple Quadrangulation, {n_primal} vertices)")

        adj = dual_data["adjacency"]
        degrees = [len(neighbors) for neighbors in adj.values()]
        if degrees and all(d == 4 for d in degrees):
            print("  [OK] 모든 정점의 차수가 4 (4-regular)")

        # Loop 체크
        has_loop = False
        for v, neighbors in adj.items():
            if v in neighbors:
                has_loop = True
                break
        if not has_loop:
            print("  [OK] Loop 없음")

        # Double edge 체크
        from collections import Counter
        has_double_edge = False
        for v, neighbors in adj.items():
            cnt = Counter(neighbors)
            if any(c > 1 for c in cnt.values()):
                has_double_edge = True
                break
        if has_double_edge:
            print("  [INFO] Double edge 존재 (multigraph)")

        # 사용자 입력 대기
        if i < len(primal_graphs):
            response = input("\n다음 구조? (Enter/q): ")
            if response.lower() == 'q':
                break

    print("\n완료!")


if __name__ == "__main__":
    main()

# 빌드 삭제 명령어
# Remove-Item -Recurse -Force build, pyplantri.egg-info, pyplantri\bin, pyplantri\__pycache__
# pip install -e .
# pip uninstall pyplantri

# python examples/example.py 3  # Q*: 3정점, Q: 5정점
# python examples/example.py 4  # Q*: 6정점, Q: 8정점