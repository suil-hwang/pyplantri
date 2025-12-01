"""
pyplantri 사용 예제

n을 입력받아 General Quadrangulation (Q)과
그에 대응하는 4-regular planar multigraph (Q*)를 출력합니다.

- Q: (n+2) vertices, 2n edges, n faces (all 4-gon)
- Q*: n vertices, 2n edges, 4-regular multigraph
"""

from pyplantri import Plantri


def parse_plantri_ascii(ascii_line: str) -> dict:
    """
    plantri ASCII 출력을 파싱하여 인접 리스트로 변환

    plantri 출력 형식: "5 b,acc,bbded,cc,c"
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


def generate_general_quadrangulation(n: int, dual: bool = False):
    """
    General Quadrangulation 생성 (-Q 옵션 사용)

    Args:
        n: dual vertices 개수 (최소 3)
           - Primal Q: (n+2) vertices
           - Dual Q*: n vertices, 4-regular
        dual: True이면 Q* (4-regular multigraph) 출력
    """
    plantri = Plantri()
    plantri_n = n + 2  # plantri 입력값
    options = ["-Q", "-a"]  # General quadrangulation, ASCII output

    if dual:
        options.append("-d")

    output = plantri.run(plantri_n, options)

    for line in output.decode(errors="replace").split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            yield line


def main():
    print("=" * 50)
    print("pyplantri - General Quadrangulation Generator")
    print("=" * 50)
    print("\nQ:  General quadrangulation (primal)")
    print("Q*: 4-regular planar multigraph (dual)")
    print("\n관계: n (입력) -> Q는 (n+2)정점, Q*는 n정점")

    # n 입력 (dual vertices 개수)
    while True:
        try:
            n = int(input("\nn을 입력하세요 (Q*의 정점 수, 최소 3): "))
            if n < 3:
                print("최소 3 이상이어야 합니다.")
                continue
            break
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    print(f"\nn={n} -> Q: {n+2}정점, Q*: {n}정점")
    print(f"생성 중...")

    # Q (primal) 생성
    primal_graphs = list(generate_general_quadrangulation(n, dual=False))

    if not primal_graphs:
        print(f"n={n}에 대한 quadrangulation이 없습니다.")
        return

    print(f"총 {len(primal_graphs)}개의 비동형 구조 발견")

    # Q* (dual) 생성
    dual_graphs = list(generate_general_quadrangulation(n, dual=True))

    # 각 그래프 쌍 출력
    for i, (primal_ascii, dual_ascii) in enumerate(zip(primal_graphs, dual_graphs), 1):
        print(f"\n{'#'*50}")
        print(f"# 구조 {i}/{len(primal_graphs)}")
        print(f"{'#'*50}")

        # Q* (4-regular multigraph)
        dual_data = parse_plantri_ascii(dual_ascii)
        print_graph_info(dual_data, f"Q* (4-Regular Multigraph, {n} vertices)")

        adj = dual_data["adjacency"]
        degrees = [len(neighbors) for neighbors in adj.values()]
        if degrees and all(d == 4 for d in degrees):
            print("  [OK] 모든 정점의 차수가 4 (4-regular)")

        # Q (quadrangulation)
        primal_data = parse_plantri_ascii(primal_ascii)
        print_graph_info(primal_data, f"Q (Quadrangulation, {n+2} vertices)")

        # 사용자 입력 대기
        if i < len(primal_graphs):
            response = input("\n다음 구조? (Enter/q): ")
            if response.lower() == 'q':
                break

    print("\n완료!")


if __name__ == "__main__":
    main()

# python example.py