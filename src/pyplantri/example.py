# src/pyplantri/example.py
import argparse
from typing import Dict, List

from .core import GraphConverter, SQSEnumerator


def display_graph_info(graph_data: Dict, title: str = "Graph") -> None:
    """Displays graph information with CW-ordered adjacency list."""
    adjacency_list = graph_data["adjacency_list"]
    vertex_count = graph_data["vertex_count"]
    edge_count = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Vertices: {vertex_count}")
    print(f"Edges: {edge_count}")
    print(f"\nAdjacency List (CW order):")
    for vertex in sorted(adjacency_list.keys()):
        neighbors = " ".join(map(str, adjacency_list[vertex]))
        degree = len(adjacency_list[vertex])
        print(f"  {vertex}: [{neighbors}] (degree: {degree})")


def validate_dual_graph(adjacency_list: Dict[int, List[int]]) -> None:
    """Validates and prints dual graph properties."""
    if GraphConverter.is_4_regular(adjacency_list):
        print("  [OK] All vertices have degree 4 (4-regular)")

    if GraphConverter.is_loop_free(adjacency_list):
        print("  [OK] Loop-free")

    edge_multiplicity = GraphConverter.adjacency_to_edge_multiplicity(
        adjacency_list, is_one_based=True
    )
    has_double_edge = any(mult > 1 for mult in edge_multiplicity.values())
    if has_double_edge:
        print("  [INFO] Contains double edges (multigraph)")


def main() -> None:
    """Main entry point for SQS example CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Simple Quadrangulation (Q) and 4-regular multigraph (Q*)",
    )
    parser.add_argument("n", type=int, help="Number of vertices in Q* (minimum 3)")
    args = parser.parse_args()

    dual_vertex_count = args.n
    if dual_vertex_count < 3:
        parser.error("n must be at least 3.")

    primal_vertex_count = dual_vertex_count + 2

    print("=" * 60)
    print("pyplantri - Simple Quadrangulation (SQS) Generator")
    print("=" * 60)
    print("\nBased on: Samuel Peltier et al. (2021)")
    print("  Q:  Simple quadrangulation (no loops/multi-edges)")
    print("  Q*: 4-regular planar multigraph (loop-free, double edges allowed)")
    print(f"\nplantri options: -q -c2 -m2 -T (double_code)")
    print(f"Relationship: Q*({dual_vertex_count}) <- Q({primal_vertex_count})")

    sqs = SQSEnumerator()
    total_count = sqs.count(dual_vertex_count)

    if total_count == 0:
        print(f"\nNo simple quadrangulations exist for n={dual_vertex_count}.")
        return

    print(f"\nNon-isomorphic structures: {total_count}")

    response = input("\nView structures? (y/n): ")
    if response.lower() != 'y':
        print("Exiting.")
        return

    print("\nGenerating...")
    for pair_index, (primal_data, dual_data) in enumerate(
        sqs.generate_pairs(dual_vertex_count), start=1
    ):
        print(f"\n{'#'*60}")
        print(f"# Structure {pair_index}/{total_count}")
        print(f"{'#'*60}")

        display_graph_info(dual_data, f"Q* ({dual_vertex_count} vertices)")
        display_graph_info(primal_data, f"Q ({primal_vertex_count} vertices)")
        validate_dual_graph(dual_data["adjacency_list"])

        if pair_index < total_count:
            response = input("\nNext structure? (Enter/q): ")
            if response.lower() == 'q':
                break

    print("\nDone!")


if __name__ == "__main__":
    main()