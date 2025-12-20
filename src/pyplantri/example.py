# src/pyplantri/example.py
"""Example script for Simple Quadrangulation on Sphere (SQS) enumeration.

This module provides an interactive CLI for exploring SQS structures,
displaying both primal (Q) and dual (Q*) graphs.

Usage:
    python -m pyplantri.example 4
"""
import argparse
from typing import Dict, List

from .core import GraphConverter, SQSEnumerator


def display_graph_info(graph_data: Dict, title: str = "Graph") -> None:
    """Displays graph information with CW-ordered adjacency list.

    Args:
        graph_data: Dictionary containing 'vertex_count' and 'adjacency_list'.
        title: Display title for the graph section.
    """
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
    """Validates and prints dual graph properties.

    Checks for 4-regularity, loop-free, and multi-edge properties
    using GraphConverter utility methods.

    Args:
        adjacency_list: Dual graph adjacency list (1-indexed).
    """
    # 4-regular check
    if GraphConverter.is_4_regular(adjacency_list):
        print("  [OK] All vertices have degree 4 (4-regular)")

    # Loop check
    if GraphConverter.is_loop_free(adjacency_list):
        print("  [OK] Loop-free")

    # Double edge check using edge multiplicity
    edge_multiplicity = GraphConverter.adjacency_to_edge_multiplicity(
        adjacency_list, is_one_based=True
    )
    has_double_edge = any(mult > 1 for mult in edge_multiplicity.values())
    if has_double_edge:
        print("  [INFO] Contains double edges (multigraph)")


def main() -> None:
    """Main entry point for SQS example CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Simple Quadrangulation (Q) and 4-regular planar multigraph (Q*)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Based on: Samuel Peltier et al. (2021)
                - Q (Primal): Simple quadrangulation (no loops/multi-edges)
                - Q* (Dual): 4-regular planar multigraph (loop-free, double edges allowed)

                plantri options: -q -c2 -m2 -T (double_code)
                -> Dual: 4-edge-connected quartic multigraph

                Vertex count relationship:
                Q* (Dual): n vertices
                Q (Primal): n + 2 vertices (Euler's formula)
                """,
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of vertices in Q* (Dual), minimum 3"
    )
    args = parser.parse_args()

    dual_vertex_count = args.n
    if dual_vertex_count < 3:
        parser.error("n must be at least 3.")

    primal_vertex_count = dual_vertex_count + 2  # Euler's formula

    print("=" * 60)
    print("pyplantri - Simple Quadrangulation (SQS) Generator")
    print("=" * 60)
    print("\nBased on: Samuel Peltier et al. (2021)")
    print("  Q:  Simple quadrangulation (no loops/multi-edges)")
    print("  Q*: 4-regular planar multigraph (loop-free, double edges allowed)")
    print(f"\nplantri options: -q -c2 -m2 -T (double_code)")
    print(f"Relationship: Q*({dual_vertex_count} vertices) <- Q({primal_vertex_count} vertices)")

    # Count first (memory-efficient)
    sqs = SQSEnumerator()
    total_count = sqs.count(dual_vertex_count)

    if total_count == 0:
        print(f"\nNo simple quadrangulations exist for n={dual_vertex_count}.")
        return

    print(f"\nNon-isomorphic structures: {total_count}")
    print(f"  Q* (4-regular multigraph): {dual_vertex_count} vertices")
    print(f"  Q  (Simple Quadrangulation): {primal_vertex_count} vertices")

    response = input("\nView structures? (y/n): ")
    if response.lower() != 'y':
        print("Exiting.")
        return

    # Iterate through pairs (memory-efficient)
    print("\nGenerating...")
    for pair_index, (primal_data, dual_data) in enumerate(
        sqs.generate_pairs(dual_vertex_count), start=1
    ):
        print(f"\n{'#'*60}")
        print(f"# Structure {pair_index}/{total_count}")
        print(f"{'#'*60}")

        # Display Q* (4-regular multigraph) first
        display_graph_info(
            dual_data,
            f"Q* (4-Regular Planar Multigraph, {dual_vertex_count} vertices)"
        )

        # Display Q (Simple Quadrangulation)
        display_graph_info(
            primal_data,
            f"Q (Simple Quadrangulation, {primal_vertex_count} vertices)"
        )

        # Validate dual graph properties
        validate_dual_graph(dual_data["adjacency_list"])

        # Wait for user input
        if pair_index < total_count:
            response = input("\nNext structure? (Enter/q): ")
            if response.lower() == 'q':
                break

    print("\nDone!")


if __name__ == "__main__":
    main()