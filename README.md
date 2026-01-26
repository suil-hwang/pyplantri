# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given the dual vertex count `n`, it enumerates all **non-isomorphic Primal and Dual plane graphs**.

## Terminology: Planar Graph vs Plane Graph

| Term             | Definition                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------- |
| **Planar graph** | A graph that _can be_ embedded on a sphere/plane without edge crossings (abstract property) |
| **Plane graph**  | A planar graph _with a fixed embedding_ — cyclic edge ordering at each vertex is specified  |

> "A **plane graph** is a planar graph together with a crossing-free drawing."
> — Brinkmann & McKay, "Fast generation of planar graphs" (2007)

**plantri generates plane graphs**, not just planar graphs. The output includes the combinatorial embedding (clockwise neighbor ordering), which fully determines the topological structure.

## What is plantri?

[plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) is a C program for fast enumeration of plane graphs.

- **Authors**: Gunnar Brinkmann (University of Ghent), Brendan McKay (Australian National University)
- **Key Feature**: Outputs exactly one representative from each isomorphism class without storing them
- **Speed**: Generates over 2,000,000 graphs per second
- **License**: Apache License 2.0

This package wraps plantri's **Simple Quadrangulation** enumeration functionality for use in Python.

### Related Papers

- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, **"Generation of simple quadrangulations of the sphere"**, Discrete Mathematics, 305 (2005) 33-54. [PDF](https://users.cecs.anu.edu.au/~bdm/papers/plantri-full.pdf)
- G. Brinkmann and B. D. McKay, **"Fast generation of planar graphs"**, MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.

## SQS and Dual Graph

### Q (Primal) - Simple Quadrangulation

- **Plane graph** where every face is a quadrilateral
- Simple graph (no loops, no multi-edges)
- Vertex count: `n + 2`

### Q\* (Dual) - 4-regular Plane Multigraph

| Property    | Description                                          |
| ----------- | ---------------------------------------------------- |
| Plane graph | Embedded on a sphere with fixed cyclic edge ordering |
| Multigraph  | Double edges allowed                                 |
| Loop-free   | No loops (since Q has no degree-1 vertex)            |
| 4-regular   | Every vertex has exactly degree 4                    |

### Vertex Count Relationship (Euler's Formula)

For plane graphs: `V - E + F = 2`

| Graph          | Description                            | Vertices |
| -------------- | -------------------------------------- | -------- |
| **Q\*** (Dual) | 4-regular plane multigraph (loop-free) | n        |
| **Q** (Primal) | Simple Quadrangulation                 | n + 2    |

**Input Rule:** The input `n` to `QuadrangulationEnumerator` is the **number of vertices in Q\* (Dual)**. Internally, `n + 2` (the primal vertex count) is passed to plantri.

### Adjacency List Order (Combinatorial Embedding)

The neighbor order in the output `adjacency_list` represents the **cyclic order** of edges at each vertex, given in **clockwise (CW)** direction. This cyclic ordering defines the **combinatorial embedding** of the plane graph.

plantri's `-T` (double_code) option outputs edges in clockwise order around each vertex.

The combinatorial embedding uniquely determines:

- Face boundaries (via half-edge traversal)
- Topological structure on the sphere
- Dual graph structure

## Installation

```bash
git clone https://github.com/suil-hwang/pyplantri.git
cd pyplantri

pip install -e .
```

CMake automatically builds plantri during installation.

## Usage

### Basic Usage

```python
from pyplantri import QuadrangulationEnumerator

enumerator = QuadrangulationEnumerator()

# Enumerate all non-isomorphic plane graph structures
for primal, dual in enumerator.generate_pairs(4):  # Q*: 4 vertices, Q: 6 vertices
    print(f"Q (Primal): {primal['vertex_count']} vertices")
    print(f"Q* (Dual): {dual['vertex_count']} vertices")
    # adjacency_list is in CW (clockwise) order, 1-indexed
    # This ordering defines the combinatorial embedding
    print(f"  Dual adjacency list: {dual['adjacency_list']}")

# Count only
count = enumerator.count(4)
print(f"Number of non-isomorphic structures for n=4: {count}")
```

### Graph Conversion Utilities

```python
from pyplantri import GraphConverter

# Convert 1-based adjacency list to 0-based embedding
adj_1based = {1: [2, 2, 3, 3], 2: [1, 1, 3, 3], 3: [1, 1, 2, 2]}
embedding = GraphConverter.to_zero_based_embedding(adj_1based)
# {0: (1, 1, 2, 2), 1: (0, 0, 2, 2), 2: (0, 0, 1, 1)}

# Extract faces from embedding (uses the combinatorial embedding)
edge_mult = GraphConverter.adjacency_to_edge_multiplicity(adj_1based, is_one_based=True)
faces = GraphConverter.extract_faces(embedding, edge_mult)

# Check graph properties
GraphConverter.is_4_regular(adj_1based)  # True
GraphConverter.is_loop_free(adj_1based)  # True

# Convert to NumPy/SciPy matrices
import numpy as np
adj_matrix = GraphConverter.to_adjacency_matrix(edge_mult, vertex_count=3)
sparse_matrix = GraphConverter.to_sparse_adjacency_matrix(edge_mult, vertex_count=3)

# Generate Gurobi MIP start dictionary
gurobi_dict = GraphConverter.to_gurobi_start_dict(edge_mult)
# {'x[0,1]': 2, 'x[0,2]': 2, 'x[1,2]': 2}
```

### Plane Graph Enumeration (PlaneGraph)

`PlaneGraph` is an immutable dataclass representing a **plane graph** — a planar graph
with a fixed combinatorial embedding. Contains both dual (Q\*) and primal (Q) topology
with 0-based indexing, suitable for optimization solvers.

```python
from pyplantri import enumerate_plane_graphs, iter_plane_graphs

# Enumerate all plane graphs (loads into memory)
graphs = enumerate_plane_graphs(6, verbose=True)

for g in graphs:
    print(f"Graph #{g.graph_id}")
    print(f"  Vertices: {g.num_vertices}, Faces: {g.num_faces}")
    print(f"  Single edges: {len(g.single_edges)}, Double edges: {len(g.double_edges)}")

    # Embedding is 0-based, CW order (defines the plane graph structure)
    for v in range(g.num_vertices):
        print(f"  Vertex {v}: {g.get_neighbors_cw(v)}")

    # Validate graph invariants
    is_valid, errors = g.validate()
    print(f"  Valid: {is_valid}")

# Memory-efficient iterator (for large n)
for g in iter_plane_graphs(8):
    # Process one plane graph at a time
    pass
```

### CLI Usage

```bash
# Interactive SQS example
python -m pyplantri.example 4

# Plane graph enumeration with face information
python -m pyplantri.plane_graph 6 --show-faces
python -m pyplantri.plane_graph 6 --export output.json
python -m pyplantri.plane_graph 8 --max 5 -v
```

## plantri Options

| Option | Description                     | Notes                       |
| ------ | ------------------------------- | --------------------------- |
| `-q`   | Simple quadrangulation          | **Recommended** (SQS)       |
| `-Q`   | General quadrangulation         | Dual may have loops         |
| `-d`   | Output dual graph               |                             |
| `-a`   | ASCII output (vertex-based)     |                             |
| `-T`   | double_code output (edge-based) | **Recommended** primal+dual |
| `-c2`  | 2-connected                     | Use with `-q`               |
| `-m2`  | Minimum degree 2                | Use with `-q`               |

**Options for SQS enumeration: `-q -c2 -m2 -T`**

- `-q`: Simple quadrangulation (no multi-edges in primal)
- `-c2`: 2-connected
- `-m2`: Minimum degree 2
- `-T`: double_code output (primal + dual simultaneously, CW order for embedding)
- → Dual: 2-connected 4-regular plane multigraph (double edges allowed, loop-free)

## Number of Non-isomorphic Plane Graphs by n

| n (Q\* vertices) | Q vertices | Non-isomorphic count |
| ---------------- | ---------- | -------------------- |
| 3                | 5          | 1                    |
| 4                | 6          | 2                    |
| 5                | 7          | 3                    |
| 6                | 8          | 9                    |
| 7                | 9          | 18                   |
| 8                | 10         | 62                   |
| 9                | 11         | 198                  |
| 10               | 12         | 803                  |
| 11               | 13         | 3,378                |
| 12               | 14         | 15,882               |
| 13               | 15         | 77,185               |
| 14               | 16         | 393,075              |

## License

- **pyplantri wrapper**: [MIT License](LICENSE)
- **plantri**: [Apache License 2.0](src/plantri/LICENSE-2.0.txt)
  - Authors: Gunnar Brinkmann, Brendan McKay

## References

- [plantri Official Page](https://users.cecs.anu.edu.au/~bdm/plantri/)
- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, "Generation of simple quadrangulations of the sphere", Discrete Mathematics, 305 (2005) 33-54.
- G. Brinkmann and B. D. McKay, "Fast generation of planar graphs", MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.
