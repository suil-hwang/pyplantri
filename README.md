# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given the dual vertex count `n`, it enumerates all **non-isomorphic Primal and Dual graphs**.

## What is plantri?

[plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) is a C program for fast enumeration of planar graphs.

- **Authors**: Gunnar Brinkmann (University of Ghent), Brendan McKay (Australian National University)
- **Key Feature**: Outputs exactly one representative from each isomorphism class without storing them
- **Speed**: Generates over 2,000,000 graphs per second
- **License**: Apache License 2.0

This package wraps plantri's **Simple Quadrangulation** enumeration functionality for use in Python.

### Related Paper

- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, **"Generation of simple quadrangulations of the sphere"**, Discrete Mathematics, 305 (2005) 33-54. [PDF](https://users.cecs.anu.edu.au/~bdm/papers/plantri-full.pdf)

## SQS and Dual Graph

### Q (Primal) - Simple Quadrangulation

- Planar graph where every face is a quadrilateral
- Simple graph (no loops, no multi-edges)
- Vertex count: `n + 2`

### Q\* (Dual) - 4-regular Planar Multigraph

| Property     | Description                               |
| ------------ | ----------------------------------------- |
| Planar graph | Embeddable on a plane                     |
| Multigraph   | Double edges allowed                      |
| Loop-free    | No loops (since Q has no degree-1 vertex) |
| 4-regular    | Every vertex has exactly degree 4         |

### Vertex Count Relationship (Euler's Formula)

| Graph          | Description                             | Vertices |
| -------------- | --------------------------------------- | -------- |
| **Q\*** (Dual) | 4-regular planar multigraph (loop-free) | n        |
| **Q** (Primal) | Simple Quadrangulation                  | n + 2    |

**Input Rule:** The input `n` to `SQSEnumerator` is the **number of vertices in Q\* (Dual)**. Internally, `n + 2` (the primal vertex count) is passed to plantri.

### Adjacency List Order

The neighbor order in the output `adjacency_list` is **CW (Clockwise)** order. This is because plantri's `-T` (double_code) option lists edges around each vertex in clockwise order in the planar embedding.

## Installation

```bash
git clone https://github.com/suil-hwang/pyplantri.git
cd pyplantri

pip install -e .
```

CMake automatically builds plantri during installation.

## Usage

### Basic Usage (SQSEnumerator)

```python
from pyplantri import SQSEnumerator

sqs = SQSEnumerator()

# Enumerate all non-isomorphic structures
for primal, dual in sqs.generate_pairs(4):  # Q*: 4 vertices, Q: 6 vertices
    print(f"Q (Primal): {primal['vertex_count']} vertices")
    print(f"Q* (Dual): {dual['vertex_count']} vertices")
    # adjacency_list is in CW (clockwise) order, 1-indexed
    print(f"  Dual adjacency list: {dual['adjacency_list']}")

# Count only
count = sqs.count(4)
print(f"Number of non-isomorphic structures for n=4: {count}")
```

### ILP-Ready Graphs (PlantriGraph)

For ILP solvers and numerical analysis, use `PlantriGraph` which provides:

- **0-based indexing** (Python standard)
- **Face extraction** (including digons)
- **CW/CCW neighbor access**
- **Graph validation**

```python
from pyplantri import enumerate_plantri_graphs, iter_plantri_graphs

# Enumerate all 6-vertex 4-regular planar multigraphs
graphs = enumerate_plantri_graphs(6)

for graph in graphs:
    print(f"Graph #{graph.graph_id}")
    print(f"  Vertices: {graph.num_vertices}")
    print(f"  Edges: {graph.num_simple_edges} unique, {graph.num_edges} total")
    print(f"  Faces: {graph.num_faces}")
    print(f"  Double edges: {graph.double_edges}")

    # Access CW-ordered neighbors (0-based)
    for v in range(graph.num_vertices):
        neighbors = graph.get_neighbors_cw(v)
        print(f"    Vertex {v}: {neighbors}")

    # Get consecutive pairs for ILP constraints
    pairs = graph.get_consecutive_pairs(0)
    print(f"  Consecutive pairs at v=0: {pairs}")

    # Validate graph invariants
    is_valid, errors = graph.validate()
    print(f"  Valid: {is_valid}")

# Memory-efficient iterator for large n
for graph in iter_plantri_graphs(12):
    # Process one graph at a time
    pass
```

### Graph Conversion Utilities

```python
from pyplantri import GraphConverter

# Convert 1-based adjacency list to 0-based embedding
adj_1based = {1: [2, 2, 3, 3], 2: [1, 1, 3, 3], 3: [1, 1, 2, 2]}
embedding = GraphConverter.to_zero_based_embedding(adj_1based)
# {0: (1, 1, 2, 2), 1: (0, 0, 2, 2), 2: (0, 0, 1, 1)}

# Extract faces from embedding
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

### Caching for Large Datasets

```python
from pyplantri import save_graphs_to_cache, load_graphs_from_cache

# Save to JSON cache
graphs = enumerate_plantri_graphs(10)
save_graphs_to_cache(graphs, "cache/n10_graphs.json")

# Load from cache
graphs = load_graphs_from_cache("cache/n10_graphs.json")
```

### CLI Usage

```bash
# Interactive SQS example
python -m pyplantri.example 4

# ILP Bridge with face information
python -m pyplantri.ilp_bridge 6 --show-faces
python -m pyplantri.ilp_bridge 6 --export output.json
python -m pyplantri.ilp_bridge 8 --max 5 -v
```

### Low-level API (Direct Plantri Call)

```python
from pyplantri import Plantri

plantri = Plantri()

# Generate Simple Quadrangulation and Dual simultaneously (double_code format)
output = plantri.run(6, options=["-q", "-c2", "-m2", "-T"])
print(output.decode())
# Output format: "6 ABCD AE ... 4 AEHB BHGC ..." (primal + dual)
# Neighbors of each vertex are listed in CW (clockwise) order
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
- `-T`: double_code output (primal + dual simultaneously, CW order)
- → Dual: 4-edge-connected quartic multigraph (double edges allowed, loop-free)

## Number of Non-isomorphic Graphs by n

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

## Project Structure

```
pyplantri/
├── CMakeLists.txt              # CMake build configuration
├── pyproject.toml              # Project metadata (numpy, scipy deps)
├── README.md
├── LICENSE                     # MIT License
├── build/                      # CMake build output (auto-generated)
│   └── cpXXX-win_amd64/
│       └── Release/
│           └── plantri.exe
└── src/
    ├── plantri/                # C source
    │   ├── plantri.c
    │   └── LICENSE-2.0.txt     # Apache 2.0 (plantri)
    └── pyplantri/              # Python package
        ├── __init__.py         # Public API exports
        ├── core.py             # Core classes:
        │                       #   - Plantri (executable wrapper)
        │                       #   - SQSEnumerator (graph enumeration)
        │                       #   - GraphConverter (format conversion)
        │                       #   - PlantriError (exception)
        ├── ilp_bridge.py       # ILP integration:
        │                       #   - PlantriGraph (immutable dataclass)
        │                       #   - enumerate_plantri_graphs()
        │                       #   - iter_plantri_graphs()
        │                       #   - save/load_graphs_to_cache()
        └── example.py          # Interactive CLI example
```

### Public API

```python
from pyplantri import (
    # Core
    Plantri,              # Plantri executable wrapper
    PlantriError,         # Exception class
    SQSEnumerator,        # SQS graph pair generator
    GraphConverter,       # Format conversion utilities

    # ILP Bridge
    PlantriGraph,         # Immutable 4-regular graph dataclass
    enumerate_plantri_graphs,  # List all graphs
    iter_plantri_graphs,       # Memory-efficient iterator
    save_graphs_to_cache,      # Save to JSON/pickle
    load_graphs_from_cache,    # Load from cache
)
```

## License

- **pyplantri wrapper**: [MIT License](LICENSE)
- **plantri**: [Apache License 2.0](src/plantri/LICENSE-2.0.txt)
  - Authors: Gunnar Brinkmann, Brendan McKay

## References

- [plantri Official Page](https://users.cecs.anu.edu.au/~bdm/plantri/)
- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, "Generation of simple quadrangulations of the sphere", Discrete Mathematics, 305 (2005) 33-54.
- G. Brinkmann and B. D. McKay, "Fast generation of planar graphs", MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.
