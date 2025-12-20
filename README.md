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

### SQSEnumerator (Recommended)

```python
from pyplantri import SQSEnumerator

sqs = SQSEnumerator()

# Enumerate all non-isomorphic structures
for primal, dual in sqs.generate_pairs(4):  # Q*: 4 vertices, Q: 6 vertices
    print(f"Q (Primal): {primal['vertex_count']} vertices")
    print(f"Q* (Dual): {dual['vertex_count']} vertices")
    # adjacency_list is in CW (clockwise) order
    print(f"  Primal adjacency list: {primal['adjacency_list']}")
    print(f"  Dual adjacency list: {dual['adjacency_list']}")

# Count only
count = sqs.count(4)
print(f"Number of non-isomorphic structures for n=4: {count}")
```

### CLI Usage

```bash
# Run SQS example
python -m pyplantri.example 4    # Q*: 4 vertices, Q: 6 vertices

# ILP Bridge (Gurobi Warm Start)
python -m pyplantri.ilp_bridge 6              # Q*: 6 vertices
python -m pyplantri.ilp_bridge 6 --show-matrix
python -m pyplantri.ilp_bridge 6 --export output.json
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
pyplantri/                              # Project root
├── CMakeLists.txt                      # CMake build configuration
├── setup.py                            # Python package setup (CMake integration)
├── pyproject.toml                      # Project metadata
├── README.md
├── build/                              # CMake build output (auto-generated on pip install)
│   └── cpXXX-cpXXX-win_amd64/          # Python version-specific build folder
│       └── Release/
│           └── plantri.exe             # Built executable
└── src/
    ├── plantri/                        # C source
    │   ├── plantri.c                   # plantri source
    │   └── LICENSE-2.0.txt             # Apache 2.0 License
    └── pyplantri/                      # Python package
        ├── __init__.py
        ├── core.py                     # Plantri, SQSEnumerator, GraphConverter
        ├── example.py                  # SQS example (python -m pyplantri.example)
        └── ilp_bridge.py               # ILP Warm Start conversion (python -m pyplantri.ilp_bridge)
```

## License

- **pyplantri wrapper**: [MIT License](LICENSE)
- **plantri**: [Apache License 2.0](src/plantri/LICENSE-2.0.txt)
  - Authors: Gunnar Brinkmann, Brendan McKay

## References

- [plantri Official Page](https://users.cecs.anu.edu.au/~bdm/plantri/)
- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, "Generation of simple quadrangulations of the sphere", Discrete Mathematics, 305 (2005) 33-54.
- G. Brinkmann and B. D. McKay, "Fast generation of planar graphs", MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.
