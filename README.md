# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given the dual vertex count `n`, it enumerates all **non-isomorphic Primal and Dual plane graphs**.

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
