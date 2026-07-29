# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given the dual vertex count `n`, it enumerates all **non-isomorphic duals of simple quadrangulations of the sphere**, together with the corresponding primal and dual plane-graph topology.

## What is plantri?

[plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) is a C program for fast enumeration of plane graphs.

- **Bundled Version**: plantri 5.5 (May 17, 2024)
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
| Loop-free   | No loops because `-c2` makes Q bridge-free           |
| 4-regular   | Every vertex has exactly degree 4                    |

### Vertex Count Relationship (Euler's Formula)

For plane graphs: `V - E + F = 2`

| Graph          | Description                            | Vertices |
| -------------- | -------------------------------------- | -------- |
| **Q\*** (Dual) | 4-regular plane multigraph (loop-free) | n        |
| **Q** (Primal) | Simple Quadrangulation                 | n + 2    |

**Input Rule:** The input `n` to `QuadrangulationEnumerator` is the **number of vertices in Q\* (Dual)**. Internally, `n + 2` (the primal vertex count) is passed to plantri.

**Input Constraint:** The bundled count path supports `3 <= n <= 62`, while `-T` double-code generation supports `3 <= n <= 55`. The simple-quartic subclass is empty for `n < 6`.

Set `include_primal=False` to omit primal topology when only the dual is needed. Such objects still rely on plantri's generation guarantees for properties that cannot be certified from the stored dual alone.

### Adjacency List Order (Combinatorial Embedding)

The neighbor order in `PlaneGraph.dual_embedding` (or a parsed section's `cyclic_adjacency`) represents the **cyclic order** of edges at each vertex, given **clockwise (CW) as viewed from outside the sphere**. This cyclic ordering defines the **combinatorial embedding** of the plane graph.

plantri's `-T` (double_code) option preserves this exterior-view clockwise order around each vertex.

Without `-o`, as in the predefined quadrangulation modes, plantri identifies
an embedded graph with its mirror image. With `-o`, orientation-preserving
isomorphism classes are emitted separately.

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

## Output and Cache Safety

- `Plantri.run()` returns raw bytes and supports binary `planar_code`.
- `Plantri.iter_stdout_lines()` accepts only line-oriented ASCII, graph6, sparse6, or double-code output.
- Schema-v10 caches use compact footer manifests, a dense Graph-ID index, and independent pickle chunks. Chunk offsets and index layout are derived rather than stored. `load_graph_catalog()` opens cheaply and verifies chunks on access; `audit_all_graphs()` performs an explicit full audit. Cache loading requires `trusted=True`, and older formats must be regenerated.
- Saving semantically validates graphs by default. Only a producer that just enumerated with `validate=True` should pass `validate_graphs=False`.
- Loading always uses the restricted unpickler. `validate_graphs=True` semantically validates only the returned prefix; use `audit_all_graphs()` for the complete cache.

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
| 15               | 17         | 2,049,974            |
| 16               | 18         | 10,938,182           |
| 17               | 19         | 59,312,272           |
| 18               | 20         | 326,258,544          |
| 19               | 21         | 1,815,910,231        |
| 20               | 22         | 10,213,424,233       |

## License

- **pyplantri wrapper**: [MIT License](LICENSE)
- **plantri**: [Apache License 2.0](src/plantri/LICENSE-2.0.txt)
  - Authors: Gunnar Brinkmann, Brendan McKay

## References

- [plantri Official Page](https://users.cecs.anu.edu.au/~bdm/plantri/)
- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, "Generation of simple quadrangulations of the sphere", Discrete Mathematics, 305 (2005) 33-54.
- G. Brinkmann and B. D. McKay, "Fast generation of planar graphs", MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.
