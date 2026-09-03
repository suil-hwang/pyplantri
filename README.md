# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given a supported dual vertex count `n`, it enumerates one representative of each **plane-map isomorphism class of duals of simple quadrangulations of the sphere** as compact candidate plane graphs `G*`. Global reflection is identified.
The candidate primal `G` and dual `G*` topology are derived exactly from the stored dart involution; realized SQS graphs `Q` and `Q*` belong to the downstream assignment and geometry pipeline.

## What is plantri?

[plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) is a C program for fast enumeration of plane graphs.

- **Bundled Version**: plantri 5.5 (May 17, 2024)
- **Authors**: Gunnar Brinkmann (University of Ghent), Brendan McKay (Australian National University)
- **Key Feature**: Outputs exactly one representative from each isomorphism class without storing them
- **Speed**: Generates over 2,000,000 graphs per second
- **License**: Apache License 2.0

This package builds a dedicated plantri FILTER executable, `plantri_sqs`, for **Simple Quadrangulation** enumeration.
The FILTER converts each generated map in C and streams one fixed record directly to Python:

```text
twin[4n] + descending primal degree profile[n+2]
```

The record order is the namespace-local `graph_id`. Python never decodes `planar_code`, reconstructs primal face orbits, or uses a worker pool.
The FILTER enforces its output contract on every record (a connected map fully covered by disjoint 4-faces with `ne = 4nv - 8`, in-range darts, and a complete degree profile) before Python's private trusted construction path; plantri's own rotation-system and simplicity invariants are re-verified only in a `-DSQS_VERIFY` build.
Public raw construction and cache restoration independently validate the common topology family from `twin`; hashes verify stored bytes and order, not generator provenance or enumeration completeness.

### Related Papers

- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, **"Generation of simple quadrangulations of the sphere"**, Discrete Mathematics, 305 (2005) 33-54. [PDF](https://users.cecs.anu.edu.au/~bdm/papers/plantri-full.pdf)
- G. Brinkmann and B. D. McKay, **"Fast generation of planar graphs"**, MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.

## Candidate Primal and Dual Plane Graphs

### G (Primal) - Simple Quadrangulation

- **Plane graph** where every face boundary is a 4-cycle
- Simple graph (no loops, no multi-edges)
- Vertex count: `n + 2`

### G\* (Dual) - 4-regular Plane Multigraph

| Property         | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| Plane map        | Embedded on a sphere with fixed cyclic edge ordering              |
| Multigraph       | Parallel edges are permitted; simple members are included         |
| Loop-free        | A simple quadrangulation is 2-connected and therefore bridge-free |
| 4-regular        | Every vertex has exactly degree 4                                 |
| 4-edge-connected | Every non-trivial edge cut contains at least four edge copies     |

### Primal Minimum-Degree Policies

`PrimalMinimumDegree` selects the minimum-degree policy for plantri's primal quadrangulation `G` through the `primal_minimum_degree` keyword.
It is an enumeration policy, not a claim that every emitted graph attains the lower bound exactly.
Callers must pass an enum member; bare numeric or string values are rejected.

| Enum member  | Value | Primal plantri flags | Exact dual family                                                                     |
| ------------ | ----- | -------------------- | ------------------------------------------------------------------------------------- |
| `AT_LEAST_2` | `2`   | `-q -c2 -m2`         | Loop-free, 4-regular, 4-edge-connected plane multigraphs; parallel edges may occur    |
| `AT_LEAST_3` | `3`   | `-q -c2`             | Simple, 4-regular, 4-edge-connected plane graphs; primal minimum degree is at least 3 |

`AT_LEAST_3` uses plantri's default minimum degree 3; `-m3` is omitted.
Its stream is a topological subset of the `AT_LEAST_2` stream, but the two source streams use independent source-order `graph_id` namespaces.

`DualPlaneGraph` owns the compact candidate dual `G*`: `embedding`, `faces`, `right_faces`, `support_edges`, `edge_multiplicity`, `double_edges`, and `face_size_sequence`, plus the dart operations `vertex`, `neighbor`, `next_at_vertex`, `prev_at_vertex`, `right_face_next`, and `right_face`.
`SimpleQuadrangulation(dual)` is candidate `G` read over the same darts with the vertex and face roles exchanged: a dart leaves the dual face on its right, primal rotations are dual right-face orbits, and primal faces are dual vertices. It exposes `embedding`, `faces`, `edges`, `degrees`, `degree_sequence`, `min_degree`, the same six dart operations, and `dual`, which points back to the paired candidate dual. Nothing is copied: `twin` and `graph_id` are shared, and the `right_faces` labels are computed once on the dual.
Neither type denotes the realized `Q` or `Q*` produced by the downstream geometry pipeline.
Direct `DualPlaneGraph(twin, graph_id)` construction validates a connected spherical map paired with a simple quadrangulation whose faces are 4-cycles and whose minimum degree is at least two. Derived support rotations remain a downstream SQS concern rather than part of this topology API.

```python
dual = enumerate_simple_quadrangulation_duals(n, max_count=1).graphs[0]
primal = SimpleQuadrangulation(dual)

assert primal.dual is dual
assert primal.num_vertices == dual.num_faces == n + 2
assert primal.degree_sequence == dual.face_size_sequence
dual.embedding    # vertex-indexed rotations of G*
primal.embedding  # rotations of G, indexed by dual right face
primal.faces      # 4-cycles of G, indexed by dual vertex
```

### Vertex Count Relationship (Euler's Formula)

For plane graphs: `V - E + F = 2`

| Graph          | Description                            | Vertices |
| -------------- | -------------------------------------- | -------- |
| **G\*** (Dual) | 4-regular plane multigraph (loop-free) | n        |
| **G** (Primal) | Simple Quadrangulation                 | n + 2    |

**Input Rule:** The input `n` to `enumerate_simple_quadrangulation_duals()` is the **number of vertices in G\* (Dual)**. Internally, `n + 2` (the candidate-primal vertex count) is passed to `plantri_sqs`.

**Input Constraint:** The bundled count and materialization paths support `3 <= n <= 61`. The `AT_LEAST_3` stream is empty for `n < 6`.
The full literature family selected by `AT_LEAST_2` also contains the square's two-vertex dual, which lies outside this wrapper's supported range.

## Installation

```bash
git clone https://github.com/suil-hwang/pyplantri.git
cd pyplantri

pip install -e .
```

## Number of `AT_LEAST_2` Dual Plane Maps by n

| n (G\* vertices) | G vertices | Non-isomorphic count |
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

[Apache License 2.0](src/LICENSE-2.0.txt)

- Authors: Gunnar Brinkmann, Brendan McKay

## References

- [plantri Official Page](https://users.cecs.anu.edu.au/~bdm/plantri/)
- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas and P. Wollan, "Generation of simple quadrangulations of the sphere", Discrete Mathematics, 305 (2005) 33-54.
- G. Brinkmann and B. D. McKay, "Fast generation of planar graphs", MATCH Commun. Math. Comput. Chem., 58 (2007) 323-357.
