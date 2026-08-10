# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given a supported dual vertex count `n`, it enumerates one representative of each **plane-map isomorphism class of duals of simple quadrangulations of the sphere** as compact quartic plane maps. Global reflection is identified. Their primal and dual topology is derived exactly from the stored dart involution.

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

- **Plane graph** where every face boundary is a 4-cycle
- Simple graph (no loops, no multi-edges)
- Vertex count: `n + 2`

### Q\* (Dual) - 4-regular Plane Multigraph

| Property         | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| Plane map        | Embedded on a sphere with fixed cyclic edge ordering              |
| Multigraph       | Parallel edges are permitted; simple members are included         |
| Loop-free        | A simple quadrangulation is 2-connected and therefore bridge-free |
| 4-regular        | Every vertex has exactly degree 4                                 |
| 4-edge-connected | Every non-trivial edge cut contains at least four edge copies     |

### Enumeration Families

| Enum member          | Primal plantri flags | Exact dual family                                                                     |
| -------------------- | -------------------- | ------------------------------------------------------------------------------------- |
| `QUARTIC_MULTIGRAPH` | `-q -c2 -m2`         | Loop-free, 4-regular, 4-edge-connected plane multigraphs; parallel edges may occur    |
| `SIMPLE_QUARTIC`     | `-q -c2`             | Simple, 4-regular, 4-edge-connected plane graphs; primal minimum degree is at least 3 |

The second family is a topological subset of the first, but the two plantri
streams use independent source-order `graph_id` namespaces.

### Vertex Count Relationship (Euler's Formula)

For plane graphs: `V - E + F = 2`

| Graph          | Description                            | Vertices |
| -------------- | -------------------------------------- | -------- |
| **Q\*** (Dual) | 4-regular plane multigraph (loop-free) | n        |
| **Q** (Primal) | Simple Quadrangulation                 | n + 2    |

**Input Rule:** The input `n` to `QuadrangulationEnumerator` is the **number of vertices in Q\* (Dual)**. Internally, `n + 2` (the primal vertex count) is passed to plantri.

**Input Constraint:** The bundled count and materialization paths support `3 <= n <= 62`. The `SIMPLE_QUARTIC` family is empty for `n < 6`. The full literature `QUARTIC_MULTIGRAPH` family also contains the square's two-vertex dual, which lies outside this wrapper's supported range.

### Streaming enumeration

`iter_simple_quadrangulation_duals()` is the source-ordered primitive and does
not materialize the complete result tuple; its sequential path retains bounded
Python graph state. Close a partially consumed iterator explicitly.
`enumerate_simple_quadrangulation_duals()` is the bounded-workload collector and
materializes the same stream as a tuple; library calls do not write progress
messages to stdout.

`num_workers=1` is always sequential, an explicit value above one requests that
many processes, and `None` selects up to 16 usable CPUs. For a finite
`max_count`, the effective process count is capped by the number of task chunks.
`save_graphs_to_cache()` accepts a one-pass iterable; an unsized iterable must
provide `graph_count` (and an empty iterable must also provide
`dual_vertex_count`). This can bound retained Python graph objects, but a process
pool may buffer in-flight results, the growing planar-code file may grow with the
producer, the v11 dense Graph-ID index remains `O(N)`, and priority-order
generation still requires the caller's global sort.

`QuarticPlaneMap` numbers the four clockwise darts at vertex `v` as
`4*v, ..., 4*v+3` and stores only the opposite-dart involution `twin` plus the
source-stream `graph_id`. The rotation is implicit, so dual adjacency, support
edges, multiplicities, face cycles, and the canonical simple primal are all
derived values. For the supported `n <= 62`, at most 248 darts are present and
every opposite-dart index fits in one byte.

### Adjacency List Order (Combinatorial Embedding)

The neighbor order in `QuarticPlaneMap.dual_embedding` and in streamed primal embeddings represents the **cyclic order** of edges at each vertex, given **clockwise (CW) as viewed from outside the sphere**. This cyclic ordering defines the **combinatorial embedding** of the plane graph.

Production enumeration asks the unmodified bundled plantri executable to write
headerless one-byte `planar_code` to a unique temporary binary file. Python
drains complete records while the child runs and removes the file on normal,
failed, and early-closed paths. This avoids platform text-mode translation
without changing the bundled C source. The simple primal rotation system is
dualized exactly while preserving the exterior-view-CW convention. Because a
primal quadrangulation with `N` vertices has `2N-4` edges, the production path
decodes fixed `5N-7` byte records; the public generic decoder remains available
for other headerless one-byte `planar_code` streams.

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

`Plantri()` resolves only the bundled executable belonging to the active
package installation. To use another build, pass its path explicitly as
`Plantri(executable=...)`; no executable is selected implicitly from `PATH`.

## Number of `QUARTIC_MULTIGRAPH` Plane Maps by n

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
