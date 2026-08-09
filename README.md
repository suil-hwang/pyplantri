# pyplantri

A Python wrapper for [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) to enumerate **Simple Quadrangulations on a Sphere (SQS)**.

Given the dual vertex count `n`, it enumerates all **non-isomorphic duals of simple quadrangulations of the sphere** as compact quartic plane maps. Their primal and dual topology is derived exactly from the stored dart involution.

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

**Input Constraint:** The bundled count and materialization paths support `3 <= n <= 62`. The simple-quartic subclass is empty for `n < 6`.

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
dualized exactly while preserving the exterior-view-CW convention.

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

### Version 0.5 API change

Version 0.5 removes the former text codec and multi-field builder API without
aliases. Production enumeration decodes the simple primal `planar_code` stream
and constructs `QuarticPlaneMap` directly through
`QuarticPlaneMap.from_primal_embedding()`. The persistent map state is only
`twin: bytes` and `graph_id`; embeddings, faces, multiplicities, and
primal-dual maps are derived.

`enumerate_simple_quadrangulation_duals()` is now the sole enumeration API:
`dual_class` selects the Graph-ID namespace, while `num_workers`, `chunk_size`,
and `start_method` select the execution policy without changing source order.
It returns an immutable
`PlantriEnumerationResult` containing a tuple of maps plus direct `startup_s`,
`post_startup_s`, and derived `total_s` fields. The former
`enumerate_simple_quadrangulation_duals_filtered()`,
`enumerate_simple_quadrangulation_duals_parallel()`,
`FilteredEnumerationResult`, and `EnumerationTiming` APIs and aliases are
removed.

## Output and Cache Safety

- `Plantri.iter_planar_code()` is the binary record boundary; generic line iteration remains restricted to line-oriented ASCII formats.
- Schema-v11 caches store only `QuarticPlaneMap` records in independent pickle chunks, with compact footer manifests and a dense Graph-ID index. Chunk offsets and index layout are derived rather than stored. `load_graph_catalog()` opens cheaply and verifies chunks on access; `audit_all_graphs()` performs an explicit full audit. Cache loading requires `trusted=True`, and older formats must be regenerated.
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
