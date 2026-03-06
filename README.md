# Decision Mesh

An adaptive triangular mesh for nonlinear function approximation. Instead of
the axis-aligned rectangular splits used by decision trees, Decision Mesh
partitions 2D space into triangles and fits a piecewise-linear surface via
local regression on barycentric coordinates.

## How It Works

1. **Initialization** — The bounding box of the data is split into two right
   triangles along the diagonal. Each triangle is a _face_ in the mesh and a
   leaf in an underlying binary decision tree.

2. **Vertex activation** — Every edge maintains an inactive _midpoint_ vertex.
   A local least-squares regression estimates the optimal height for each
   midpoint and the squared-error reduction that activating it would achieve.
   These candidates are stored in a priority queue.

3. **Greedy refinement** — On each step the algorithm picks the midpoint with
   the largest loss reduction from the priority queue. Activating a midpoint
   splits its parent edge, which in turn splits the adjacent faces into new
   triangles. Heights and loss reductions are recomputed for all affected
   vertices.

4. **Newest vertex bisection** — Faces are only split along their _refinement
   edge_ (the edge opposite the newest vertex). If the greedy algorithm wants
   to split an edge that isn't a face's refinement edge, a **completion
   cascade** first splits the face along its own refinement edge recursively
   until the target edge becomes a refinement edge. This guarantees all
   triangles remain right triangles.

5. **Aspect-ratio guard** — Splits that would create triangles with an aspect
   ratio above `max_aspect_ratio` (default 5) are disqualified to prevent
   degenerate slivers.

6. **Prediction** — Within each triangular face, the target value is
   interpolated as a linear combination of the three vertex heights weighted by
   barycentric coordinates, giving a continuous piecewise-affine surface. The
   underlying binary decision tree enables O(log n) point lookup.

```
Before refinement          After 50 steps            After 500 steps
+-------------+          +------+------+          +--+--+--+--+--+--+
|           / |          |    /|    / |          |\/|\/|\/|\/|\/|\/|
|         /   |          |  /  | /   |          |/\|/\|/\|/\|/\|/\|
|       /     |   -->    |/    |/    |   -->    +--+--+--+--+--+--+
|     /       |          +-----+-----+          |\/|\/|\/|\/|\/|\/|
|   /         |          |   / |   / |          |/\|/\|/\|/\|/\|/\|
| /           |          | /   | /   |          +--+--+--+--+--+--+
+-------------+          +-----+-----+          (adaptive, not uniform)
```

## Algorithm Details

### Right-Triangle Barycentric Coordinates

Every face is a right triangle with the right-angle vertex at `vertices[0]`
(opposite the hypotenuse `edges[0]`). The two legs `a = v1 - v0` and
`b = v2 - v0` are perpendicular (`a · b = 0`), which simplifies barycentric
coordinate computation:

```
u = (p · a) / |a|²     (weight for v1)
v = (p · b) / |b|²     (weight for v2)
w = 1 - u - v           (weight for v0)
```

This eliminates the cross-term and determinant from the general formula.

### Sufficient Statistics

Instead of storing per-point barycentric weights, each face maintains aggregate
statistics computed once during creation:

```
S_ww[i][j] = Σ  w_i * w_j    (3×3 symmetric matrix)
S_wy[i]    = Σ  w_i * y       (3-vector)
S_yy       = Σ  y²            (scalar)
n_covered  = point count
```

Local regression for a vertex reduces to accumulating these across its adjacent
faces — no iteration over individual data points:

```
xTx = Σ_faces S_ww[si][si]
xTr = Σ_faces (S_wy[si] - Σ_{j≠si} h_j * S_ww[si][j])
beta_opt = xTr / xTx
loss_reduction = (xTr² / xTx) - 2*h_current*xTr + h_current²*xTx
```

### Complexity

| Operation | Cost |
|-----------|------|
| Find best vertex (heap peek) | O(1) |
| Heap insert/update/remove | O(log n) |
| Vertex activation (split) | O(k), k = points in affected faces |
| Local regression | O(adjacent faces) ≈ O(1) |
| Prediction query | O(tree depth) = O(log num_faces) |

## Implementations

### Python

The original implementation in pure Python/NumPy.

```bash
pip install numpy pandas matplotlib heapdict
```

```python
import numpy as np
import pandas as pd
from decision_mesh import DecisionMesh

# Generate sample data: z = cos(x) * cos(y) + noise
rng = np.random.default_rng(42)
n = 10_000
x0 = rng.uniform(-4, 4, size=n)
x1 = rng.uniform(-4, 4, size=n)
z = 2 * np.cos(5 * x0) * np.cos(2 * x1) + rng.standard_normal(n)

df = pd.DataFrame({0: x0, 1: x1, 2: z})
mesh = DecisionMesh(df)

for _ in range(500):
    mesh.update_best_vertex(random=0.1)

mesh.plot_height(cmap="RdBu_r", vmax_abs=2, draw_edges=True)
```

### C++

A high-performance C++ implementation with the same algorithm. Achieves
~142,000 refinement steps in 90 seconds on 200k data points.

```bash
cd cpp
mkdir build && cd build
cmake .. && make
./decision_mesh -n 200000 -t 90 -o mesh_output.svg
```

| Flag | Description |
|------|-------------|
| `-n` | Number of generated data points (default: 200000) |
| `-t` | Time limit for refinement in seconds (default: 90) |
| `-i` | Input CSV file (x,y,z columns) |
| `-o` | Output SVG file (default: mesh_output.svg) |

The C++ build produces a timing CSV alongside the SVG output. Use
`plot_timing.py` to visualize the per-refinement timing breakdown:

```bash
python plot_timing.py mesh_output_timing.csv
```

## Python API

### `DecisionMesh(df)`

Create a mesh from a DataFrame where the first two columns are the 2D
coordinates and the third column is the target value.

| Attribute / Method | Description |
|-|-|
| `update_best_vertex(random=0)` | Perform one refinement step. `random` controls the exploration probability (0 = pure greedy). |
| `find_best_vertex(random=0.05)` | Return `(vertex, loss_reduction)` for the best candidate without activating it. |
| `plot_height(cmap, draw_edges, draw_vertices, vmax_abs)` | Visualize the mesh surface with Gouraud shading. |
| `max_aspect_ratio` | Maximum allowed triangle aspect ratio (default 5). |
| `active_faces` | Set of currently active `Face` objects. |
| `vertices` | Set of all `Vertex` objects in the mesh. |
| `root` | Root `TreeNode` of the underlying decision tree. |

### Supporting Classes

| Class | Role |
|-|-|
| `TreeNode` | Binary decision tree node. Splits partition the mesh; leaves correspond to active faces. |
| `Vertex` | 2D point with a fitted height. Handles local regression and activation. |
| `Edge` | Line segment connecting two vertices. Manages midpoints, face attachments, and subdivision. |
| `Face` | Triangle defined by three vertices and three edges. Stores covered point indices and sufficient statistics. |

## Project Structure

```
decision_mesh/          # Python implementation
    __init__.py
    _helpers.py         # Repr formatting utilities
    tree.py             # TreeNode
    vertex.py           # Vertex
    edge.py             # Edge
    face.py             # Face
    mesh.py             # DecisionMesh

cpp/                    # C++ implementation
    CMakeLists.txt
    main.cpp            # CLI entry point
    mesh.h / mesh.cpp   # DecisionMesh, heap, factory methods
    vertex.h / vertex.cpp  # Vertex, local regression
    edge.h / edge.cpp   # Edge, splitting, child face creation
    face.h / face.cpp   # Face, barycentric coords, sufficient stats
    tree.h / tree.cpp   # TreeNode (binary decision tree)
    plot_timing.py      # Timing breakdown visualization

Mesh Tester.ipynb       # Interactive examples and validation tests
```
