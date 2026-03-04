# Decision Mesh

An adaptive triangular mesh for nonlinear function approximation. Instead of
the axis-aligned rectangular splits used by decision trees, Decision Mesh
partitions 2D space into triangles and fits a piecewise-linear surface via
local regression on barycentric coordinates.

## How It Works

1. **Initialization** -- The bounding box of the data is split into two
   triangles along the diagonal. Each triangle is a _face_ in the mesh and a
   leaf in an underlying binary decision tree.

2. **Vertex activation** -- Every edge maintains an inactive _midpoint_ vertex.
   A local least-squares regression estimates the optimal height for each
   midpoint and the squared-error reduction that activating it would achieve.
   These candidates are stored in a priority queue.

3. **Greedy refinement** -- On each step the algorithm either:
   - **Exploits**: picks the midpoint with the largest loss reduction from the
     priority queue, or
   - **Explores** (with probability `random`): samples a face weighted by
     `area * #points` and picks the midpoint of its longest edge.

   Activating a midpoint splits its parent edge, which in turn splits the two
   adjacent faces into four new triangles. Heights and loss reductions are
   recomputed for all affected vertices.

4. **Aspect-ratio guard** -- Splits that would create triangles with an aspect
   ratio above `max_aspect_ratio` (default 5) are disqualified to prevent
   degenerate slivers.

5. **Prediction** -- Within each triangular face, the target value is
   interpolated as a linear combination of the three vertex heights weighted by
   barycentric coordinates, giving a continuous piecewise-affine surface.

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

## Installation

```bash
pip install numpy pandas matplotlib heapdict
```

## Quick Start

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

# Refine the mesh for 500 iterations
for _ in range(500):
    mesh.update_best_vertex(random=0.1)

# Visualize the fitted surface
mesh.plot_height(cmap="RdBu_r", vmax_abs=2, draw_edges=True)
```

## API

### `DecisionMesh(df)`

Create a mesh from a DataFrame where the first two columns are the 2D
coordinates and the third column is the target value.

| Attribute / Method | Description |
|-|-|
| `update_best_vertex(random=0)` | Perform one refinement step. `random` controls the exploration probability (0 = pure greedy, 1 = pure random). |
| `find_best_vertex(random=0.05)` | Return `(vertex, loss_reduction)` for the best candidate without activating it. |
| `random_face(rng=None)` | Sample a face with probability proportional to `area * #points`. |
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
| `Face` | Triangle defined by three vertices and three edges. Stores the data-point mask and barycentric coordinates. |

## Project Structure

```
decision_mesh/
    __init__.py      # Public API exports
    _helpers.py      # Repr formatting utilities
    tree.py          # TreeNode
    vertex.py        # Vertex
    edge.py          # Edge
    face.py          # Face
    mesh.py          # DecisionMesh
Mesh Tester.ipynb    # Interactive examples and validation tests
```
