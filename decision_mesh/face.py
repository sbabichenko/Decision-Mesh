from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Polygon

from ._helpers import _r

if TYPE_CHECKING:
    from .edge import Edge
    from .mesh import DecisionMesh
    from .tree import TreeNode


class Face:
    """
    Faces should carry the bulk of information

    They should contain:
    -a mask of points in the face (change later to set of indices)
    -the edges attached to the face
    -the vertices attached to the face
    - the coordinates of each point in the face
    """
    _seq = 0

    def __init__(self, mesh: DecisionMesh, edge0: Edge, edge1: Edge, edge2: Edge,
                 mask, active: bool = True, path='', skip_coords=False):
        self._id = Face._seq; Face._seq += 1
        self.path = path
        self.mesh = mesh
        self.mask = mask
        self.node = None
        self.edges = [edge0, edge1, edge2]
        self.sub_divisions = [{'e': None, '+': None, '-': None},
                              {'e': None, '+': None, '-': None},
                              {'e': None, '+': None, '-': None},]

        vertex1 = self.edges[0].vertex0
        vertex2 = self.edges[0].vertex1
        vertex0 = self.edges[2].other_vertex(vertex1)
        if not vertex0:
            vertex2, vertex1 = vertex1, vertex2
            vertex0 = self.edges[2].other_vertex(vertex1)
        self.vertices = [vertex0, vertex1, vertex2]

        # Sufficient statistics for regression
        # S_ww[i][j] = sum of w_i * w_j over all points in face
        # S_wy[i]    = sum of w_i * y over all points in face
        # S_yy       = sum of y^2 over all points in face
        self.S_ww = np.zeros((3, 3), dtype=np.float64)
        self.S_wy = np.zeros(3, dtype=np.float64)
        self.S_yy = 0.0
        self.n_covered = 0
        self.coords_indices = []

        if not skip_coords:
            self.update_coords()

        self.active = False
        if active:
            self.activate()

    @property
    def sid(self) -> str:
        return f"F{self._id:03d}"

    def __repr__(self):
        vs = ",".join(getattr(v, "sid", "V?") for v in self.vertices)
        es = ",".join(getattr(e, "sid", "E?") for e in self.edges)
        npts = self.n_covered
        # show which edges have created sub-divisions
        split_flags = "".join('1' if sd['e'] is not None else '0' for sd in self.sub_divisions)
        return (f"<{self.sid} path='{self.path}' active={self.active} "
                f"verts=[{vs}] edges=[{es}] area={_r(self.area)} n={npts} "
                f"split={split_flags}>")

    __str__ = __repr__

    def addnode(self, node: TreeNode):
        self.node = node

    @property
    def refinement_edge(self) -> Edge:
        """The edge opposite the newest vertex -- the only edge this face may be split along."""
        return self.edges[0]

    def split(self, edge: Edge):
        if edge is not self.edges[0]:
            # Completion: must split along refinement edge first.
            self.edges[0].midpoint.activate()
            return

        sub_division = self.sub_divisions[0]
        if sub_division['e'] is None:
            raise ValueError("Edge has not been activated/split yet.")

        self.deactivate()
        self.node.split(edge, sub_division['+'], sub_division['-'])
        sub_division['+'].activate()
        sub_division['-'].activate()

    def deactivate(self):
        if not self.active:
            return
        self.active = False
        self.mesh.active_faces.discard(self)
        for edge in self.edges:
            edge.remove_face(self)

    def activate(self):
        if self.active:
            return
        self.active = True
        self.mesh.active_faces.add(self)
        for edge in self.edges:
            edge.activate()
            edge.add_face(self)

    @property
    def area(self):
        #Shoelace Formula
        det = (self.vertices[0].x - self.vertices[2].x)*(self.vertices[1].y - self.vertices[0].y) - (self.vertices[0].x-self.vertices[1].x) * (self.vertices[2].y - self.vertices[0].y)
        return 0.5 * abs(det)

    def aspect_ratio(self):
        return max(edge.length for edge in self.edges)/min(edge.length for edge in self.edges)

    def update_coords(self, eps=1e-14):
        """
        Compute barycentric weights and accumulate sufficient statistics
        for all points where this face's mask is True.

        Stores coords_indices (list of int point indices), and
        S_ww, S_wy, S_yy sufficient statistics for regression.
        """
        self.coords_indices = []
        self.S_ww[:] = 0.0
        self.S_wy[:] = 0.0
        self.S_yy = 0.0
        self.n_covered = 0

        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]

        # Right-triangle optimization: v0 is opposite hypotenuse
        # Legs a = v1-v0, b = v2-v0
        ax = v1.x - v0.x; ay = v1.y - v0.y
        bx = v2.x - v0.x; by = v2.y - v0.y

        d00 = ax * ax + ay * ay
        d11 = bx * bx + by * by

        if d00 < eps or d11 < eps:
            # Degenerate triangle — just collect indices
            X = self.mesh.X
            mask = self.mask
            for i in range(len(mask)):
                if mask[i]:
                    self.coords_indices.append(i)
            self.n_covered = len(self.coords_indices)
            return

        inv_d00 = 1.0 / d00
        inv_d11 = 1.0 / d11
        v0x = v0.x; v0y = v0.y

        X = self.mesh.X
        values = self.mesh.values
        mask = self.mask

        # Vectorized: get indices where mask is True
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return

        # Vectorized barycentric computation
        px = X[indices, 0] - v0x
        py = X[indices, 1] - v0y
        u = (px * ax + py * ay) * inv_d00
        v = (px * bx + py * by) * inv_d11
        w0 = 1.0 - u - v
        y = values[indices]

        self.coords_indices = indices.tolist()
        self.n_covered = len(indices)

        # Accumulate sufficient statistics vectorized
        W = np.column_stack([w0, u, v])  # (n, 3)
        self.S_ww = W.T @ W              # (3, 3)
        self.S_wy = W.T @ y              # (3,)
        self.S_yy = float(y @ y)

    def update_coords_from_indices(self, indices, eps=1e-14):
        """
        Fast path: compute sufficient statistics only for given point indices
        (subset of a parent face). Avoids scanning all N points.
        """
        self.coords_indices = []
        self.S_ww[:] = 0.0
        self.S_wy[:] = 0.0
        self.S_yy = 0.0
        self.n_covered = 0

        if not indices:
            return

        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]

        ax = v1.x - v0.x; ay = v1.y - v0.y
        bx = v2.x - v0.x; by = v2.y - v0.y

        d00 = ax * ax + ay * ay
        d11 = bx * bx + by * by

        if d00 < eps or d11 < eps:
            self.coords_indices = list(indices)
            self.n_covered = len(indices)
            return

        inv_d00 = 1.0 / d00
        inv_d11 = 1.0 / d11
        v0x = v0.x; v0y = v0.y

        X = self.mesh.X
        values = self.mesh.values

        idx_arr = np.asarray(indices, dtype=np.intp)
        px = X[idx_arr, 0] - v0x
        py = X[idx_arr, 1] - v0y
        u = (px * ax + py * ay) * inv_d00
        v = (px * bx + py * by) * inv_d11
        w0 = 1.0 - u - v
        y = values[idx_arr]

        self.coords_indices = idx_arr.tolist()
        self.n_covered = len(idx_arr)

        W = np.column_stack([w0, u, v])
        self.S_ww = W.T @ W
        self.S_wy = W.T @ y
        self.S_yy = float(y @ y)

    @property
    def coords(self):
        """Backward-compatible property: lazily build a DataFrame of barycentric weights."""
        import pandas as pd
        if self.n_covered == 0:
            return pd.DataFrame(columns=list(self.vertices))

        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]

        ax = v1.x - v0.x; ay = v1.y - v0.y
        bx = v2.x - v0.x; by = v2.y - v0.y
        d00 = ax * ax + ay * ay
        d11 = bx * bx + by * by

        idx_arr = np.array(self.coords_indices, dtype=np.intp)
        X = self.mesh.X

        if d00 < 1e-14 or d11 < 1e-14:
            n = len(idx_arr)
            W = np.full((n, 3), np.nan)
        else:
            px = X[idx_arr, 0] - v0.x
            py = X[idx_arr, 1] - v0.y
            u = (px * ax + py * ay) / d00
            v = (px * bx + py * by) / d11
            w0 = 1.0 - u - v
            W = np.column_stack([w0, u, v])

        rows = self.mesh.index[idx_arr]
        return pd.DataFrame(W, index=rows, columns=list(self.vertices))

    def add_sub_division(self, edge: Edge, sub_division: dict):
        idx = self.edges.index(edge)
        self.sub_divisions[idx] = sub_division

    def affine_height_model(self, eps: float = 1e-12):
        """
        Build the affine model z(x,y) = m*[x,y] + b that interpolates the
        current vertex heights of this triangular face.

        Returns
        -------
        m : np.ndarray shape (2,)
            Linear coefficients [a, b] so z = a*x + b*y + c.
        b : float
            Intercept c.
        f : callable
            Evaluator: f(xy) where xy is shape (2,) or (n,2). Returns float or np.ndarray.
        """
        v0, v1, v2 = self.vertices
        A = np.array([
            [v0.x, v0.y, 1.0],
            [v1.x, v1.y, 1.0],
            [v2.x, v2.y, 1.0],
        ], dtype=float)
        h = np.array([float(getattr(v0, "height", np.nan)),
                      float(getattr(v1, "height", np.nan)),
                      float(getattr(v2, "height", np.nan))], dtype=float)

        # Degenerate triangle or bad data -> NaNs
        detA = np.linalg.det(A)
        if not np.isfinite(detA) or abs(detA) < eps or not np.all(np.isfinite(h)):
            m = np.array([np.nan, np.nan], dtype=float)
            c = float("nan")
            def f_bad(X):
                X = np.asarray(X, dtype=float)
                if X.ndim == 1:
                    return float("nan")
                if X.ndim == 2 and X.shape[1] == 2:
                    return np.full((X.shape[0],), np.nan, dtype=float)
                raise ValueError("X must be shape (2,) or (n,2)")
            return m, c, f_bad

        a_, b_, c = np.linalg.solve(A, h)  # z = a_*x + b_*y + c
        m = np.array([a_, b_], dtype=float)
        c = float(c)

        def f(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                if X.shape[0] != 2:
                    raise ValueError("X must be length-2 when 1D")
                return float(X @ m + c)
            if X.ndim == 2 and X.shape[1] == 2:
                return (X @ m) + c
            raise ValueError("X must be shape (2,) or (n,2)")

        return m, c, f

    def as_patch(self, **kwargs):
        """Return a matplotlib.patches.Polygon for this face."""
        verts = [(v.x, v.y) for v in self.vertices]
        return Polygon(verts, closed=True, **kwargs)
