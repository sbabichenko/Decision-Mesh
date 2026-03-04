from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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
                 mask, active: bool = True, path=''):
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
        try:
            npts = int(self.mask.sum())
        except Exception:
            npts = "?"
        # show which edges have created sub-divisions
        split_flags = "".join('1' if sd['e'] is not None else '0' for sd in self.sub_divisions)
        return (f"<{self.sid} path='{self.path}' active={self.active} "
                f"verts=[{vs}] edges=[{es}] area={_r(self.area)} n={npts} "
                f"split={split_flags}>")

    __str__ = __repr__

    def addnode(self, node: TreeNode):
        self.node = node

    def split(self, edge: Edge):
        idx = self.edges.index(edge)
        sub_division = self.sub_divisions[idx]
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
        return 0.5 * np.abs(det)

    def aspect_ratio(self):
        return max(edge.length for edge in self.edges)/min(edge.length for edge in self.edges)

    def update_coords(self, eps=1e-14):
        """
        Compute barycentric weights for the rows where this face's mask is True.
        Stores a DataFrame in self.coords with:
        rows  = dataset index restricted to face.mask
        cols  = [v0, v1, v2] (the actual Vertex objects)
        values = [w0, w1, w2]
        """
        rows = self.mesh.index[self.mask]  # pandas Index aligned to dataset
        X = self.mesh.X[self.mask]         # (n,2) points in this face

        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[2] - self.vertices[0]
        p = X - self.vertices[0]

        d00 = np.dot(a, a)
        d01 = np.dot(a, b)
        d11 = np.dot(b, b)
        d20 = p @ a
        d21 = p @ b

        denom = d00 * d11 - d01 * d01
        if abs(denom) < eps:
            n = len(X)
            W = np.column_stack([np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)])
        else:
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            w0 = 1.0 - u - v
            W = np.column_stack([w0, u, v])

        # DataFrame with Vertex-object columns
        self.coords = pd.DataFrame(W, index=rows, columns=list(self.vertices))

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
