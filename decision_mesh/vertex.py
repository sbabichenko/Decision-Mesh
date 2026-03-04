from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._helpers import _r

if TYPE_CHECKING:
    from .edge import Edge
    from .mesh import DecisionMesh


class Vertex:
    """
    A vertex should contain the following information:

    Its location
    Its parent edge, if such edge exists
    If it is active or not
    height
    -method
    -returns average of parent edge vertex heights if inactive
    -returns set height if active
    affected_vertices
    -method
    -which other vertices will have their regression values/loss change if this vertex gets activated (if inactive) or changes height
    -equal to neighbors of vertex if active and neighbors if split if inactive
    edges attached
    -empty set if inactive
    """
    _seq = 0

    def __init__(self, mesh: DecisionMesh, x: float, y: float, active=False, parent_edge: Edge = None, real_vertex=True):
        self._id = Vertex._seq; Vertex._seq += 1
        self.x = x
        self.y = y
        self.mesh = mesh

        if real_vertex:
            mesh.vertices.add(self)
        self.active = active
        self.disqualified = False
        self.to_update = set()
        self.parent_edge = parent_edge
        if parent_edge is not None:
            self.height = (parent_edge.vertex0.height + parent_edge.vertex1.height) / 2
        else:
            self.height = 0.0  # will be set later by regression
        self.edges: set[Edge] = set()
        self.neighbors: set[Vertex] = set()

    @property
    def sid(self) -> str:
        return f"V{self._id:03d}"

    def __repr__(self):
        deg = len(self.edges)
        parent = getattr(self.parent_edge, "sid", None)
        return (f"<{self.sid} xy=({_r(self.x)},{_r(self.y)}) "
                f"h={_r(self.height)} active={self.active} deg={deg} "
                f"parent={parent}>")

    __str__ = __repr__

    # --- arithmetic operations
    def __add__(self, other):
        return Vertex(self.mesh, self.x + other.x, self.y + other.y, real_vertex=False)

    def __sub__(self, other):
        return Vertex(self.mesh, self.x - other.x, self.y - other.y, real_vertex=False)

    def __mul__(self, s: float):
        return Vertex(self.mesh, self.x * s, self.y * s, real_vertex=False)

    def __truediv__(self, s: float):
        return Vertex(self.mesh, self.x / s, self.y / s, real_vertex=False)

    def __iter__(self):
        yield self.x
        yield self.y

    # Let numpy convert it automatically
    def __array__(self, dtype=None):
        return np.array([self.x, self.y], dtype=dtype)

    def __getitem__(self, i):
        if i == 0: return self.x
        elif i == 1: return self.y
        else: raise IndexError("Vertex only has 2 coordinates")

    def __hash__(self):  # identity-based hashing
        return id(self)

    def __matmul__(self, other):
        return float(np.array([self.x, self.y]) @ np.asarray(other))

    def norm(self):
        return math.hypot(self.x, self.y)

    def add_edge(self, edge):
        self.edges.add(edge)
        other = edge.other_vertex(self)
        self.neighbors.add(other)

    def remove_edge(self, edge):
        self.edges.remove(edge)
        other = edge.other_vertex(self)
        if other in self.neighbors:
            self.neighbors.remove(other)

    def activate(self):
        """
        sets active to true                                            1
        updates height to locally optimal value                        2
        tells the parent edge to split                                 3
            parent edge tells adjacent faces to split using that edge  4
                faces tell edges to remove the face                    5
                faces tell subfaces to activate                        6
                    subfaces tell edges to create subsubfaces for it   7
                    subfaces tell their edges to activate if inactive  8
                        edges tell adjacent vertices to add the edge   9
                        edges create a midpoint                       10
                        midpoints update info                         11
            parent edge removes itself from adjacent vertices         12
            parent edge tells subedges to activate                    13
                edges tell adjacent vertices to add the edge          14
                edges create a midpoint                               15
                    midpoint runs loc_regress                         16
        tells affected_vertices to run loc_regress                    17
        sets loss_reduction to 0                                      18

        checks:
        1: if self.active: return  #do nothing if midpoint is already active
        2: check that locally optimal value exists, otherwise raise error
        3: check if parent_edge exists, otherwise raise error
        4: check that at least one adjacent face exists, otherwise raise error
        5: check if face is known to edge, otherwise raise error
        6: check if subface exists, otherwise raise error
       12: check if edge is known to vertices, otherwise raise error
       13: check if two such subedges exist, otherwise raise error
        """
        if self.active:
            return
        self.active = True
        self.height = self.new_height
        self.parent_edge.split()
        for v in self.affected_vertices:
            v.update_info()
        self.loss_reduction = 0
        self.mesh.loss_heap[self] = 0

    def update_info(self):
        self.new_height, self.loss_reduction, self.affected_vertices, self.affected_points = self.loc_regress()
        if not self.disqualified:
            self.mesh.loss_heap[self] = -self.loss_reduction

    def update_height(self):
        self.height = self.new_height
        for v in self.affected_vertices:
            v.update_info()
        self.loss_reduction = 0
        self.mesh.loss_heap[self] = 0

    def get_extended_faces(self):
        # neighborhoods
        neighborhood = set(self.neighbors); neighborhood.add(self)

        # faces touching self or its neighbors (assume non-empty)
        def faces_of(v):
            out = set()
            for e in v.edges:
                for s in ('+', '-'):
                    f = e.faces.get(s)
                    if f is not None and v in f.vertices:
                        out.add(f)
            return out
        faces = set().union(*(faces_of(v) for v in neighborhood))

        # column order: [neighborhood | distance-2]
        extended = set().union(*(set(f.vertices) for f in faces))
        return neighborhood, extended - neighborhood, faces

    def get_faces(self):
        faces = set()
        for e in self.edges:
            for s in ('+', '-'):
                f = e.faces.get(s)
                if f is not None and self in f.vertices:
                    faces.add(f)
        neighbors = {v for f in faces for v in f.vertices if v is not self}
        return neighbors, faces

    def build_design_matrix(self, faces):
        mask_union = pd.Series(False, index=self.mesh.index, dtype=bool)
        for f in faces:
            mask_union |= f.mask
        rows = mask_union[mask_union].index

        cols = set().union(*(set(f.vertices) for f in faces))
        col_key = {v: v._id for v in cols}
        X = pd.DataFrame(0.0, index=rows, columns=[col_key[v] for v in cols])

        # Use the mapped keys when writing/reading
        for f in faces:
            W = f.coords.copy()
            W.columns = [col_key[v] for v in f.vertices]  # map Vertex->stable scalar
            X.loc[W.index, W.columns] = W.values

        return X, rows, col_key

    def loc_regress(self):
        """
        Fit this vertex's optimal height given neighbors fixed.

        - Collect faces touching this vertex (or simulated if inactive).
        - Build design matrix from barycentric weights.
        - Subtract fixed contribution of neighbors.
        - Solve 1D least-squares for this vertex's best height.
        - Return (beta_opt, loss_reduction, neighbors, n_points).
        """
        # 1) which faces / neighbors?
        if self.active:
            neighbors, faces = self.get_faces()
        else:
            neighbors, faces = self.get_sim_faces()

        if not faces:
            return float(self.height), 0.0, neighbors, 0

        # 2) design matrix for those faces
        X, rows, col_key = self.build_design_matrix(faces)
        points_attached = len(X)
        y = pd.Series(self.mesh.values, index=self.mesh.index, dtype=float).loc[rows].to_numpy()

        # 3) fixed contribution from neighbors ("correction")
        other_cols = [col_key[v] for v in neighbors if v in col_key]
        if other_cols:
            other_heights = np.array([v.height for v in neighbors], dtype=float)
            correction = X[other_cols].to_numpy() @ other_heights
        else:
            correction = np.zeros_like(y)

        # 4) column for self (the only regressor)
        x = X[col_key[self]].to_numpy().astype(float)

        # 5) residual after removing neighbors
        r = y - correction

        # 6) original loss with current self.height
        beta_orig = float(self.height)
        orig_loss = float(((r - x * beta_orig) ** 2).sum())

        # 7) optimal 1D least-squares for beta (self only)
        xTx = float(x @ x)
        if xTx > 0.0:
            xTr = float(x @ r)
            beta_opt = xTr / xTx
            # SSE_post = ||r - x*beta_opt||^2 = r^Tr - (x^Tr)^2 / (x^Tx)
            sse_post = float((r @ r) - (xTr * xTr) / xTx)
        else:
            # self's column has no support; nothing to fit
            beta_opt = beta_orig
            sse_post = float(r @ r)

        loss_reduction = orig_loss - sse_post

        return float(beta_opt), float(loss_reduction), neighbors, points_attached

    @property
    def degree(self) -> int:
        return len(self.edges)

    def neighbors_after_split(self):
        if not self.parent_edge:
            return
        parent_edge = self.parent_edge
        neighbors = set()
        neighbors.add(self)
        neighbors.add(parent_edge.vertex0)
        neighbors.add(parent_edge.vertex1)
        if parent_edge.opposing_vertices['+']:
            neighbors.add(parent_edge.opposing_vertices['+'])
        if parent_edge.opposing_vertices['-']:
            neighbors.add(parent_edge.opposing_vertices['-'])
        return neighbors

    def get_simulated_faces(self):
        if not self.parent_edge:
            return
        parent_edge = self.parent_edge
        face_plus = parent_edge.faces.get('+')
        face_minus = parent_edge.faces.get('-')
        def faces_of(v):
            out = set()
            for e in v.edges:
                for s in ('+', '-'):
                    f = e.faces.get(s)
                    if f is not None and v in f.vertices:
                        out.add(f)
            return out

        neighborhood = self.neighbors_after_split()

        #get all faces touching self or its neighbors
        faces = set().union(*(faces_of(v) for v in neighborhood))
        if face_plus is not None:
            idx = face_plus.edges.index(parent_edge)
            faces.add(face_plus.sub_divisions[idx].get('+'))
            faces.add(face_plus.sub_divisions[idx].get('-'))
        if face_minus is not None:
            idx = face_minus.edges.index(parent_edge)
            faces.add(face_minus.sub_divisions[idx].get('+'))
            faces.add(face_minus.sub_divisions[idx].get('-'))
        faces.discard(self.parent_edge.faces.get('+'))
        faces.discard(self.parent_edge.faces.get('-'))

        extended = set().union(*(set(f.vertices) for f in faces))
        return neighborhood, extended - neighborhood, faces

    def get_sim_faces(self):
        if not self.parent_edge:
            return

        faces = set()
        parent_edge = self.parent_edge
        face_plus = parent_edge.faces.get('+')
        face_minus = parent_edge.faces.get('-')
        if face_plus is not None:
            idx = face_plus.edges.index(parent_edge)
            faces.add(face_plus.sub_divisions[idx].get('+'))
            faces.add(face_plus.sub_divisions[idx].get('-'))
        if face_minus is not None:
            idx = face_minus.edges.index(parent_edge)
            faces.add(face_minus.sub_divisions[idx].get('+'))
            faces.add(face_minus.sub_divisions[idx].get('-'))

        neighbors = {v for f in faces for v in f.vertices if v is not self}
        return neighbors, faces
