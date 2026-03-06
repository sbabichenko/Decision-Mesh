from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

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

        # Empirical Bayes partial pooling fields
        if parent_edge is not None:
            self.depth = max(parent_edge.vertex0.depth, parent_edge.vertex1.depth) + 1
        else:
            self.depth = 0
        self.prior_mean = 0.0
        self.lambda_v = 0.0
        self.sigma_sq = float('inf')
        self.delta_pooled = 0.0
        self.sigma_pooled = float('inf')
        self.prior_children: set[Vertex] = set()

        # Register with parent edge endpoints for prior propagation
        if real_vertex and parent_edge is not None:
            parent_edge.vertex0.prior_children.add(self)
            parent_edge.vertex1.prior_children.add(self)

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

        # Set EB posterior fields before split (new midpoints need these during split)
        if self.parent_edge is not None:
            self.delta_pooled = self.height - self._mu_lin()
            inv_sigma = (1.0 / self.sigma_sq if self.sigma_sq < float('inf') else 0.0)
            xTx_reg = inv_sigma + self.lambda_v
            self.sigma_pooled = 1.0 / xTx_reg if xTx_reg > 0 else float('inf')

        self.parent_edge.split()

        affected = set(self.affected_vertices)
        affected.update(self.prior_children)
        for v in affected:
            v.update_info()
        self.loss_reduction = 0
        self.mesh.loss_heap[self] = 0

    def update_info(self):
        beta_data, loss_red_data, neighbors, n_points, xTx, xTr, rTr = self.loc_regress()
        self.affected_vertices = neighbors
        self.affected_points = n_points
        self.sigma_sq = 1.0 / xTx if xTx > 0 else float('inf')

        lambda_v, mu_v = self._compute_eb_params(xTx)
        self.lambda_v = lambda_v

        if lambda_v > 0 and xTx > 0:
            xTx_reg = xTx + lambda_v
            xTr_reg = xTr + lambda_v * mu_v
            beta_reg = xTr_reg / xTx_reg

            beta_orig = float(self.height)
            orig_loss = (beta_orig * beta_orig * xTx
                         - 2 * beta_orig * xTr + rTr
                         + lambda_v * (beta_orig - mu_v) ** 2)
            sse_post = rTr + lambda_v * mu_v * mu_v - xTr_reg * xTr_reg / xTx_reg

            self.new_height = beta_reg
            self.loss_reduction = max(0.0, orig_loss - sse_post)
            self.sigma_pooled = 1.0 / xTx_reg
            self.delta_pooled = beta_reg - self._mu_lin()
        else:
            self.new_height = beta_data
            self.loss_reduction = loss_red_data
            if self.parent_edge is not None and xTx > 0:
                self.delta_pooled = beta_data - self._mu_lin()
                self.sigma_pooled = self.sigma_sq

        if not self.disqualified:
            self.mesh.loss_heap[self] = -self.loss_reduction

    def update_height(self):
        self.height = self.new_height
        if self.parent_edge is not None:
            self.delta_pooled = self.height - self._mu_lin()
        affected = set(self.affected_vertices)
        affected.update(self.prior_children)
        for v in affected:
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

    def loc_regress(self):
        """
        Fit this vertex's optimal height given neighbors fixed.

        Uses per-face sufficient statistics (S_ww, S_wy, S_yy) to compute
        the 1D least-squares solution in O(faces) time, without iterating
        over individual data points.
        """
        # 1) which faces / neighbors?
        if self.active:
            neighbors, faces = self.get_faces()
        else:
            neighbors, faces = self.get_sim_faces()

        if not faces:
            return float(self.height), 0.0, neighbors, 0, 0.0, 0.0, 0.0

        # 2) Accumulate sufficient statistics across faces
        xTx = 0.0
        xTr = 0.0
        rTr = 0.0
        total_points = 0

        for f in faces:
            # Find which vertex index in this face is "self"
            si = -1
            for i in range(3):
                if f.vertices[i] is self:
                    si = i
                    break
            if si < 0 or f.n_covered == 0:
                continue

            total_points += f.n_covered

            # Gather neighbor heights for this face
            h = [f.vertices[0].height, f.vertices[1].height, f.vertices[2].height]

            xTx += f.S_ww[si, si]

            # xTr contribution
            xTr_face = f.S_wy[si]
            for j in range(3):
                if j == si:
                    continue
                xTr_face -= h[j] * f.S_ww[si, j]
            xTr += xTr_face

            # rTr contribution: ||y - sum_{j!=si} w_j h_j||^2
            rTr_face = f.S_yy
            for j in range(3):
                if j == si:
                    continue
                rTr_face -= 2.0 * h[j] * f.S_wy[j]
                for k in range(3):
                    if k == si:
                        continue
                    rTr_face += h[j] * h[k] * f.S_ww[j, k]
            rTr += rTr_face

        if total_points == 0:
            return float(self.height), 0.0, neighbors, 0, 0.0, 0.0, 0.0

        # Original loss with current height
        beta_orig = float(self.height)
        orig_loss = rTr - 2.0 * beta_orig * xTr + beta_orig * beta_orig * xTx

        # Optimal 1D least squares
        if xTx > 0.0:
            beta_opt = xTr / xTx
            sse_post = rTr - (xTr * xTr) / xTx
        else:
            xTr = 0.0
            beta_opt = beta_orig
            sse_post = rTr

        loss_reduction = orig_loss - sse_post

        return float(beta_opt), float(loss_reduction), neighbors, total_points, xTx, xTr, rTr

    # --- Empirical Bayes helpers ---

    def _mu_lin(self):
        if self.parent_edge is None:
            return 0.0
        return (self.parent_edge.vertex0.height + self.parent_edge.vertex1.height) / 2.0

    def _compute_overlap_factor(self, a, b):
        def faces_of(v):
            out = set()
            for e in v.edges:
                for s in ('+', '-'):
                    f = e.faces.get(s)
                    if f is not None and v in f.vertices:
                        out.add(f)
            return out

        faces_a = faces_of(a)
        faces_b = faces_of(b)
        shared = faces_a & faces_b

        n_shared = sum(f.n_covered for f in shared)
        n_a = sum(f.n_covered for f in faces_a)
        n_b = sum(f.n_covered for f in faces_b)

        denom = n_a + n_b - n_shared
        if denom <= 0:
            return 0.0
        return n_shared / denom

    def _compute_eb_params(self, xTx):
        if self.parent_edge is None or xTx <= 0:
            return 0.0, 0.0

        tau_sq_dict = self.mesh.tau_sq
        if not tau_sq_dict or self.depth not in tau_sq_dict:
            return 0.0, 0.0

        tau_sq = tau_sq_dict[self.depth]

        a = self.parent_edge.vertex0
        b = self.parent_edge.vertex1
        mu_lin = (a.height + b.height) / 2.0

        non_corner_parents = [v for v in (a, b) if v.parent_edge is not None]
        if non_corner_parents:
            delta_prior = sum(v.delta_pooled for v in non_corner_parents) / len(non_corner_parents)
        else:
            delta_prior = self.mesh.mu_delta.get(self.depth, 0.0)

        mu_v = mu_lin + delta_prior

        overlap_frac = self._compute_overlap_factor(a, b)
        s_v = 1.0 / max(1e-15, 1.0 - overlap_frac)

        parent_sigmas = [v.sigma_pooled for v in non_corner_parents
                         if v.sigma_pooled < float('inf')]
        if parent_sigmas:
            sigma_prior = tau_sq + s_v * sum(parent_sigmas) / len(parent_sigmas)
        else:
            sigma_prior = tau_sq

        if sigma_prior <= 0:
            lambda_v = 1e12
        else:
            lambda_v = 1.0 / sigma_prior

        self.prior_mean = mu_v
        return lambda_v, mu_v

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
