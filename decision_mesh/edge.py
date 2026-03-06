from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._helpers import _r, _rn
from .face import Face
from .vertex import Vertex

if TYPE_CHECKING:
    from .mesh import DecisionMesh


class Edge:
    """
    Edges should have the following information:

    vertices at each end
    a test (test_vertex) to check which side of the edge a point lies on
    the faces attached to the edge (+ and/or -)
    activity of the edge
    """
    _seq = 0

    def __init__(self, mesh: DecisionMesh, vertex0: Vertex, vertex1: Vertex, active: bool):
        self._id = Edge._seq; Edge._seq += 1
        self.vertex0 = vertex0
        self.vertex1 = vertex1
        self.mesh = mesh
        self.faces = dict({'+': None, '-': None})
        self.disqualifying = set()
        self.opposing_vertices = dict({'+': None, '-': None})

        dx = vertex1.x - vertex0.x
        dy = vertex1.y - vertex0.y

        # unit normal — stored as numpy array for bulk operations, and as scalars for hot paths
        n0 = -dy
        n1 = dx
        self.length = (n0 * n0 + n1 * n1) ** 0.5
        inv_len = 1.0 / self.length if self.length > 0 else 0.0
        n0 *= inv_len
        n1 *= inv_len
        self.normal = np.array([n0, n1], dtype=float)
        self._n0 = n0
        self._n1 = n1

        # signed line equation: normal * p = intercept
        self.intercept = n0 * vertex0.x + n1 * vertex0.y
        self.midpoint = None
        self.sub_edges = {'0': None, '1': None, '+': None, '-': None}
        self.active = False
        if active:
            self.activate()

    @property
    def sid(self) -> str:
        return f"E{self._id:03d}"

    def __repr__(self):
        v0 = getattr(self.vertex0, "sid", "V?")
        v1 = getattr(self.vertex1, "sid", "V?")
        # show which faces are attached by path (not full repr to avoid recursion)
        fp = self.faces.get('+'); fm = self.faces.get('-')
        fp_tag = f"{getattr(fp, 'sid', None)}:{getattr(fp, 'path', None)}" if fp else None
        fm_tag = f"{getattr(fm, 'sid', None)}:{getattr(fm, 'path', None)}" if fm else None
        has_mid = hasattr(self, "midpoint")
        subs = getattr(self, "sub_edges", None)
        sub_ready = [k for k in ('0', '1', '+', '-') if isinstance(subs, dict) and subs.get(k) is not None]

        return (f"<{self.sid} {v0}->{v1} active={self.active} "
                f"n={_rn(self.normal)} b={_r(self.intercept)} "
                f"faces(+={fp_tag},-={fm_tag}) mid={has_mid} sub={sub_ready}>")

    __str__ = __repr__

    def activate(self):
        if self.active:
            return
        self.active = True
        self.mesh.active_edges.add(self)
        self.add_midpoint()
        self.vertex0.add_edge(self)
        self.vertex1.add_edge(self)
        self.sub_edges = {'0': Edge(self.mesh, self.vertex0, self.midpoint, False), '1': Edge(self.mesh, self.midpoint, self.vertex1, False), '+': None, '-': None}

    def add_midpoint(self):
        if self.midpoint is not None:
            return
        self.midpoint = Vertex(self.mesh, (self.vertex0.x + self.vertex1.x)/2, (self.vertex0.y + self.vertex1.y)/2, parent_edge=self)

    def split(self):
        for sign in ('+', '-'):
            while self.faces.get(sign) is not None:
                self.faces[sign].split(self)

        self.sub_edges['0'].activate()
        self.sub_edges['1'].activate()
        self.deactivate()

    def deactivate(self):
        if not self.active:
            return
        self.active = False
        self.mesh.active_edges.discard(self)
        self.vertex0.remove_edge(self)
        self.vertex1.remove_edge(self)

    def test_vertex(self, v: Vertex) -> float:
        return self._n0 * v.x + self._n1 * v.y - self.intercept

    def other_vertex(self, v: Vertex) -> Vertex | None:
        if v is self.vertex0: return self.vertex1
        if v is self.vertex1: return self.vertex0
        return None

    def add_face(self, face: Face):
        """
        Attach a face to this edge and precompute its subdivision.

        Uses index-based point splitting: iterates only the parent face's
        coords_indices (not all N data points) and routes each point to
        the correct child face via the chord's half-plane test.
        """
        idx = face.edges.index(self)
        opposing_vertex = face.vertices[idx]
        if self.test_vertex(opposing_vertex) > 0:
            face_type = '+'
        else:
            face_type = '-'

        self.faces[face_type] = face
        self.opposing_vertices[face_type] = opposing_vertex

        self.sub_edges[face_type] = Edge(self.mesh, self.midpoint, opposing_vertex, False)
        idx = face.vertices.index(self.vertex1)
        edge0 = face.edges[idx]
        idx = face.vertices.index(self.vertex0)
        edge1 = face.edges[idx]

        # Split parent face's points by chord half-plane
        # Iterate only parent's coords_indices — O(points_in_parent) not O(N_total)
        chord = self.sub_edges[face_type]
        cn0 = chord._n0
        cn1 = chord._n1
        cint = chord.intercept

        X = self.mesh.X
        indices0 = []
        indices1 = []

        for i in face.coords_indices:
            val = cn0 * X[i, 0] + cn1 * X[i, 1]
            on_plus = val >= cint
            if face_type == '+':
                if on_plus:
                    indices0.append(i)
                else:
                    indices1.append(i)
            else:
                if not on_plus:
                    indices0.append(i)
                else:
                    indices1.append(i)

        if face_type == '+':
            path0 = face.path + '+'
            path1 = face.path + '-'
        else:
            path0 = face.path + '-'
            path1 = face.path + '+'

        # Create child faces with skip_coords=True, then use fast index path
        face0 = Face(self.mesh, edge0, self.sub_edges['0'], self.sub_edges[face_type], None, False, path0, skip_coords=True)
        face0.update_coords_from_indices(indices0)

        face1 = Face(self.mesh, edge1, self.sub_edges[face_type], self.sub_edges['1'], None, False, path1, skip_coords=True)
        face1.update_coords_from_indices(indices1)

        if face_type == '+':
            face.add_sub_division(self, {'e': self.sub_edges[face_type], '+': face0, '-': face1})
        else:
            face.add_sub_division(self, {'e': self.sub_edges[face_type], '+': face1, '-': face0})

        if max(face0.aspect_ratio(), face1.aspect_ratio()) >= self.mesh.max_aspect_ratio:
            self.disqualifying.add(face)
            if not self.midpoint.disqualified:
                self.midpoint.disqualified = True
                self.mesh.loss_heap.pop(self.midpoint, None)

        self.midpoint.update_info()

    def remove_face(self, face):
        if self.faces['+'] is face:
            self.faces['+'] = None
        if self.faces['-'] is face:
            self.faces['-'] = None

        self.disqualifying.discard(face)
        if len(self.disqualifying) == 0:
            self.midpoint.disqualified = False
