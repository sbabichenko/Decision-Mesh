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

        # unit normal as numpy array
        n = np.array([-dy, dx], dtype=float)
        self.length = np.linalg.norm(n)
        n /= self.length
        self.normal = n

        # signed line equation: normal * p = intercept
        self.intercept = float(self.normal @ np.array([vertex0.x, vertex0.y], dtype=float))
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
        if self.faces['+']:
            self.faces['+'].split(self)
        if self.faces['-']:
            self.faces['-'].split(self)

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
        return np.array(v) @ self.normal - self.intercept

    def other_vertex(self, v: Vertex) -> Vertex | None:
        if v is self.vertex0: return self.vertex1
        if v is self.vertex1: return self.vertex0
        return None

    def add_face(self, face: Face):
        """
        Attach a face to this edge and precompute its subdivision.

        - Decide whether face is '+' or '-' side (by testing opposing vertex).
        - Store face + opposing vertex in self.faces / self.opposing_vertices.
        - Build internal chord (midpoint -> opposing_vertex) in self.sub_edges.
        - Create the two child faces (mask split + path update).
        - Register subdivision so face.split(edge) can later activate them.
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

        mask = self.mesh.X @ self.sub_edges[face_type].normal >= self.sub_edges[face_type].intercept

        if face_type == '+':
            mask0 = mask & face.mask
            mask1 = ~mask & face.mask

            path0 = face.path + '+'
            path1 = face.path + '-'
        else:
            mask0 = ~mask & face.mask
            mask1 = mask & face.mask

            path0 = face.path + '-'
            path1 = face.path + '+'

        face0 = Face(self.mesh, self.sub_edges[face_type], edge0, self.sub_edges['0'], mask0, False, path0)
        face1 = Face(self.mesh, self.sub_edges[face_type], edge1, self.sub_edges['1'], mask1, False, path1)

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
