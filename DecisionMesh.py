import pandas as pd
import numpy as np
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class TreeNode:
    def __init__(self, mesh, parent: 'TreeNode' = None, face: 'Face' = None):
        self.parent = parent
        self.split_criterion = None
        self.face = face
        self.mesh = mesh
        if face:
            self.face.addnode(self)
    

    def is_leaf(self):
        return self.split_criterion is None
    
    def split(self, edge: 'Edge', plus: 'Face', minus: 'Face'):
        self.split_criterion = {'normal': edge.normal, 'intercept': edge.intercept}
        self.children = {'+': TreeNode(self.mesh, self, plus), '-': TreeNode(self.mesh, self, minus)}
        self.mesh.leaves.remove(self)
        self.mesh.leaves.add(self.children['+'])
        self.mesh.leaves.add(self.children['-'])
class DecisionMesh:
    
    
    def __init__(self, df : pd.DataFrame):
        self.X = df.iloc[:, :2].to_numpy()
        self.residuals = df.iloc[:,2].to_numpy()
        self.index = df.index
        self.root = TreeNode(self)
        self.leaves = set()
        self.leaves.add(self.root)
        self.active_faces = set()
        self.active_vertices = set()
        self.active_edges = set()
        
        
        #dict with vertices as keys and masks as ownership
        self.ownership = defaultdict(self._empty_mask)
        self.coefficients = defaultdict(self._empty_coeff)
        self.midpoints = set()
        
        self.create_outer_vertices()
        self.create_outer_edges()
        
        
        

        self.first_split()
    
    def _empty_mask(self) -> pd.Series:
        """Default factory for ownership: a fresh all-False mask aligned to data."""
        return pd.Series(False, index=self.index, dtype=bool)
    
    def _empty_coeff(self) -> pd.Series:
        """All-zero coefficients aligned to dataset index."""
        return pd.Series(0.0, index=self.index, dtype=float)
    
    def first_split(self):
        
        self.split_edge = Edge(self, self.outer_vertices['bottom_left'],self.outer_vertices['top_right'], True)
        self.top_face = Face(self, self.split_edge,self.outer_edges['left'],self.outer_edges['top'], path = '+')
        self.bottom_face = Face(self, self.split_edge, self.outer_edges['bottom'], self.outer_edges['right'], path = '-')
        self.root.split(self.split_edge, self.top_face, self.bottom_face)
        mask = (self.X @ self.split_edge.normal) >= self.split_edge.intercept
        self.top_face.update_mask(mask)
        self.bottom_face.update_mask(~mask)
        self.regress_and_assign()
        
    def create_outer_vertices(self):
        self.xmin = self.X[:,0].min()
        self.xmax = self.X[:,0].max()
        self.ymin = self.X[:,1].min()
        self.ymax = self.X[:,1].max()
        self.outer_vertices = dict()
        self.outer_vertices['bottom_left'] = Vertex(self, self.xmin,self.ymin, True)
        self.outer_vertices['top_left'] = Vertex(self, self.xmin,self.ymax, True)
        self.outer_vertices['bottom_right'] = Vertex(self, self.xmax,self.ymin, True)
        self.outer_vertices['top_right'] = Vertex(self, self.xmax,self.ymax, True)
        
        for vertex in self.outer_vertices.values():
            self.active_vertices.add(vertex)
            self.coefficients[vertex]

    def create_outer_edges(self):
        self.outer_edges = dict()
        self.outer_edges['left'] = Edge(self, self.outer_vertices['top_left'],self.outer_vertices['bottom_left'], True)
        self.outer_edges['bottom'] = Edge(self, self.outer_vertices['bottom_right'],self.outer_vertices['bottom_left'], True)
        self.outer_edges['right'] = Edge(self, self.outer_vertices['top_right'],self.outer_vertices['bottom_right'], True)
        self.outer_edges['top'] = Edge(self, self.outer_vertices['top_right'],self.outer_vertices['top_left'], True)
        for edge in self.outer_edges.values():
            self.active_edges.add(edge)
    
    def _all_vertices(self):
        """Return a stable list of unique vertices in this mesh."""
        # If you later add inner vertices, add them here too.
        # For now we include the four outer ones (and anything referenced by faces).
        verts = set(self.outer_vertices.values())
        for f in [self.top_face, self.bottom_face]:
            verts.update(f.vertices)
        # make order stable
        return list(verts)

    def plot_height(self, cmap="viridis", draw_edges=True, draw_vertices=True):
        """
        Color the mesh by linearly interpolated vertex heights using Gouraud shading.
        Assumes each vertex has .height set (e.g., via regress_and_assign()).
        """
        verts = self.active_vertices
        idx = {v: i for i, v in enumerate(verts)}
        xs  = [v.x for v in verts]
        ys  = [v.y for v in verts]
        zs  = [getattr(v, "height", 0.0) for v in verts]  # per-vertex heights

        # Triangles from faces
        tris = []
        for f in self.active_faces:
            tris.append([idx[v] for v in f.vertices])
        tris = np.asarray(tris, dtype=int)

        tri = mtri.Triangulation(xs, ys, triangles=tris)

        fig, ax = plt.subplots()
        tpc = ax.tripcolor(tri, zs, shading="gouraud", cmap=cmap)
        cbar = fig.colorbar(tpc, ax=ax, label="Height")

        if draw_edges:
            # overlay edges for crisp boundaries
            for edge in list(self.active_edges):
                ax.plot([edge.vertex0.x, edge.vertex1.x],
                        [edge.vertex0.y, edge.vertex1.y],
                        "k-", lw=1.0, alpha=0.9)

        if draw_vertices:
            ax.scatter(xs, ys, s=20, c="k", zorder=3)

        ax.set_aspect("equal", "box")
        ax.set_xlim(self.xmin - 1, self.xmax + 1)
        ax.set_ylim(self.ymin - 1, self.ymax + 1)
        ax.set_title("Mesh height (linear interpolation)")
        plt.show()

    def regress_and_assign(self, target_col: int = 2):
        """
        Ordinary Least Squares with NO intercept.
        y = X beta, where X's columns are vertex coefficient vectors.
        Sets vertex.height = beta_j for each vertex column.
        """
        # Build dense design matrix (columns are Vertex objects)
        X_df = pd.DataFrame({v: s for v, s in self.coefficients.items()}, index=self.index).astype(float)

        # Solve OLS
        X = X_df.to_numpy()
        beta, *_ = np.linalg.lstsq(X, self.residuals, rcond=None)

        # Assign back to vertices
        for j, v in enumerate(X_df.columns):
            v.add_height(float(beta[j]))

    
class Vertex:
    def __init__(self, mesh, x: float, y: float, active = False, parent_edge: 'Edge' = None):
        self.x = x
        self.y = y
        self.mesh = mesh
        self.active = active
        
        self.parent_edge = parent_edge
        self.height = 0.0  # will be set later by regression
        
        self.edges: set[Edge] = set()
        self.neighbors: set[Vertex] = set()

    def __repr__(self):
        return f"Vertex(x={self.x}, y={self.y})"

    # --- arithmetic operations
    def __add__(self, other):
        return Vertex(self.mesh, self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vertex(self.mesh, self.x - other.x, self.y - other.y)

    def __mul__(self, s: float):  
        return Vertex(self.mesh, self.x * s, self.y * s)
    
    def __truediv__(self, s: float): return Vertex(self.mesh, self.x / s, self.y / s)
    

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
    
    def add_height(self, value: float):
        self.height += float(value)
        
    def activate(self):
        self.active = True
        self.mesh.active_vertices.add(self)
        if self.parent_edge:
            self.parent_edge.split()
            
    def regress_and_assign(self):
        pass
            
        
    @property
    def degree(self) -> int:
        return len(self.edges)


class Edge:
    def __init__(self, mesh: DecisionMesh, vertex0: Vertex, vertex1: Vertex, active : bool):
        self.active = active
        self.vertex0 = vertex0
        self.vertex1 = vertex1
        self.mesh = mesh
        self.faces = dict({'+': None, '-': None})
        self.opposing_vertices = dict({'+': None, '-': None})
        
        
        dx = vertex1.x - vertex0.x
        dy = vertex1.y - vertex0.y

        # unit normal as numpy array
        n = np.array([-dy, dx], dtype=float)
        n /= np.linalg.norm(n)
        self.normal = n

        # signed line equation: normal · p = intercept
        self.intercept = float(self.normal @ np.array([vertex0.x, vertex0.y], dtype=float))

        if active:
            self.activate()
            
    
    def activate(self):
        self.active = True
        self.mesh.active_edges.add(self)
        self.midpoint = Vertex(self.mesh, (self.vertex0.x + self.vertex1.x)/2, (self.vertex0.y + self.vertex1.y)/2, parent_edge=self)
        self.mesh.midpoints.add(self.midpoint)
        self.vertex0.add_edge(self)
        self.vertex1.add_edge(self)
        self.sub_edges = {'0': Edge(self.mesh, self.vertex0, self.midpoint, False), '1': Edge(self.mesh, self.midpoint, self.vertex1, False),'+': None, '-': None}
            
    def split(self):
        if self.faces['+']:
            self.faces['+'].split(self)
        if self.faces['-']:
            self.faces['-'].split(self)
        
        self.deactivate()
        self.sub_edges['0'].activate()
        self.sub_edges['1'].activate()
            
    def deactivate(self):
        self.active = False
        self.mesh.active_edges.remove(self)
        self.vertex0.remove_edge(self)
        self.vertex1.remove_edge(self)
        self.mesh.midpoints.remove(self.midpoint)
            
    def test_vertex(self, v: Vertex) -> bool:
        return np.array(v) @ self.normal - self.intercept  
        
    def other_vertex(self, v: Vertex) -> Vertex | None:
        if v is self.vertex0: return self.vertex1
        if v is self.vertex1: return self.vertex0
        return None
        
        
    
    def add_face(self, face: 'Face'):
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
        face0 = Face(self.mesh, self.sub_edges[face_type], edge0, self.sub_edges['0'], False)
        face1 = Face(self.mesh, self.sub_edges[face_type], edge1, self.sub_edges['1'], False)
        if face_type == '+':
            face0.path = face.path + '+'
            face1.path = face.path + '-'
            face.add_sub_division(self, {'e': self.sub_edges[face_type], '+': face0, '-': face1})
        else:
            face0.path = face.path + '-'
            face1.path = face.path + '+'
            face.add_sub_division(self, {'e': self.sub_edges[face_type], '+': face1, '-': face0})
    
    def remove_face(self, face):
        if self.faces['+'] is face:
            self.faces['+'] = None
        if self.faces['-'] is face:
            self.faces['-'] = None
class Face:
    def __init__(self, mesh: DecisionMesh, edge0: 'Edge', edge1: 'Edge', edge2: 'Edge', active: bool = True, path = ''):
        self.path = path
        self.active = active
        self.mesh = mesh
        self.node: None
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
        
        if active:
            self.activate()
                
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
        self.active = False
        self.mesh.active_faces.remove(self)
        for edge in self.edges:
            edge.remove_face(self)
            
    def activate(self):
        self.active = True
        self.mesh.active_faces.add(self)
        for edge in self.edges:
            if not edge.active:
                edge.activate()
            edge.add_face(self)
    
    @property
    def area(self):
        #Shoelace Formula
        det = (self.vertices[0].x - self.vertices[2].x)*(self.vertices[1].y - self.vertices[0].y) - (self.vertices[0].x-self.vertices[1].x) * (self.vertices[2].y - self.vertices[0].y)
        return 0.5 * np.abs(det)
    
    def update_mask(self, mask):
        self.mask = mask
        
        for i in range(3):
            self.mesh.ownership[self.vertices[i]] = self.mesh.ownership[self.vertices[i]] | mask
            
        self.update_coords()
    
    
    def update_coords(self, eps=1e-14):
        """
        v0, v1, v2: array-like shape (2,)
        X:          array-like shape (2,) or (n, 2)
        returns:    (w0, w1, w2) each shape (n,) (or scalars if X is (2,))
        """

        X  = self.mesh.X[self.mask]
        if X.ndim == 1:
            X = X[None, :]  # make (1,2)

        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[2] - self.vertices[0]
        p = X  - self.vertices[0]

        d00 = np.dot(a, a)
        d01 = np.dot(a, b)
        d11 = np.dot(b, b)
        d20 = p @ a
        d21 = p @ b

        denom = d00 * d11 - d01 * d01
        # handle near-degenerate triangles
        if abs(denom) < eps:
            # Return NaNs to signal degeneracy
            n = len(X)
            return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))

        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        w = [1.0 - u - v, u, v]
        for i, vertex in enumerate(self.vertices):
            self.mesh.coefficients[vertex][self.mask] += w[i]
    
    def add_sub_division(self, edge: Edge, sub_division: dict):
        idx = self.edges.index(edge)
        self.sub_divisions[idx] = sub_division
        

            
        
    
    def as_patch(self, **kwargs):
        """Return a matplotlib.patches.Polygon for this face."""
        verts = [(v.x, v.y) for v in self.vertices]
        return Polygon(verts, closed=True, **kwargs)