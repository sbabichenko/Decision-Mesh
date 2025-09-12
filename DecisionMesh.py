import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ---- repr helpers ----
def _r(x, k=3):
    try: return f"{float(x):.{k}f}"
    except Exception: return str(x)

def _rn(arr, k=3):
    try:
        a = np.asarray(arr).ravel()
        return "(" + ",".join(f"{float(t):.{k}f}" for t in a) + ")"
    except Exception:
        return str(arr)



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
        self.values = df.iloc[:,2].to_numpy()
        self.index = df.index
        self.root = TreeNode(self)
        self.leaves = set()
        self.leaves.add(self.root)
        self.active_faces = set()
        self.vertices = set()
        self.active_edges = set()
        self.midpoints = set()
        
        self.create_outer_vertices()
        self.create_outer_edges()
        
        self.split_edge = Edge(self, self.outer_vertices['bottom_left'],self.outer_vertices['top_right'], True)
        mask = (self.X @ self.split_edge.normal) >= self.split_edge.intercept
        self.top_face = Face(self, self.split_edge,self.outer_edges['left'],self.outer_edges['top'], mask, path = '+')
        self.bottom_face = Face(self, self.split_edge, self.outer_edges['bottom'], self.outer_edges['right'], ~mask, path = '-')
        self.root.split(self.split_edge, self.top_face, self.bottom_face)
        
        for v in list(self.vertices):
            v.update_info()
    
    def _empty_mask(self) -> pd.Series:
        """Default factory for ownership: a fresh all-False mask aligned to data."""
        return pd.Series(False, index=self.index, dtype=bool)
    
    def _empty_coeff(self) -> pd.Series:
        """All-zero coefficients aligned to dataset index."""
        return pd.Series(0.0, index=self.index, dtype=float)
    

        
    def activate_best_midpoint(self):
        best = None
        best_gain = -float("inf")
        for mid in list(self.midpoints):          # list() so activating later doesn't mutate our iterator
            if not hasattr(mid, "loss_reduction"):
                continue
            # coerce to a scalar in case it's a 1-elem array
            try:
                gain = float(np.asarray(mid.loss_reduction).ravel()[0])
            except:
                print(mid.loss_reduction)
            if gain > best_gain:
                best_gain = gain
                best = mid

        if best is None:
            return None, None

        best.activate()
        return best, best_gain
    
    def update_best_vertex(self):
        best = None
        best_gain = -float("inf")
        for mid in list(self.vertices):          # list() so activating later doesn't mutate our iterator
            if mid.loss_reduction > best_gain:
                best = mid
                best_gain = mid.loss_reduction
        if best is None:
            print('no best')
            return None, None

        # print(f'best vertex:{best}, old height {best.height}, new height {best.new_height}, loss reduction {best.loss_reduction}')
        if best.active:
            best.update_height()
        else:
            best.activate()
        
        return best, best_gain
        
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
        

    def create_outer_edges(self):
        self.outer_edges = dict()
        self.outer_edges['left'] = Edge(self, self.outer_vertices['top_left'],self.outer_vertices['bottom_left'], False)
        self.outer_edges['bottom'] = Edge(self, self.outer_vertices['bottom_right'],self.outer_vertices['bottom_left'], False)
        self.outer_edges['right'] = Edge(self, self.outer_vertices['top_right'],self.outer_vertices['bottom_right'], False)
        self.outer_edges['top'] = Edge(self, self.outer_vertices['top_right'],self.outer_vertices['top_left'], False)

    
    def _all_vertices(self):
        """All unique vertices currently referenced by active faces."""
        verts = set()
        for f in list(self.active_faces):
            verts.update(f.vertices)
        return list(verts)

    def plot_height(self, cmap="viridis", draw_edges= False, draw_vertices= False):
        """
        Color the mesh by linearly interpolated vertex heights using Gouraud shading.
        Assumes each vertex has .height set (e.g., via regress_and_assign()).
        """
        verts = self.vertices
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
    
    def __init__(self, mesh: DecisionMesh, x: float, y: float, active = False, parent_edge: 'Edge' = None, real_vertex = True):
        self._id = Vertex._seq; Vertex._seq += 1  # << add
        self.x = x
        self.y = y
        self.mesh = mesh
        
        if real_vertex:
            mesh.vertices.add(self)
        self.active = active
        self.to_update = set()
        self.parent_edge = parent_edge
        if parent_edge is not None:
            self.height = (parent_edge.vertex0.height + parent_edge.vertex1.height) / 2
        else:
            self.height = 0.0  # will be set later by regression
        self.edges: set['Edge'] = set()
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
        return Vertex(self.mesh, self.x + other.x, self.y + other.y, real_vertex = False)

    def __sub__(self, other):
        return Vertex(self.mesh, self.x - other.x, self.y - other.y, real_vertex = False)

    def __mul__(self, s: float):  
        return Vertex(self.mesh, self.x * s, self.y * s, real_vertex = False)
    
    def __truediv__(self, s: float): return Vertex(self.mesh, self.x / s, self.y / s, real_vertex = False)
    

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
            
    def update_info(self):
        self.new_height, self.loss_reduction, self.affected_vertices = self.loc_regress()
        
    def update_height(self):
        self.height = self.new_height
        for v in self.affected_vertices:
            v.update_info()
        self.loss_reduction = 0
        
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
            for s in ('+','-'):
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
        # 1) which faces / neighbors?
        if self.active:
            neighbors, faces = self.get_faces()
        else:
            neighbors, faces = self.get_sim_faces()
            

        if not faces:
            self.mesh.bad_vertex = self
            print('vertex',self,'regressed without being attached to faces')
            print('real', self.real_vertex)
            return {self: float(self.height)}, 0.0, neighbors



        # 2) design matrix for those faces
        X, rows, col_key = self.build_design_matrix(faces)
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

        return float(beta_opt), float(loss_reduction), neighbors

        


    def update_heights(self):
        for v, b in self.height_dict.items():
            v.height = b
        for v in self.to_update:
            v.loc_regress()
    
    
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

class Edge:
    """
    Edges should have the following information:
    
    vertices at each end
    a test (test_vertex) to check which side of the edge a point lies on
    the faces attached to the edge (+ and/or -)
    activity of the edge
    """
    _seq = 0
    def __init__(self, mesh: DecisionMesh, vertex0: Vertex, vertex1: Vertex, active : bool):
        self._id = Edge._seq; Edge._seq += 1
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
        self.midpoint = None
        self.sub_edges = {'0': None, '1':None, '+': None, '-': None}
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
        sub_ready = [k for k, v in subs.items()] if isinstance(subs, dict) else []
        sub_ready = [k for k in ('0','1','+','-') if isinstance(subs, dict) and subs.get(k) is not None]

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
        self.sub_edges = {'0': Edge(self.mesh, self.vertex0, self.midpoint, False), '1': Edge(self.mesh, self.midpoint, self.vertex1, False),'+': None, '-': None}
          
    
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
        if self.midpoint is not None:
            self.mesh.midpoints.discard(self.midpoint)
            
    def test_vertex(self, v: Vertex) -> float:
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
            
        

        self.midpoint.update_info()
    
    def remove_face(self, face):
        if self.faces['+'] is face:
            self.faces['+'] = None
        if self.faces['-'] is face:
            self.faces['-'] = None
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

    def __init__(self, mesh: 'DecisionMesh', edge0: 'Edge', edge1: 'Edge', edge2: 'Edge',
                 mask, active: bool = True, path=''):
        self._id = Face._seq; Face._seq += 1  # <-- stable id
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
        

            
        
    
    def as_patch(self, **kwargs):
        """Return a matplotlib.patches.Polygon for this face."""
        verts = [(v.x, v.y) for v in self.vertices]
        return Polygon(verts, closed=True, **kwargs)