from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from heapdict import heapdict

from .tree import TreeNode
from .vertex import Vertex
from .edge import Edge
from .face import Face


class DecisionMesh:

    def __init__(self, df: pd.DataFrame):
        self.max_aspect_ratio = 5
        self.X = df.iloc[:, :2].to_numpy()
        self.values = df.iloc[:, 2].to_numpy()
        self.index = df.index
        self.root = TreeNode(self)
        self.leaves = set()
        self.leaves.add(self.root)
        self.active_faces = set()
        self.vertices = set()
        self.active_edges = set()
        self.midpoints = set()
        self.loss_heap = heapdict()

        # Empirical Bayes partial pooling state
        self.tau_sq = {}                        # depth -> tau_sq
        self.mu_delta = {}                      # depth -> population mean curvature
        self.tau_sq_vertex_counts = {}           # depth -> count at last recomputation
        self.tau_sq_recompute_interval = 20
        self.steps_since_tau_recompute = 0

        self.create_outer_vertices()
        self.create_outer_edges()

        self.split_edge = Edge(self, self.outer_vertices['bottom_left'], self.outer_vertices['top_right'], True)
        mask = (self.X @ self.split_edge.normal) >= self.split_edge.intercept
        self.top_face = Face(self, self.split_edge, self.outer_edges['left'], self.outer_edges['top'], mask, path='+')
        self.bottom_face = Face(self, self.split_edge, self.outer_edges['bottom'], self.outer_edges['right'], ~mask, path='-')
        self.root.split(self.split_edge, self.top_face, self.bottom_face)

        for v in list(self.vertices):
            v.update_info()

        # Bootstrap empirical Bayes: compute tau_sq from initial estimates,
        # then re-update non-corner vertices with regularization
        self.recompute_tau_sq()
        for v in list(self.vertices):
            if v.parent_edge is not None:
                v.update_info()

    def _empty_mask(self) -> pd.Series:
        """Default factory for ownership: a fresh all-False mask aligned to data."""
        return pd.Series(False, index=self.index, dtype=bool)

    def _empty_coeff(self) -> pd.Series:
        """All-zero coefficients aligned to dataset index."""
        return pd.Series(0.0, index=self.index, dtype=float)

    def _faces_and_weights(self, eps: float = 1e-12):
        """
        Active faces only. Returns (faces, weights) where
        weights = area(face) * number_of_points(face).

        - Filters to finite, nonnegative areas.
        - Clips tiny positive areas up to eps to give them some mass.
        """
        faces = list(self.active_faces)
        if not faces:
            return [], np.array([], dtype=float)

        areas = np.array([float(getattr(f, "area", 0.0)) for f in faces], dtype=float)
        # count points in each face via mask
        counts = np.array([int(getattr(f, "mask", None).sum()) if getattr(f, "mask", None) is not None else 0
                           for f in faces], dtype=float)

        mask = np.isfinite(areas) & (areas >= 0.0)
        faces  = [f for f, keep in zip(faces, mask) if keep]
        areas  = areas[mask]
        counts = counts[mask]

        if areas.size == 0:
            return [], np.array([], dtype=float)

        tiny = (areas > 0) & (areas < eps)
        if np.any(tiny):
            areas = areas.copy()
            areas[tiny] = eps

        weights = areas * counts
        return faces, weights

    def random_face(self, rng=None):
        """
        Return a single active Face, sampled with probability proportional to (area * #points).
        Falls back to uniform if all weights are zero.
        """
        faces, weights = self._faces_and_weights()
        if not faces:
            raise RuntimeError("No active faces available to sample from.")

        total = float(weights.sum())
        if total > 0.0:
            p = weights / total
        else:
            p = np.full(len(faces), 1.0 / len(faces), dtype=float)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        idx = int(rng.choice(len(faces), p=p, replace=True))
        return faces[idx]

    def find_best_vertex(self, random: float = 0.05, rng=None):
        """
        With prob `random`, explore:
        - pick a face via self.random_face() (proportional to area * #points),
        - pick its longest edge (ties random, using .length),
        - return that edge's existing midpoint.

        Otherwise, exploit: return the vertex with max loss_reduction.
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        # --- exploration: use midpoint of longest edge in a weighted-random face
        if random > 0.0 and rng.random() < random:
            face = self.random_face(rng=rng)

            # pick longest edge by .length (tie-break randomly)
            lengths = [float(e.length) for e in face.edges]
            max_len = max(lengths)
            tol = 1e-12
            candidates = [e for e, L in zip(face.edges, lengths) if abs(L - max_len) <= tol]
            edge = rng.choice(candidates)

            mid = edge.midpoint  # assumed to exist
            if mid is None:
                raise RuntimeError("Expected midpoint to exist on the chosen edge.")
            if not hasattr(mid, "loss_reduction"):
                mid.update_info()
            return mid, float(getattr(mid, "loss_reduction", 0.0))

        # --- exploitation: best current vertex
        best = None
        best_gain = -float("inf")
        for v in list(self.vertices):
            gain = float(getattr(v, "loss_reduction", -float("inf")))
            if gain > best_gain:
                best = v
                best_gain = gain

        if best is None:
            return None, None

        return best, best_gain

    def update_best_vertex(self, random=0, rng=None):
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        self.maybe_recompute_tau_sq()

        # --- exploration: use midpoint of longest edge in a weighted-random face
        if random > 0.0 and rng.random() < random:
            face = self.random_face(rng=rng)

            # pick longest edge by .length (tie-break randomly)
            lengths = [float(e.length) for e in face.edges]
            max_len = max(lengths)
            tol = 1e-12
            candidates = [e for e, L in zip(face.edges, lengths) if abs(L - max_len) <= tol]
            edge = rng.choice(candidates)

            best = edge.midpoint
        else:
            best, _ = self.loss_heap.peekitem()

        if best.active:
            best.update_height()
        else:
            best.activate()

    def recompute_tau_sq(self):
        by_depth = defaultdict(list)
        for v in self.vertices:
            if v.parent_edge is None:
                continue
            if v.sigma_sq >= float('inf') or v.sigma_sq <= 0:
                continue
            by_depth[v.depth].append(v)

        for d, verts in by_depth.items():
            m = len(verts)
            self.tau_sq_vertex_counts[d] = m

            if m < 3:
                continue

            sum_w = sum(1.0 / v.sigma_sq for v in verts)
            if sum_w <= 0:
                continue
            sum_wd = sum(v.delta_pooled / v.sigma_sq for v in verts)
            mu_d = sum_wd / sum_w
            self.mu_delta[d] = mu_d

            chi_sq = sum((v.delta_pooled - mu_d) ** 2 / v.sigma_sq for v in verts)
            tau_sq = max(0.0, (chi_sq - (m - 1)) / sum_w)
            self.tau_sq[d] = tau_sq

        # For depths with < 3 vertices, borrow from nearest depth
        all_depths_with_tau = sorted(self.tau_sq.keys())
        for d in sorted(by_depth.keys()):
            if d not in self.tau_sq and all_depths_with_tau:
                nearest = min(all_depths_with_tau, key=lambda d2: abs(d2 - d))
                self.tau_sq[d] = self.tau_sq[nearest]
                self.mu_delta.setdefault(d, self.mu_delta.get(nearest, 0.0))

        self.steps_since_tau_recompute = 0

    def maybe_recompute_tau_sq(self):
        self.steps_since_tau_recompute += 1

        if self.steps_since_tau_recompute >= self.tau_sq_recompute_interval:
            self.recompute_tau_sq()
            return

        # Check if any depth has grown by 50%
        by_depth = defaultdict(int)
        for v in self.vertices:
            if v.parent_edge is not None and v.sigma_sq < float('inf'):
                by_depth[v.depth] += 1

        for d, count in by_depth.items():
            old_count = self.tau_sq_vertex_counts.get(d, 0)
            if old_count > 0 and count >= old_count * 1.5:
                self.recompute_tau_sq()
                return

    def create_outer_vertices(self):
        self.xmin = self.X[:, 0].min()
        self.xmax = self.X[:, 0].max()
        self.ymin = self.X[:, 1].min()
        self.ymax = self.X[:, 1].max()
        self.outer_vertices = dict()
        self.outer_vertices['bottom_left'] = Vertex(self, self.xmin, self.ymin, True)
        self.outer_vertices['top_left'] = Vertex(self, self.xmin, self.ymax, True)
        self.outer_vertices['bottom_right'] = Vertex(self, self.xmax, self.ymin, True)
        self.outer_vertices['top_right'] = Vertex(self, self.xmax, self.ymax, True)

    def create_outer_edges(self):
        self.outer_edges = dict()
        self.outer_edges['left'] = Edge(self, self.outer_vertices['top_left'], self.outer_vertices['bottom_left'], False)
        self.outer_edges['bottom'] = Edge(self, self.outer_vertices['bottom_right'], self.outer_vertices['bottom_left'], False)
        self.outer_edges['right'] = Edge(self, self.outer_vertices['top_right'], self.outer_vertices['bottom_right'], False)
        self.outer_edges['top'] = Edge(self, self.outer_vertices['top_right'], self.outer_vertices['top_left'], False)

    def _all_vertices(self):
        """All unique vertices currently referenced by active faces."""
        verts = set()
        for f in list(self.active_faces):
            verts.update(f.vertices)
        return list(verts)

    def plot_height(
        self,
        cmap="viridis",
        draw_edges=False,
        draw_vertices=False,
        vmax_abs=None
    ):
        """
        Color the mesh by linearly interpolated vertex heights using Gouraud shading.
        Assumes each vertex has .height set (e.g., via regress_and_assign()).

        Parameters
        ----------
        cmap : str
            Colormap to use.
        draw_edges : bool
            Whether to overlay mesh edges.
        draw_vertices : bool
            Whether to scatter plot vertices.
        vmax_abs : float or None
            If provided, the colorscale is symmetric from -vmax_abs to +vmax_abs.
        """
        verts = self.vertices
        idx = {v: i for i, v in enumerate(verts)}
        xs  = [v.x for v in verts]
        ys  = [v.y for v in verts]
        zs  = [getattr(v, "height", 0.0) for v in verts]

        # Triangles from faces
        tris = []
        for f in self.active_faces:
            tris.append([idx[v] for v in f.vertices])
        tris = np.asarray(tris, dtype=int)

        tri = mtri.Triangulation(xs, ys, triangles=tris)

        # Set symmetric color limits if requested
        if vmax_abs is not None:
            vmin, vmax = -vmax_abs, vmax_abs
        else:
            vmin, vmax = None, None

        fig, ax = plt.subplots()
        tpc = ax.tripcolor(tri, zs, shading="gouraud", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(tpc, ax=ax, label="Height")

        if draw_edges:
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
