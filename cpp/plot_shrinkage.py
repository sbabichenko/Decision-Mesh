#!/usr/bin/env python3
"""
Side-by-side visualization: EB mesh height (left) and shrinkage factor (right).
Shrinkage = lambda_v / (lambda_v + xTx), ranges 0 (pure data) to 1 (fully shrunk to prior).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def main():
    verts = pd.read_csv("mesh_eb_vertices.csv")
    tris = pd.read_csv("mesh_eb_triangles.csv")
    print(f"EB mesh: {len(verts)} vertices, {len(tris)} faces")
    print(f"Shrinkage range: [{verts['shrinkage'].min():.4f}, {verts['shrinkage'].max():.4f}]")
    print(f"Depth range: [{verts['depth'].min()}, {verts['depth'].max()}]")

    tri = mtri.Triangulation(verts["x"].values, verts["y"].values,
                             triangles=tris[["v0", "v1", "v2"]].values)

    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left: height
    vlim = 3.0
    tpc_h = ax_h.tripcolor(tri, verts["height"].values, shading="gouraud",
                            cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
    ax_h.set_aspect("equal", "box")
    ax_h.set_title("Vertex Height", fontsize=13, fontweight="bold")
    cb_h = fig.colorbar(tpc_h, ax=ax_h, shrink=0.75)
    cb_h.set_label("height")

    # Right: shrinkage
    tpc_s = ax_s.tripcolor(tri, verts["shrinkage"].values, shading="gouraud",
                            cmap="magma_r", vmin=0, vmax=1, rasterized=True)
    ax_s.set_aspect("equal", "box")
    ax_s.set_title("Shrinkage Factor", fontsize=13, fontweight="bold")
    cb_s = fig.colorbar(tpc_s, ax=ax_s, shrink=0.75)
    cb_s.set_label(r"$\lambda_v / (\lambda_v + x^T x)$    [0 = data, 1 = prior]")

    fig.suptitle(f"Empirical Bayes: Height vs Shrinkage ({len(tris):,} faces)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = "mesh_shrinkage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
