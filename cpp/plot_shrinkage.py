#!/usr/bin/env python3
"""
Three-panel visualization: EB mesh height, shrinkage factor s_v, and
inclusion probability p_v from wavelet spike-and-slab model.
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
    print(f"Shrinkage s_v range: [{verts['shrinkage'].min():.4f}, {verts['shrinkage'].max():.4f}]")
    print(f"Inclusion p_v range: [{verts['inclusion_prob'].min():.4f}, {verts['inclusion_prob'].max():.4f}]")
    print(f"Depth range: [{verts['depth'].min()}, {verts['depth'].max()}]")

    tri = mtri.Triangulation(verts["x"].values, verts["y"].values,
                             triangles=tris[["v0", "v1", "v2"]].values)

    fig, (ax_h, ax_s, ax_p) = plt.subplots(1, 3, figsize=(22, 6.5))

    # Left: height
    vlim = 3.0
    tpc_h = ax_h.tripcolor(tri, verts["height"].values, shading="gouraud",
                            cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
    ax_h.set_aspect("equal", "box")
    ax_h.set_title("Vertex Height", fontsize=13, fontweight="bold")
    cb_h = fig.colorbar(tpc_h, ax=ax_h, shrink=0.75)
    cb_h.set_label("height")

    # Center: shrinkage factor s_v
    tpc_s = ax_s.tripcolor(tri, verts["shrinkage"].values, shading="gouraud",
                            cmap="magma_r", vmin=0, vmax=1, rasterized=True)
    ax_s.set_aspect("equal", "box")
    ax_s.set_title("Shrinkage Factor $s_v$", fontsize=13, fontweight="bold")
    cb_s = fig.colorbar(tpc_s, ax=ax_s, shrink=0.75)
    cb_s.set_label(r"$s_v = p_v \cdot \tau^2/(\tau^2 + \sigma^2_v)$    [0 = prior, 1 = data]")

    # Right: inclusion probability p_v
    tpc_p = ax_p.tripcolor(tri, verts["inclusion_prob"].values, shading="gouraud",
                            cmap="viridis", vmin=0, vmax=1, rasterized=True)
    ax_p.set_aspect("equal", "box")
    ax_p.set_title("Inclusion Probability $p_v$", fontsize=13, fontweight="bold")
    cb_p = fig.colorbar(tpc_p, ax=ax_p, shrink=0.75)
    cb_p.set_label(r"$p_v$    [0 = spike (linear), 1 = slab (curvature)]")

    fig.suptitle(f"Wavelet Spike-and-Slab EB: Height / Shrinkage / Inclusion ({len(tris):,} faces)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = "mesh_shrinkage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
