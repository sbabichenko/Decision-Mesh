#!/usr/bin/env python3
"""
Render mesh CSVs (from C++ decision_mesh) as side-by-side PNG comparison.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def load_mesh(prefix):
    verts = pd.read_csv(f"{prefix}_vertices.csv")
    tris = pd.read_csv(f"{prefix}_triangles.csv")
    return verts, tris


def main():
    print("Loading meshes...")
    noreg_v, noreg_t = load_mesh("mesh_noreg")
    eb_v, eb_t = load_mesh("mesh_eb")

    n_noreg = len(noreg_t)
    n_eb = len(eb_t)
    print(f"  No reg: {len(noreg_v)} vertices, {n_noreg} faces")
    print(f"  EB:     {len(eb_v)} vertices, {n_eb} faces")

    # Use EB height range for the shared color scale (ground truth range is ~[-2, 2])
    vlim = 3.0

    # --- Figure 1: Full mesh comparison (each with own colorbar) ---
    print("Plotting full comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Ground truth on a grid
    gx = np.linspace(eb_v["x"].min(), eb_v["x"].max(), 300)
    gy = np.linspace(eb_v["y"].min(), eb_v["y"].max(), 300)
    GX, GY = np.meshgrid(gx, gy)
    GT = 2.0 * np.cos(5.0 * GX) * np.cos(2.0 * GY)

    im0 = axes[0].imshow(GT, extent=[gx[0], gx[-1], gy[0], gy[-1]],
                          origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                          aspect="equal")
    axes[0].set_title("Ground Truth\nf(x,y) = 2cos(5x)cos(2y)", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], shrink=0.75)

    # No-reg: clip heights to visible range, subsample faces
    noreg_h = noreg_v["height"].values.copy()
    pct = np.percentile(np.abs(noreg_h), 99)
    print(f"  No-reg height 99th percentile: {pct:.1f} (clipping to +/-{vlim})")
    noreg_h_clipped = np.clip(noreg_h, -vlim, vlim)
    if n_noreg > 150000:
        step = max(1, n_noreg // 150000)
        noreg_t_sub = noreg_t.iloc[::step]
        sub_label = f", 1/{step} shown"
    else:
        noreg_t_sub = noreg_t
        sub_label = ""
    tri_noreg = mtri.Triangulation(noreg_v["x"].values, noreg_v["y"].values,
                                    triangles=noreg_t_sub[["v0", "v1", "v2"]].values)
    tpc1 = axes[1].tripcolor(tri_noreg, noreg_h_clipped, shading="gouraud",
                              cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
    axes[1].set_aspect("equal", "box")
    axes[1].set_title(f"No Regularization\n({n_noreg:,} faces{sub_label}, test MSE ~863)",
                      fontsize=12, fontweight="bold")
    fig.colorbar(tpc1, ax=axes[1], shrink=0.75)

    # EB mesh
    tri_eb = mtri.Triangulation(eb_v["x"].values, eb_v["y"].values,
                                 triangles=eb_t[["v0", "v1", "v2"]].values)
    tpc2 = axes[2].tripcolor(tri_eb, eb_v["height"].values, shading="gouraud",
                              cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
    axes[2].set_aspect("equal", "box")
    axes[2].set_title(f"Empirical Bayes\n({n_eb:,} faces, test MSE ~2.8)",
                      fontsize=12, fontweight="bold")
    fig.colorbar(tpc2, ax=axes[2], shrink=0.75)

    fig.suptitle("Decision Mesh: Ground Truth vs No Reg vs Empirical Bayes\n"
                 "(200k points, 80/20 split, 90s refinement)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out1 = "mesh_comparison.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig)

    # --- Figure 2: Zoomed detail with mesh edges ---
    print("Plotting zoomed detail...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

    cx, cy = 0.0, 0.0
    zoom = 1.0

    for ax, (verts, tris_df, label) in zip(axes2, [
        (noreg_v, noreg_t, "No Regularization"),
        (eb_v, eb_t, "Empirical Bayes")
    ]):
        xs = verts["x"].values
        ys = verts["y"].values
        hs = verts["height"].values
        tri_arr = tris_df[["v0", "v1", "v2"]].values

        # Filter to triangles in zoom region
        margin = zoom * 1.3
        in_box = np.zeros(len(tri_arr), dtype=bool)
        for col in range(3):
            vi = tri_arr[:, col]
            in_box |= ((xs[vi] >= cx - margin) & (xs[vi] <= cx + margin) &
                       (ys[vi] >= cy - margin) & (ys[vi] <= cy + margin))
        tri_local = tri_arr[in_box]
        n_local = len(tri_local)

        # Clip heights for no-reg
        hs_plot = np.clip(hs, -vlim, vlim)

        tri = mtri.Triangulation(xs, ys, triangles=tri_local)
        tpc = ax.tripcolor(tri, hs_plot, shading="gouraud",
                           cmap="RdBu_r", vmin=-vlim, vmax=vlim, rasterized=True)
        ax.triplot(tri, "k-", lw=0.25, alpha=0.4)
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
        ax.set_aspect("equal", "box")
        ax.set_title(f"{label} ({n_local:,} faces in view)", fontsize=12, fontweight="bold")
        fig2.colorbar(tpc, ax=ax, shrink=0.75)

    fig2.suptitle("Mesh Detail with Edges (zoomed to center)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = "mesh_detail.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
