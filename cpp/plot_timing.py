#!/usr/bin/env python3
"""Plot timing breakdown from Decision Mesh C++ timing CSV."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_timing.py <timing.csv> [output_prefix]")
        sys.exit(1)

    csv_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace("_timing.csv", "")

    df = pd.read_csv(csv_path)

    # Convert microseconds to milliseconds
    for col in ["find_best_us", "split_us", "update_info_us", "total_us"]:
        df[col.replace("_us", "_ms")] = df[col] / 1000.0

    has_cascade = "cascade_activations" in df.columns

    if has_cascade:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = max(1, len(df) // 100)

    # --- Plot 1: Stacked area of time per refinement ---
    ax = axes[0, 0]
    find_smooth = df["find_best_ms"].rolling(window, min_periods=1).mean()
    split_smooth = df["split_ms"].rolling(window, min_periods=1).mean()
    update_smooth = df["update_info_ms"].rolling(window, min_periods=1).mean()

    ax.stackplot(
        df["iteration"],
        find_smooth, split_smooth, update_smooth,
        labels=["Find best vertex", "Split/activate", "Update info (regression)"],
        colors=["#2196F3", "#FF9800", "#4CAF50"],
        alpha=0.8,
    )
    ax.set_xlabel("Refinement iteration")
    ax.set_ylabel("Time (ms, smoothed)")
    ax.set_title("Time per refinement (stacked)")
    ax.legend(loc="upper left")

    # --- Plot 2: Percentage breakdown ---
    ax = axes[0, 1]
    total = df["find_best_ms"] + df["split_ms"] + df["update_info_ms"]
    total = total.replace(0, np.nan)
    pct_find = (df["find_best_ms"] / total * 100).rolling(window, min_periods=1).mean()
    pct_split = (df["split_ms"] / total * 100).rolling(window, min_periods=1).mean()
    pct_update = (df["update_info_ms"] / total * 100).rolling(window, min_periods=1).mean()

    ax.stackplot(
        df["iteration"],
        pct_find, pct_split, pct_update,
        labels=["Find best vertex", "Split/activate", "Update info (regression)"],
        colors=["#2196F3", "#FF9800", "#4CAF50"],
        alpha=0.8,
    )
    ax.set_xlabel("Refinement iteration")
    ax.set_ylabel("Percentage of time")
    ax.set_title("Time breakdown (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")

    # --- Plot 3: Total time vs mesh size ---
    ax = axes[1, 0]
    ax.scatter(df["active_faces"], df["total_us"] / 1000, s=3, alpha=0.3, c="#673AB7")
    ax.set_xlabel("Active faces")
    ax.set_ylabel("Time per refinement (ms)")
    ax.set_title("Refinement cost vs mesh complexity")

    # --- Plot 4: Per-phase time vs mesh size ---
    ax = axes[1, 1]
    ax.scatter(df["active_faces"], df["find_best_ms"], s=3, alpha=0.3, label="Find best", c="#2196F3")
    ax.scatter(df["active_faces"], df["split_ms"], s=3, alpha=0.3, label="Split/activate", c="#FF9800")
    ax.scatter(df["active_faces"], df["update_info_ms"], s=3, alpha=0.3, label="Update info", c="#4CAF50")
    ax.set_xlabel("Active faces")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-phase cost vs mesh complexity")
    ax.legend(markerscale=5)

    if has_cascade:
        # --- Plot 5: Cascade depth and object counts vs iteration ---
        ax = axes[2, 0]
        cascade_smooth = df["cascade_activations"].rolling(window, min_periods=1).mean()
        ax.plot(df["iteration"], cascade_smooth, color="#E91E63", linewidth=1, label="Cascade activations")
        ax.set_xlabel("Refinement iteration")
        ax.set_ylabel("Count (smoothed)")
        ax.set_title("Completion cascade depth per refinement")
        ax.legend()

        ax2 = ax.twinx()
        faces_smooth = df["faces_created"].rolling(window, min_periods=1).mean()
        add_face_smooth = df["add_face_calls"].rolling(window, min_periods=1).mean()
        ax2.plot(df["iteration"], faces_smooth, color="#9C27B0", linewidth=1, alpha=0.7, label="Faces created")
        ax2.plot(df["iteration"], add_face_smooth, color="#FF5722", linewidth=1, alpha=0.7, label="add_face calls")
        ax2.set_ylabel("Objects created")
        ax2.legend(loc="upper right")

        # --- Plot 6: Split time vs cascade depth ---
        ax = axes[2, 1]
        ax.scatter(df["cascade_activations"], df["split_ms"], s=5, alpha=0.3, c="#E91E63")
        ax.set_xlabel("Cascade activations (completion depth)")
        ax.set_ylabel("Split time (ms)")
        ax.set_title("Split cost driven by cascade depth")

        # Add text annotation with correlation
        corr = df["cascade_activations"].corr(df["split_ms"])
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = f"{prefix}_timing_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
