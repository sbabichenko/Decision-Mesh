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

    # --- Plot 1: Stacked area of time per refinement ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    window = max(1, len(df) // 100)
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

    # --- Plot 3: Total time per refinement vs mesh size ---
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

    plt.tight_layout()
    out_path = f"{prefix}_timing_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
