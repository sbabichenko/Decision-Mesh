"""
Train/test comparison: Decision Mesh without vs with Empirical Bayes partial pooling.
Tests across multiple noise levels to show where EB helps most.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def make_data(n=3000, noise_sd=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, n)
    y = rng.uniform(-4, 4, n)
    signal = 2 * np.cos(5 * x) * np.cos(2 * y)
    z = signal + noise_sd * rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: z}), signal


def mesh_to_triangulation(mesh):
    verts = list(mesh.vertices)
    idx = {v: i for i, v in enumerate(verts)}
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.height for v in verts]
    tris = [[idx[v] for v in f.vertices] for f in mesh.active_faces]
    tri = mtri.Triangulation(xs, ys, triangles=np.array(tris, dtype=int))
    return tri, xs, ys, zs


def compute_train_mse(mesh):
    n = len(mesh.values)
    preds = np.zeros(n)
    for f in mesh.active_faces:
        if f.coords is None or f.coords.empty:
            continue
        for idx_val, row in f.coords.iterrows():
            iloc = mesh.index.get_loc(idx_val)
            preds[iloc] = sum(row[v] * v.height for v in f.vertices)
    return float(np.mean((preds - mesh.values) ** 2))


def compute_signal_mse(mesh, signal):
    """MSE against the true signal (no noise), measures denoising ability."""
    n = len(signal)
    preds = np.zeros(n)
    for f in mesh.active_faces:
        if f.coords is None or f.coords.empty:
            continue
        for idx_val, row in f.coords.iterrows():
            iloc = mesh.index.get_loc(idx_val)
            preds[iloc] = sum(row[v] * v.height for v in f.vertices)
    return float(np.mean((preds - signal) ** 2))


def run_mesh(df, n_steps, use_eb=True, rng_seed=42):
    from decision_mesh import DecisionMesh
    mesh = DecisionMesh(df)
    if not use_eb:
        mesh.tau_sq = {}
        mesh.mu_delta = {}
        for v in mesh.vertices:
            v.lambda_v = 0.0
            v.update_info()
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_steps):
        if not use_eb:
            mesh.tau_sq = {}
            mesh.mu_delta = {}
        mesh.update_best_vertex(random=0.1, rng=rng)
    return mesh


def run_and_track(df, signal, n_steps, use_eb=True, rng_seed=42):
    from decision_mesh import DecisionMesh
    mesh = DecisionMesh(df)
    if not use_eb:
        mesh.tau_sq = {}
        mesh.mu_delta = {}
        for v in mesh.vertices:
            v.lambda_v = 0.0
            v.update_info()

    rng = np.random.default_rng(rng_seed)
    steps, train_mses, signal_mses, face_counts = [], [], [], []
    eval_at = set(range(0, n_steps + 1, 5))
    eval_at.add(n_steps)

    for i in range(n_steps + 1):
        if i in eval_at:
            steps.append(i)
            train_mses.append(compute_train_mse(mesh))
            signal_mses.append(compute_signal_mse(mesh, signal))
            face_counts.append(len(mesh.active_faces))
        if i < n_steps:
            if not use_eb:
                mesh.tau_sq = {}
                mesh.mu_delta = {}
            mesh.update_best_vertex(random=0.1, rng=rng)
    return mesh, steps, train_mses, signal_mses, face_counts


def main():
    n_steps = 120
    noise_sd = 2.0  # high noise, SNR=1

    print(f"=== Noise σ={noise_sd}, SNR={2/noise_sd:.1f} ===")
    df_train, signal_train = make_data(n=3000, noise_sd=noise_sd, seed=42)

    print("Running No EB...")
    mesh_no, steps_no, train_no, sig_no, fc_no = run_and_track(
        df_train, signal_train, n_steps, use_eb=False)
    print(f"  {fc_no[-1]} faces, train MSE={train_no[-1]:.4f}, signal MSE={sig_no[-1]:.4f}")

    print("Running With EB...")
    mesh_eb, steps_eb, train_eb, sig_eb, fc_eb = run_and_track(
        df_train, signal_train, n_steps, use_eb=True)
    print(f"  {fc_eb[-1]} faces, train MSE={train_eb[-1]:.4f}, signal MSE={sig_eb[-1]:.4f}")

    # --- Figure 1: Surface comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = 2.5

    gx = np.linspace(-4, 4, 100)
    gy = np.linspace(-4, 4, 100)
    GX, GY = np.meshgrid(gx, gy)
    Z_true = 2 * np.cos(5 * GX) * np.cos(2 * GY)

    ax = axes[0, 0]
    ax.contourf(GX, GY, Z_true, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('True signal: 2cos(5x)cos(2y)', fontsize=11)
    ax.set_aspect('equal')

    tri_no, xs_no, ys_no, zs_no = mesh_to_triangulation(mesh_no)
    ax = axes[0, 1]
    ax.tripcolor(tri_no, zs_no, shading='gouraud', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'No EB — {fc_no[-1]} faces\nSignal MSE={sig_no[-1]:.4f}', fontsize=11)
    ax.set_aspect('equal')

    tri_eb, xs_eb, ys_eb, zs_eb = mesh_to_triangulation(mesh_eb)
    ax = axes[0, 2]
    ax.tripcolor(tri_eb, zs_eb, shading='gouraud', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'With EB — {fc_eb[-1]} faces\nSignal MSE={sig_eb[-1]:.4f}', fontsize=11)
    ax.set_aspect('equal')

    ax = axes[1, 0]
    ax.triplot(tri_no, 'k-', lw=0.3, alpha=0.6)
    ax.set_title(f'No EB mesh ({fc_no[-1]} triangles)', fontsize=11)
    ax.set_aspect('equal'); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

    ax = axes[1, 1]
    ax.triplot(tri_eb, 'k-', lw=0.3, alpha=0.6)
    ax.set_title(f'EB mesh ({fc_eb[-1]} triangles)', fontsize=11)
    ax.set_aspect('equal'); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

    # Vertex heights comparison
    ax = axes[1, 2]
    h_no = sorted([v.height for v in mesh_no.vertices if v.active], key=abs, reverse=True)[:50]
    h_eb = sorted([v.height for v in mesh_eb.vertices if v.active], key=abs, reverse=True)[:50]
    ax.bar(np.arange(len(h_no)) - 0.2, h_no, 0.35, color='tomato', alpha=0.7, label='No EB')
    ax.bar(np.arange(len(h_eb)) + 0.2, h_eb, 0.35, color='steelblue', alpha=0.7, label='With EB')
    ax.axhline(2, color='gray', ls='--', alpha=0.5)
    ax.axhline(-2, color='gray', ls='--', alpha=0.5)
    ax.set_title('Top 50 vertex heights (by |h|)\nGray = true signal range', fontsize=11)
    ax.set_xlabel('Rank'); ax.set_ylabel('Height')
    ax.legend(fontsize=9)

    plt.suptitle(f'Decision Mesh: No EB vs Empirical Bayes (noise σ={noise_sd}, n=3000, {n_steps} steps)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('/home/user/Decision-Mesh/compare_eb.png', dpi=150, bbox_inches='tight')
    print("\nSaved compare_eb.png")
    plt.close()

    # --- Figure 2: Convergence curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(steps_no, sig_no, 'o-', color='tomato', label='No EB', ms=3)
    ax.plot(steps_eb, sig_eb, 's-', color='steelblue', label='With EB', ms=3)
    ax.axhline(0, color='gray', ls='--', alpha=0.3, label='Perfect recovery')
    ax.set_xlabel('Refinement steps'); ax.set_ylabel('Signal MSE')
    ax.set_title('Signal Recovery (MSE vs true signal)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    gap_no = [t - s for t, s in zip(train_no, sig_no)]
    gap_eb = [t - s for t, s in zip(train_eb, sig_eb)]
    ax.plot(steps_no, train_no, 'o-', color='tomato', label='No EB (train)', ms=3)
    ax.plot(steps_eb, train_eb, 's-', color='steelblue', label='With EB (train)', ms=3)
    ax.plot(steps_no, sig_no, 'o--', color='tomato', alpha=0.5, label='No EB (signal)', ms=3)
    ax.plot(steps_eb, sig_eb, 's--', color='steelblue', alpha=0.5, label='With EB (signal)', ms=3)
    ax.set_xlabel('Refinement steps'); ax.set_ylabel('MSE')
    ax.set_title('Train MSE vs Signal MSE\n(gap = overfitting to noise)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(fc_no, sig_no, 'o-', color='tomato', label='No EB', ms=3)
    ax.plot(fc_eb, sig_eb, 's-', color='steelblue', label='With EB', ms=3)
    ax.set_xlabel('Active faces'); ax.set_ylabel('Signal MSE')
    ax.set_title('Signal MSE vs Model Complexity')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Convergence: noise σ={noise_sd}, n=3000', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('/home/user/Decision-Mesh/compare_mse_curve.png', dpi=150, bbox_inches='tight')
    print("Saved compare_mse_curve.png")
    plt.close()

    # --- Figure 3: Multiple noise levels ---
    print("\n=== Multi-noise comparison ===")
    noise_levels = [0.5, 1.0, 2.0, 4.0]
    results = {}

    for sd in noise_levels:
        df_tr, sig_tr = make_data(n=3000, noise_sd=sd, seed=42)
        m_no = run_mesh(df_tr, 100, use_eb=False)
        m_eb = run_mesh(df_tr, 100, use_eb=True)
        sig_mse_no = compute_signal_mse(m_no, sig_tr)
        sig_mse_eb = compute_signal_mse(m_eb, sig_tr)
        results[sd] = (sig_mse_no, sig_mse_eb, len(m_no.active_faces), len(m_eb.active_faces))
        print(f"  σ={sd}: No EB sig_MSE={sig_mse_no:.4f} ({len(m_no.active_faces)} faces), "
              f"EB sig_MSE={sig_mse_eb:.4f} ({len(m_eb.active_faces)} faces)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sds = list(results.keys())
    sig_no_vals = [results[s][0] for s in sds]
    sig_eb_vals = [results[s][1] for s in sds]

    ax = axes[0]
    x_pos = np.arange(len(sds))
    ax.bar(x_pos - 0.2, sig_no_vals, 0.35, color='tomato', alpha=0.8, label='No EB')
    ax.bar(x_pos + 0.2, sig_eb_vals, 0.35, color='steelblue', alpha=0.8, label='With EB')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'σ={s}' for s in sds])
    ax.set_ylabel('Signal MSE (lower = better)')
    ax.set_title('Signal Recovery by Noise Level\n(100 steps, n=3000)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    improvement = [(n - e) / n * 100 for n, e in zip(sig_no_vals, sig_eb_vals)]
    colors = ['forestgreen' if imp > 0 else 'tomato' for imp in improvement]
    ax.bar(x_pos, improvement, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'σ={s}' for s in sds])
    ax.set_ylabel('% improvement in Signal MSE')
    ax.set_title('EB Improvement Over No EB\n(positive = EB is better)')
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/Decision-Mesh/compare_noise_levels.png', dpi=150, bbox_inches='tight')
    print("\nSaved compare_noise_levels.png")
    plt.close()


if __name__ == '__main__':
    main()
