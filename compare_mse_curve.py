"""
MSE convergence curves: No EB vs With EB.
Uses the mesh's own barycentric coordinates for accurate MSE computation.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, n)
    y = rng.uniform(-4, 4, n)
    z = 2 * np.cos(5 * x) * np.cos(2 * y) + rng.standard_normal(n)
    return pd.DataFrame({0: x, 1: y, 2: z})


def compute_mse(mesh):
    n = len(mesh.values)
    preds = np.zeros(n)
    for f in mesh.active_faces:
        if f.coords is None or f.coords.empty:
            continue
        for idx_val, row in f.coords.iterrows():
            iloc = mesh.index.get_loc(idx_val)
            preds[iloc] = sum(row[v] * v.height for v in f.vertices)
    return float(np.mean((preds - mesh.values) ** 2))


def run_and_track(df, n_steps, use_eb=True, rng_seed=42):
    from decision_mesh import DecisionMesh
    mesh = DecisionMesh(df)

    if not use_eb:
        mesh.tau_sq = {}
        mesh.mu_delta = {}
        for v in mesh.vertices:
            v.lambda_v = 0.0
            v.update_info()

    rng = np.random.default_rng(rng_seed)
    steps = []
    mses = []
    face_counts = []

    eval_at = set(range(0, n_steps + 1, 5))
    eval_at.add(n_steps)

    for i in range(n_steps + 1):
        if i in eval_at:
            steps.append(i)
            mses.append(compute_mse(mesh))
            face_counts.append(len(mesh.active_faces))

        if i < n_steps:
            if not use_eb:
                mesh.tau_sq = {}
                mesh.mu_delta = {}
            mesh.update_best_vertex(random=0.1, rng=rng)

    return steps, mses, face_counts


def main():
    df = make_data(n=5000, seed=42)
    n_steps = 150

    print("Running No EB...")
    t0 = time.time()
    steps_no, mse_no, fc_no = run_and_track(df, n_steps, use_eb=False)
    print(f"  {time.time()-t0:.1f}s")

    print("Running With EB...")
    t0 = time.time()
    steps_eb, mse_eb, fc_eb = run_and_track(df, n_steps, use_eb=True)
    print(f"  {time.time()-t0:.1f}s")

    var_z = float(np.var(df.iloc[:, 2].to_numpy()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MSE vs steps
    ax = axes[0]
    ax.plot(steps_no, mse_no, 'o-', color='tomato', label='No EB', markersize=3)
    ax.plot(steps_eb, mse_eb, 's-', color='steelblue', label='With EB', markersize=3)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Noise floor (σ²=1)')
    ax.axhline(var_z, color='gray', ls=':', alpha=0.5, label=f'Var(z)={var_z:.2f}')
    ax.set_xlabel('Refinement steps')
    ax.set_ylabel('Training MSE')
    ax.set_title('MSE vs Refinement Steps')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # MSE vs face count
    ax = axes[1]
    ax.plot(fc_no, mse_no, 'o-', color='tomato', label='No EB', markersize=3)
    ax.plot(fc_eb, mse_eb, 's-', color='steelblue', label='With EB', markersize=3)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Noise floor')
    ax.set_xlabel('Number of active faces')
    ax.set_ylabel('Training MSE')
    ax.set_title('MSE vs Model Complexity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Face count vs steps
    ax = axes[2]
    ax.plot(steps_no, fc_no, 'o-', color='tomato', label='No EB', markersize=3)
    ax.plot(steps_eb, fc_eb, 's-', color='steelblue', label='With EB', markersize=3)
    ax.set_xlabel('Refinement steps')
    ax.set_ylabel('Active faces')
    ax.set_title('Mesh Growth')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Decision-Mesh/compare_mse_curve.png', dpi=150, bbox_inches='tight')
    print("\nSaved compare_mse_curve.png")
    plt.close()

    print(f"\nFinal (step {n_steps}):")
    print(f"  No EB:   MSE={mse_no[-1]:.4f}, {fc_no[-1]} faces")
    print(f"  With EB: MSE={mse_eb[-1]:.4f}, {fc_eb[-1]} faces")
    print(f"  Noise floor: 1.0, Var(z): {var_z:.4f}")


if __name__ == '__main__':
    main()
