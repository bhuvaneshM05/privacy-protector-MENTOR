"""
Run Benchmarking Experiments for MENTOR vs Baselines (NoPriv, LapMF, GauMF, GMF, GLMF).
Sweeps privacy budget \epsilon in [0.1, 1.0] and logs RMSE, F-Measure, and per-prediction latency.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure mentor package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import generate_synthetic_data, train_test_split_observed
from src.models.pmf import PMF
from src.models.lapmf import LapMF
from src.models.gaumf import GauMF
from src.models.gmf import GMF
from src.models.glmf import GLMF
from src.models.mentor import MENTOR
from src.eval.metrics import evaluate_model
from src.eval.rationality import evaluate_laplace_rationality


def run_benchmark(num_users=300, num_items=200, density=0.1, epsilons=[0.1, 0.3, 0.5, 0.7, 1.0], seed=42):
    print("=" * 70)
    print("      MENTOR BENCHMARK EXECUTION (BBV + GLMF under LDP)")
    print("=" * 70)
    
    # 1. Dataset generation / loading
    print(f"[*] Generating rating matrix ({num_users} users, {num_items} items, density {density})...")
    full_dataset = generate_synthetic_data(num_users=num_users, num_items=num_items,
                                           latent_dim=15, density=density, seed=seed)
    train_ds, test_ds = train_test_split_observed(full_dataset, test_ratio=0.2, seed=seed)
    print(f"[*] Train observed entries: {train_ds.num_observed}, Test observed entries: {test_ds.num_observed}")

    results = []

    # 2. Non-private Baseline (NoPriv / PMF)
    print("\n[*] Training NoPriv (PMF) ground truth baseline...")
    t0 = time.time()
    nopriv_model = PMF(num_users=num_users, num_items=num_items, latent_dim=15, seed=seed)
    nopriv_model.fit(train_ds, num_epochs=15, seed=seed)
    t_nopriv = time.time() - t0
    nopriv_metrics = evaluate_model(nopriv_model, test_ds)
    print(f"    NoPriv -> RMSE: {nopriv_metrics['rmse']:.4f}, F-Measure: {nopriv_metrics['f_measure']:.4f}, Time: {t_nopriv:.2f}s")

    for eps in epsilons:
        print(f"\n[+] Running privacy budget epsilon = {eps}")
        
        # LapMF Baseline
        t0 = time.time()
        lapmf = LapMF(num_users=num_users, num_items=num_items, epsilon=eps, latent_dim=15, seed=seed)
        lapmf.fit(train_ds, num_epochs=15, seed=seed)
        t_lap = time.time() - t0
        lap_metrics = evaluate_model(lapmf, test_ds)
        print(f"    LapMF  -> RMSE: {lap_metrics['rmse']:.4f}, F-Measure: {lap_metrics['f_measure']:.4f}, Time: {t_lap:.2f}s")
        results.append({"epsilon": eps, "model": "LapMF", "rmse": lap_metrics["rmse"], "f_measure": lap_metrics["f_measure"], "time_sec": t_lap})

        # GauMF Baseline
        t0 = time.time()
        gaumf = GauMF(num_users=num_users, num_items=num_items, num_components=5, latent_dim=15, seed=seed)
        gaumf.fit(train_ds, max_em_epochs=10, sgd_inner_epochs=2, seed=seed)
        t_gau = time.time() - t0
        gau_metrics = evaluate_model(gaumf, test_ds)
        print(f"    GauMF  -> RMSE: {gau_metrics['rmse']:.4f}, F-Measure: {gau_metrics['f_measure']:.4f}, Time: {t_gau:.2f}s")
        results.append({"epsilon": eps, "model": "GauMF", "rmse": gau_metrics["rmse"], "f_measure": gau_metrics["f_measure"], "time_sec": t_gau})

        # GMF Baseline
        t0 = time.time()
        gmf = GMF(num_users=num_users, num_items=num_items, num_components=5, latent_dim=15, seed=seed)
        gmf.fit(train_ds, num_epochs=15, seed=seed)
        t_gmf = time.time() - t0
        gmf_metrics = evaluate_model(gmf, test_ds)
        print(f"    GMF    -> RMSE: {gmf_metrics['rmse']:.4f}, F-Measure: {gmf_metrics['f_measure']:.4f}, Time: {t_gmf:.2f}s")
        results.append({"epsilon": eps, "model": "GMF", "rmse": gmf_metrics["rmse"], "f_measure": gmf_metrics["f_measure"], "time_sec": t_gmf})

        # GLMF Baseline
        t0 = time.time()
        glmf = GLMF(num_users=num_users, num_items=num_items, num_components=5, alpha=0.5, latent_dim=15, seed=seed)
        glmf.fit(train_ds, max_em_epochs=10, sgd_inner_epochs=2, seed=seed)
        t_glmf = time.time() - t0
        glmf_metrics = evaluate_model(glmf, test_ds)
        print(f"    GLMF   -> RMSE: {glmf_metrics['rmse']:.4f}, F-Measure: {glmf_metrics['f_measure']:.4f}, Time: {t_glmf:.2f}s")
        results.append({"epsilon": eps, "model": "GLMF", "rmse": glmf_metrics["rmse"], "f_measure": glmf_metrics["f_measure"], "time_sec": t_glmf})

        # MENTOR (BBV + GLMF)
        t0 = time.time()
        mentor = MENTOR(num_users=num_users, num_items=num_items, epsilon=eps, num_components=5, alpha=0.5, latent_dim=15, seed=seed)
        mentor.fit(train_ds, max_em_epochs=10, sgd_inner_epochs=2, seed=seed)
        t_mentor = time.time() - t0
        mentor_metrics = evaluate_model(mentor, test_ds)
        print(f"    MENTOR -> RMSE: {mentor_metrics['rmse']:.4f}, F-Measure: {mentor_metrics['f_measure']:.4f}, Time: {t_mentor:.2f}s")
        results.append({"epsilon": eps, "model": "MENTOR", "rmse": mentor_metrics["rmse"], "f_measure": mentor_metrics["f_measure"], "time_sec": t_mentor})

        # NoPriv reference log per epsilon
        results.append({"epsilon": eps, "model": "NoPriv", "rmse": nopriv_metrics["rmse"], "f_measure": nopriv_metrics["f_measure"], "time_sec": t_nopriv})

    # Save results
    os.makedirs("results", exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/benchmark_results.csv", index=False)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n[*] Benchmark results saved to results/benchmark_results.csv")
    
    # 3. Laplace Rationality Evaluation
    rat_res = evaluate_laplace_rationality(train_ds.ratings, epsilon=0.5, seed=seed)
    print("\n[*] Laplace Rationality Test (BBV vs Laplace distribution, epsilon=0.5):")
    print(f"    KS-statistic: {rat_res['ks_stat']:.4f}, p-value: {rat_res['p_value']:.4f}")
    print(f"    RMSE BBV diff: {rat_res['rmse_bbv_diff']:.4f}, RMSE Lap diff: {rat_res['rmse_lap_diff']:.4f}")

    # 4. Plotting curves (Fig. 2 analog)
    plot_benchmark_results(df_res, epsilons)
    return df_res


def plot_benchmark_results(df: pd.DataFrame, epsilons: list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ["NoPriv", "LapMF", "GauMF", "GMF", "GLMF", "MENTOR"]
    markers = ['x', 's', '^', 'v', 'd', 'o']
    colors = ['gray', 'red', 'green', 'orange', 'purple', 'blue']
    
    # Plot RMSE vs Epsilon
    for m, marker, color in zip(models, markers, colors):
        m_df = df[df["model"] == m]
        if not m_df.empty:
            axes[0].plot(m_df["epsilon"], m_df["rmse"], marker=marker, label=m, color=color)
    axes[0].set_xlabel("Privacy Budget (epsilon)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE vs Privacy Budget (epsilon)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Plot F-Measure vs Epsilon
    for m, marker, color in zip(models, markers, colors):
        m_df = df[df["model"] == m]
        if not m_df.empty:
            axes[1].plot(m_df["epsilon"], m_df["f_measure"], marker=marker, label=m, color=color)
    axes[1].set_xlabel("Privacy Budget (epsilon)")
    axes[1].set_ylabel("F-Measure")
    axes[1].set_title("F-Measure vs Privacy Budget (epsilon)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/performance_comparison.png", dpi=300)
    plt.close()
    print("[*] Performance plot saved to results/performance_comparison.png")


if __name__ == "__main__":
    run_benchmark()
