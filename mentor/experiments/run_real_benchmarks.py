"""
Run Benchmarks on Real Datasets (Epinions or Ciao).
Designed for high-performance training on powerful multi-core / GPU workstations.
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_dataset, train_test_split_observed, RatingDataset
from src.models.pmf import PMF
from src.models.lapmf import LapMF
from src.models.gaumf import GauMF
from src.models.gmf import GMF
from src.models.glmf import GLMF
from src.models.mentor import MENTOR
from src.eval.metrics import evaluate_model
from src.eval.rationality import evaluate_laplace_rationality
from data.download_real_data import prepare_epinions, prepare_ciao, DATA_DIR


def run_real_benchmark(dataset_name="epinions", sample_fraction=1.0, epsilons=[0.1, 0.3, 0.5, 0.7, 1.0], seed=42):
    print("=" * 75)
    print(f"      MENTOR REAL-DATASET BENCHMARK: {dataset_name.upper()}")
    print("=" * 75)
    
    # 1. Download/prepare dataset if needed
    if dataset_name.lower() == "epinions":
        csv_file = os.path.join(DATA_DIR, "epinions_clean.csv")
        if not os.path.exists(csv_file):
            csv_file = prepare_epinions()
    elif dataset_name.lower() == "ciao":
        csv_file = os.path.join(DATA_DIR, "ciao_clean.csv")
        if not os.path.exists(csv_file):
            csv_file = prepare_ciao()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'epinions' or 'ciao'.")
        
    print(f"\n[*] Loading {dataset_name.upper()} dataset from: {csv_file}")
    t0 = time.time()
    full_ds = load_dataset(csv_file, sep=',', has_header=True, user_col='user_id', item_col='item_id', rating_col='rating')
    print(f"[OK] Loaded {full_ds.num_observed:,} ratings | {full_ds.num_users:,} users | {full_ds.num_items:,} items in {time.time()-t0:.2f}s")
    
    # Optional subsampling for testing
    if sample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_sample = int(full_ds.num_observed * sample_fraction)
        chosen = rng.choice(full_ds.num_observed, size=n_sample, replace=False)
        print(f"[*] Subsampling {sample_fraction*100:.1f}% -> {n_sample:,} ratings...")
        
        # Remap IDs for subsample to avoid disconnected sparse matrix
        sub_df = pd.DataFrame({
            'u': full_ds.user_indices[chosen],
            'i': full_ds.item_indices[chosen],
            'r': full_ds.ratings[chosen]
        })
        u_map = {u: idx for idx, u in enumerate(sub_df['u'].unique())}
        i_map = {i: idx for idx, i in enumerate(sub_df['i'].unique())}
        full_ds = RatingDataset(
            user_indices=sub_df['u'].map(u_map).values,
            item_indices=sub_df['i'].map(i_map).values,
            ratings=sub_df['r'].values,
            num_users=len(u_map),
            num_items=len(i_map)
        )
        print(f"[OK] Subsampled dataset ready: {full_ds.num_observed:,} ratings ({full_ds.num_users:,} users, {full_ds.num_items:,} items)")

    # 80/20 train/test split (exact paper ratio)
    print("\n[*] Creating 80/20 train/test split...")
    train_ds, test_ds = train_test_split_observed(full_ds, test_ratio=0.2, seed=seed)
    print(f"    - Training observed entries: {train_ds.num_observed:,}")
    print(f"    - Testing observed entries:  {test_ds.num_observed:,}")
    
    results = []

    # 2. Non-private Baseline (NoPriv / PMF)
    print("\n[*] Training NoPriv (PMF) ground truth baseline...")
    t0 = time.time()
    nopriv = PMF(num_users=full_ds.num_users, num_items=full_ds.num_items, latent_dim=15, learning_rate=0.002, seed=seed)
    nopriv.fit(train_ds, num_epochs=12, seed=seed)
    t_nopriv = time.time() - t0
    m_nopriv = evaluate_model(nopriv, test_ds)
    print(f"    NoPriv -> RMSE: {m_nopriv['rmse']:.4f}, F-Measure: {m_nopriv['f_measure']:.4f}, Time: {t_nopriv:.1f}s")

    for eps in epsilons:
        print(f"\n[+] Privacy Budget epsilon = {eps}")
        
        # LapMF Baseline
        t0 = time.time()
        lapmf = LapMF(num_users=full_ds.num_users, num_items=full_ds.num_items, epsilon=eps, latent_dim=15, learning_rate=0.002, seed=seed)
        lapmf.fit(train_ds, num_epochs=12, seed=seed)
        t_lap = time.time() - t0
        m_lap = evaluate_model(lapmf, test_ds)
        print(f"    LapMF  -> RMSE: {m_lap['rmse']:.4f}, F-Measure: {m_lap['f_measure']:.4f}, Time: {t_lap:.1f}s")
        results.append({"dataset": dataset_name, "epsilon": eps, "model": "LapMF", "rmse": m_lap["rmse"], "f_measure": m_lap["f_measure"], "time_sec": t_lap})

        # GauMF Baseline
        t0 = time.time()
        gaumf = GauMF(num_users=full_ds.num_users, num_items=full_ds.num_items, num_components=5, latent_dim=15, learning_rate=0.002, seed=seed)
        gaumf.fit(train_ds, max_em_epochs=8, sgd_inner_epochs=2, seed=seed)
        t_gau = time.time() - t0
        m_gau = evaluate_model(gaumf, test_ds)
        print(f"    GauMF  -> RMSE: {m_gau['rmse']:.4f}, F-Measure: {m_gau['f_measure']:.4f}, Time: {t_gau:.1f}s")
        results.append({"dataset": dataset_name, "epsilon": eps, "model": "GauMF", "rmse": m_gau["rmse"], "f_measure": m_gau["f_measure"], "time_sec": t_gau})

        # GLMF Baseline
        t0 = time.time()
        glmf = GLMF(num_users=full_ds.num_users, num_items=full_ds.num_items, num_components=5, alpha=0.5, latent_dim=15, learning_rate=0.002, seed=seed)
        glmf.fit(train_ds, max_em_epochs=8, sgd_inner_epochs=2, seed=seed)
        t_glmf = time.time() - t0
        m_glmf = evaluate_model(glmf, test_ds)
        print(f"    GLMF   -> RMSE: {m_glmf['rmse']:.4f}, F-Measure: {m_glmf['f_measure']:.4f}, Time: {t_glmf:.1f}s")
        results.append({"dataset": dataset_name, "epsilon": eps, "model": "GLMF", "rmse": m_glmf["rmse"], "f_measure": m_glmf["f_measure"], "time_sec": t_glmf})

        # MENTOR (BBV + GLMF)
        t0 = time.time()
        mentor = MENTOR(num_users=full_ds.num_users, num_items=full_ds.num_items, epsilon=eps, num_components=5, alpha=0.5, latent_dim=15, learning_rate=0.002, seed=seed)
        mentor.fit(train_ds, max_em_epochs=8, sgd_inner_epochs=2, seed=seed)
        t_mentor = time.time() - t0
        m_mentor = evaluate_model(mentor, test_ds)
        print(f"    MENTOR -> RMSE: {m_mentor['rmse']:.4f}, F-Measure: {m_mentor['f_measure']:.4f}, Time: {t_mentor:.1f}s")
        results.append({"dataset": dataset_name, "epsilon": eps, "model": "MENTOR", "rmse": m_mentor["rmse"], "f_measure": m_mentor["f_measure"], "time_sec": t_mentor})

        # Add NoPriv baseline reference row
        results.append({"dataset": dataset_name, "epsilon": eps, "model": "NoPriv", "rmse": m_nopriv["rmse"], "f_measure": m_nopriv["f_measure"], "time_sec": t_nopriv})

    # Save results
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(res_dir, exist_ok=True)
    csv_out = os.path.join(res_dir, f"real_{dataset_name}_results.csv")
    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Results saved to: {csv_out}")

    # Laplace Rationality Check on real data
    rat = evaluate_laplace_rationality(train_ds.ratings[:50000], epsilon=0.5, seed=seed)
    print("\n" + "=" * 70)
    print(f"[*] REAL {dataset_name.upper()} LAPLACE RATIONALITY TEST (epsilon=0.5):")
    print(f"    KS-Statistic: {rat['ks_stat']:.4f}, p-value: {rat['p_value']:.4f}")
    print(f"    BBV Noise Error (RMSE):        {rat['rmse_bbv_diff']:.4f}")
    print(f"    Unbounded Laplace Noise (RMSE): {rat['rmse_lap_diff']:.4f}")
    print(f"    --> BBV Variance Reduction:     {rat['rmse_lap_diff'] / rat['rmse_bbv_diff']:.2f}x")
    print("=" * 70)

    # Plot
    plot_path = os.path.join(res_dir, f"real_{dataset_name}_plot.png")
    plot_real_results(df_res, dataset_name, plot_path)
    return df_res


def plot_real_results(df: pd.DataFrame, dataset_name: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = ["NoPriv", "LapMF", "GauMF", "GLMF", "MENTOR"]
    markers = ['x', 's', '^', 'd', 'o']
    colors = ['gray', 'red', 'green', 'purple', 'blue']
    
    for m, marker, color in zip(models, markers, colors):
        sub = df[df["model"] == m]
        if not sub.empty:
            axes[0].plot(sub["epsilon"], sub["rmse"], marker=marker, label=m, color=color, linewidth=2)
    axes[0].set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].set_title(f"{dataset_name.upper()} - RMSE vs Privacy Budget", fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    for m, marker, color in zip(models, markers, colors):
        sub = df[df["model"] == m]
        if not sub.empty:
            axes[1].plot(sub["epsilon"], sub["f_measure"], marker=marker, label=m, color=color, linewidth=2)
    axes[1].set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    axes[1].set_ylabel("F-Measure", fontsize=12)
    axes[1].set_title(f"{dataset_name.upper()} - F-Measure vs Privacy Budget", fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Performance plot saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MENTOR on real Epinions/Ciao datasets.")
    parser.add_argument("--dataset", type=str, default="epinions", choices=["epinions", "ciao"], help="Dataset to benchmark")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of ratings to use (1.0 = full dataset)")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 1.0], help="Epsilon values")
    args = parser.parse_args()
    
    run_real_benchmark(dataset_name=args.dataset, sample_fraction=args.sample_fraction, epsilons=args.epsilons)
