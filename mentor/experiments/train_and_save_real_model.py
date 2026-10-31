"""
Train and save the MENTOR model on real-world datasets (Epinions and Ciao).
Produces compressed model artifacts containing trained latent factors U, V and mixture parameters.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_dataset
from src.models.mentor import MENTOR
from data.download_real_data import DATA_DIR

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def train_and_save(dataset_name="epinions", epsilon=0.5, seed=42):
    csv_file = os.path.join(DATA_DIR, f"{dataset_name.lower()}_clean.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Clean dataset not found at {csv_file}")

    print(f"\n{'='*70}")
    print(f"[*] Training MENTOR on REAL {dataset_name.upper()} DATASET (epsilon={epsilon})")
    print(f"{'='*70}")
    t0 = time.time()
    ds = load_dataset(csv_file, sep=',', has_header=True, user_col='user_id', item_col='item_id', rating_col='rating')
    print(f"[OK] Loaded {ds.num_observed:,} ratings | {ds.num_users:,} users | {ds.num_items:,} items")

    model = MENTOR(
        num_users=ds.num_users,
        num_items=ds.num_items,
        epsilon=epsilon,
        num_components=5,
        alpha=0.5,
        latent_dim=15,
        seed=seed
    )
    
    print("[*] Fitting MENTOR model (BBV perturbation + GLMF EM updates)...")
    model.fit(ds, max_em_epochs=5, sgd_inner_epochs=1, seed=seed)
    print(f"[OK] Training complete in {time.time()-t0:.2f}s")

    save_path = os.path.join(SAVED_MODELS_DIR, f"mentor_{dataset_name.lower()}_model.npz")
    model.save(save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[OK] Model artifact size: {file_size_mb:.2f} MB")
    return save_path


if __name__ == "__main__":
    train_and_save("epinions", epsilon=0.5)
    train_and_save("ciao", epsilon=0.5)
    print(f"\n[OK] Both real-world trained models saved in: {SAVED_MODELS_DIR}")

