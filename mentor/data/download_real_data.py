"""
Automated downloader and preprocessor for real-world benchmark datasets:
- Epinions (664,824 ratings, 49,289 users, 139,738 items)
- Ciao (284,086 ratings, 7,375 users, 106,797 items)
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

EPINIONS_URL = "https://raw.githubusercontent.com/chenhaotian/GraphRec-WWW19/master/data/Epinions/ratings_data.txt"
CIAO_URL = "https://raw.githubusercontent.com/chenhaotian/GraphRec-WWW19/master/data/Ciao/ratings_data.txt"


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[✓] {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path):,} bytes). Skipping download.")
        return dest_path
    print(f"[*] Downloading from {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[✓] Saved to {dest_path} ({os.path.getsize(dest_path):,} bytes).")
    return dest_path


def prepare_epinions() -> str:
    print("\n" + "=" * 60)
    print("Preparing Epinions Dataset (Paper benchmark)")
    print("=" * 60)
    raw_path = os.path.join(DATA_DIR, "epinions_raw.txt")
    download_file(EPINIONS_URL, raw_path)
    
    # Load and clean
    df = pd.read_csv(raw_path, sep=r'\s+', names=['user_id', 'item_id', 'rating'])
    # Bounded domain check [1, 5]
    df['rating'] = np.clip(df['rating'].astype(float), 1.0, 5.0)
    
    clean_path = os.path.join(DATA_DIR, "epinions_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"[✓] Cleaned Epinions saved: {clean_path}")
    print(f"    - Total ratings: {len(df):,}")
    print(f"    - Unique users:   {df['user_id'].nunique():,}")
    print(f"    - Unique items:   {df['item_id'].nunique():,}")
    return clean_path


def prepare_ciao() -> str:
    print("\n" + "=" * 60)
    print("Preparing Ciao Dataset (Paper benchmark)")
    print("=" * 60)
    raw_path = os.path.join(DATA_DIR, "ciao_raw.txt")
    download_file(CIAO_URL, raw_path)
    
    df = pd.read_csv(raw_path, sep=r'\s+', names=['user_id', 'item_id', 'rating'])
    df['rating'] = np.clip(df['rating'].astype(float), 1.0, 5.0)
    
    clean_path = os.path.join(DATA_DIR, "ciao_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"[✓] Cleaned Ciao saved: {clean_path}")
    print(f"    - Total ratings: {len(df):,}")
    print(f"    - Unique users:   {df['user_id'].nunique():,}")
    print(f"    - Unique items:   {df['item_id'].nunique():,}")
    return clean_path


if __name__ == "__main__":
    prepare_epinions()
    prepare_ciao()
    print("\n[✓] Both datasets successfully downloaded and preprocessed in:", DATA_DIR)
