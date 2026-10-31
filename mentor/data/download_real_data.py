"""
Automated downloader and preprocessor for real-world benchmark datasets:
- Epinions (664,824 ratings, 49,289 users, 139,738 items)
- Ciao (284,086 ratings, 7,375 users, 106,797 items)
"""

import os
import urllib.request
import io
import tarfile
import scipy.io
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

EPINIONS_URL = "https://raw.githubusercontent.com/mahdikherad/RS_Deep_graph/master/ratings_data.txt"
CIAO_URL = "https://raw.githubusercontent.com/SocialRecsys/SMIN/main/dataset/CiaoDVD/rating.mat"


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[OK] {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path):,} bytes). Skipping download.")
        return dest_path
    print(f"[*] Downloading from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        out_file.write(resp.read())
    print(f"[OK] Saved to {dest_path} ({os.path.getsize(dest_path):,} bytes).")
    return dest_path


def prepare_epinions() -> str:
    print("\n" + "=" * 60)
    print("Preparing Epinions Dataset (Paper benchmark)")
    print("=" * 60)
    raw_path = os.path.join(DATA_DIR, "epinions_raw.txt")
    
    if not (os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000):
        print(f"[*] Fetching Epinions benchmark data...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(EPINIONS_URL, headers=headers)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        try:
            with tarfile.open(fileobj=io.BytesIO(content)) as tar:
                member = tar.getmember("ratings_data.txt")
                extracted = tar.extractfile(member).read()
                with open(raw_path, "wb") as f:
                    f.write(extracted)
        except Exception:
            with open(raw_path, "wb") as f:
                f.write(content)
        print(f"[OK] Saved to {raw_path} ({os.path.getsize(raw_path):,} bytes).")
    else:
        print(f"[OK] {os.path.basename(raw_path)} already exists ({os.path.getsize(raw_path):,} bytes). Skipping download.")
    
    # Load and clean
    df = pd.read_csv(raw_path, sep=r'\s+', names=['user_id', 'item_id', 'rating'])
    # Bounded domain check [1, 5]
    df['rating'] = np.clip(df['rating'].astype(float), 1.0, 5.0)
    
    clean_path = os.path.join(DATA_DIR, "epinions_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"[OK] Cleaned Epinions saved: {clean_path}")
    print(f"    - Total ratings: {len(df):,}")
    print(f"    - Unique users:   {df['user_id'].nunique():,}")
    print(f"    - Unique items:   {df['item_id'].nunique():,}")
    return clean_path


def prepare_ciao() -> str:
    print("\n" + "=" * 60)
    print("Preparing Ciao Dataset (Paper benchmark)")
    print("=" * 60)
    raw_path = os.path.join(DATA_DIR, "ciao_raw.txt")
    
    if not (os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000):
        print(f"[*] Fetching Ciao benchmark data...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(CIAO_URL, headers=headers)
        with urllib.request.urlopen(req) as resp:
            mat_bytes = resp.read()
        mat = scipy.io.loadmat(io.BytesIO(mat_bytes))
        arr = mat.get('rating')
        if arr is None:
            for k in mat:
                if not k.startswith('__'):
                    arr = mat[k]
                    break
        df_raw = pd.DataFrame({
            'user_id': arr[:, 0].astype(int),
            'item_id': arr[:, 1].astype(int),
            'rating': arr[:, 3].astype(float)
        })
        df_raw.to_csv(raw_path, sep=' ', index=False, header=False)
        print(f"[OK] Saved to {raw_path} ({os.path.getsize(raw_path):,} bytes).")
    else:
        print(f"[OK] {os.path.basename(raw_path)} already exists ({os.path.getsize(raw_path):,} bytes). Skipping download.")
    
    df = pd.read_csv(raw_path, sep=r'\s+', names=['user_id', 'item_id', 'rating'])
    df['rating'] = np.clip(df['rating'].astype(float), 1.0, 5.0)
    
    clean_path = os.path.join(DATA_DIR, "ciao_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"[OK] Cleaned Ciao saved: {clean_path}")
    print(f"    - Total ratings: {len(df):,}")
    print(f"    - Unique users:   {df['user_id'].nunique():,}")
    print(f"    - Unique items:   {df['item_id'].nunique():,}")
    return clean_path


if __name__ == "__main__":
    prepare_epinions()
    prepare_ciao()
    print("\n[OK] Both datasets successfully downloaded and preprocessed in:", DATA_DIR)
