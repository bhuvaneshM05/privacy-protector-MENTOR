"""
Data Loading, Generation, and Splitting Pipeline for MENTOR.
Handles sparse rating matrices R in R^{M x N} with bounded ratings in D = [d1, d2].
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class RatingDataset:
    """
    Rating Dataset container for sparse user-item interaction matrices.
    """
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray,
                 num_users: int, num_items: int, domain: Tuple[float, float] = (1.0, 5.0)):
        self.user_indices = np.array(user_indices, dtype=int)
        self.item_indices = np.array(item_indices, dtype=int)
        self.ratings = np.array(ratings, dtype=float)
        self.num_users = num_users
        self.num_items = num_items
        self.d1, self.d2 = domain
        
        # Validation
        assert np.all((self.ratings >= self.d1) & (self.ratings <= self.d2)), \
            f"Ratings must be bounded within [{self.d1}, {self.d2}]"
        assert np.all((self.user_indices >= 0) & (self.user_indices < num_users)), \
            f"User indices out of bound [0, {num_users})"
        assert np.all((self.item_indices >= 0) & (self.item_indices < num_items)), \
            f"Item indices out of bound [0, {num_items})"

    @property
    def num_observed(self) -> int:
        return len(self.ratings)

    def to_sparse_dict(self) -> Dict[Tuple[int, int], float]:
        """Convert observed ratings to dictionary mapping (u, i) -> rating."""
        return {(u, i): r for u, i, r in zip(self.user_indices, self.item_indices, self.ratings)}


def train_test_split_observed(dataset: RatingDataset, test_ratio: float = 0.2,
                             seed: int = 42) -> Tuple[RatingDataset, RatingDataset]:
    """
    Split observed entries \Omega into 80/20 train/test split.
    """
    rng = np.random.RandomState(seed)
    n_obs = dataset.num_observed
    indices = rng.permutation(n_obs)
    test_size = int(n_obs * test_ratio)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    train_ds = RatingDataset(
        user_indices=dataset.user_indices[train_idx],
        item_indices=dataset.item_indices[train_idx],
        ratings=dataset.ratings[train_idx],
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        domain=(dataset.d1, dataset.d2)
    )
    
    test_ds = RatingDataset(
        user_indices=dataset.user_indices[test_idx],
        item_indices=dataset.item_indices[test_idx],
        ratings=dataset.ratings[test_idx],
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        domain=(dataset.d1, dataset.d2)
    )
    
    return train_ds, test_ds


def generate_synthetic_data(num_users: int = 200, num_items: int = 100,
                            latent_dim: int = 15, density: float = 0.1,
                            domain: Tuple[float, float] = (1.0, 5.0),
                            seed: int = 42) -> RatingDataset:
    """
    Generate synthetic low-rank rating matrix bounded in [d1, d2] for benchmarking.
    """
    rng = np.random.RandomState(seed)
    d1, d2 = domain
    
    # Generate latent factors
    U_true = rng.randn(num_users, latent_dim) * 0.5
    V_true = rng.randn(num_items, latent_dim) * 0.5
    
    # True ratings centered around mean of domain
    mean_val = (d1 + d2) / 2.0
    scale_val = (d2 - d1) / 4.0
    R_full = mean_val + scale_val * (U_true @ V_true.T)
    R_full = np.clip(R_full, d1, d2)
    
    # Sample observed entries
    num_obs = int(num_users * num_items * density)
    all_pairs = np.argwhere(np.ones((num_users, num_items), dtype=bool))
    chosen_indices = rng.choice(len(all_pairs), size=num_obs, replace=False)
    
    obs_pairs = all_pairs[chosen_indices]
    u_idx = obs_pairs[:, 0]
    i_idx = obs_pairs[:, 1]
    ratings = R_full[u_idx, i_idx]
    
    return RatingDataset(u_idx, i_idx, ratings, num_users, num_items, domain)


def load_dataset(file_path: str, sep: str = ',', has_header: bool = True,
                 user_col: str = 'user_id', item_col: str = 'item_id', rating_col: str = 'rating',
                 domain: Tuple[float, float] = (1.0, 5.0)) -> RatingDataset:
    """
    Load rating dataset from CSV/TSV file.
    Maps user and item IDs to contiguous range [0, M-1] and [0, N-1].
    """
    if has_header:
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = pd.read_csv(file_path, sep=sep, names=[user_col, item_col, rating_col, 'timestamp'])
        
    user_map = {uid: idx for idx, uid in enumerate(df[user_col].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df[item_col].unique())}
    
    u_indices = df[user_col].map(user_map).values
    i_indices = df[item_col].map(item_map).values
    ratings = np.clip(df[rating_col].values, domain[0], domain[1])
    
    return RatingDataset(
        user_indices=u_indices,
        item_indices=i_indices,
        ratings=ratings,
        num_users=len(user_map),
        num_items=len(item_map),
        domain=domain
    )
