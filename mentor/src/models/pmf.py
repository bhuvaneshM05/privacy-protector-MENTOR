"""
Probabilistic Matrix Factorization (PMF) / NoPriv Baseline.
Implementation of standard PMF with SGD updates per Equation (7).
"""

import numpy as np
from typing import Tuple, Optional
from ..utils.data_loader import RatingDataset


class PMF:
    """
    Probabilistic Matrix Factorization (NoPriv Baseline).
    Paper hyperparameter settings:
    - latent_dim = 15
    - lambda_reg = 0.001
    - learning_rate = 0.001
    """
    def __init__(self, num_users: int, num_items: int, latent_dim: int = 15,
                 lambda_reg: float = 0.001, learning_rate: float = 0.001,
                 domain: Tuple[float, float] = (1.0, 5.0), seed: int = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.lr = learning_rate
        self.d1, self.d2 = domain
        
        # Initialize latent factor matrices U and V centered around sqrt(mean_rating / latent_dim)
        rng = np.random.RandomState(seed)
        mean_rating = (self.d1 + self.d2) / 2.0
        init_val = np.sqrt(mean_rating / latent_dim)
        self.U = rng.normal(loc=init_val, scale=0.05, size=(num_users, latent_dim))
        self.V = rng.normal(loc=init_val, scale=0.05, size=(num_items, latent_dim))

    def fit(self, dataset: RatingDataset, num_epochs: int = 20, seed: int = 42) -> 'PMF':
        """Train PMF using SGD over observed ratings."""
        rng = np.random.RandomState(seed)
        n_obs = dataset.num_observed
        
        for epoch in range(num_epochs):
            indices = rng.permutation(n_obs)
            for idx in indices:
                u = dataset.user_indices[idx]
                i = dataset.item_indices[idx]
                r = dataset.ratings[idx]
                
                # Prediction error
                pred = float(np.dot(self.U[u], self.V[i]))
                err = r - pred
                
                # SGD update per Eq (7)
                u_grad = err * self.V[i] - self.lambda_reg * self.U[u]
                v_grad = err * self.U[u] - self.lambda_reg * self.V[i]
                
                self.U[u] += self.lr * u_grad
                self.V[i] += self.lr * v_grad
                
        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Predict rating matrix entries clipped to domain [d1, d2]."""
        preds = np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
        return np.clip(preds, self.d1, self.d2)

    def predict_full(self) -> np.ndarray:
        """Predict full M x N rating matrix."""
        R_hat = self.U @ self.V.T
        return np.clip(R_hat, self.d1, self.d2)
