"""
MENTOR Framework: BBV (Bidirectional Bounded LDP Perturbation) + GLMF (Gaussian-Laplacian Mixture MF).
Section IV End-to-End Pipeline.
"""

import os
import numpy as np
from typing import Tuple
from .glmf import GLMF
from ..perturbation.bbv import BBVPerturbation
from ..utils.data_loader import RatingDataset


class MENTOR:
    """
    MENTOR Pipeline:
    1. Perturbs private user ratings using BBV mechanism under \epsilon-LDP.
    2. Learns robust latent representation (U, V) using GLMF mixture model.
    3. Outputs predictions R_hat = U V^T.
    """
    def __init__(self, num_users: int, num_items: int, epsilon: float = 0.5,
                 num_components: int = 5, alpha: float = 0.5, latent_dim: int = 15,
                 lambda_reg: float = 0.001, learning_rate: float = 0.001,
                 em_threshold: float = 0.0001, domain: Tuple[float, float] = (1.0, 5.0),
                 seed: int = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.epsilon = float(epsilon)
        self.domain = domain
        self.d1, self.d2 = domain
        self.seed = seed
        
        self.bbv = BBVPerturbation(epsilon=self.epsilon, domain=domain)
        self.glmf = GLMF(
            num_users=num_users,
            num_items=num_items,
            num_components=num_components,
            alpha=alpha,
            latent_dim=latent_dim,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            em_threshold=em_threshold,
            domain=domain,
            seed=seed
        )

    def fit(self, dataset: RatingDataset, max_em_epochs: int = 15, sgd_inner_epochs: int = 3,
            seed: int = None) -> 'MENTOR':
        """
        End-to-end fit:
        1. Apply BBV perturbation locally to private user ratings
        2. Train GLMF model on noisy rating dataset R*
        """
        if seed is None:
            seed = self.seed
            
        # 1. BBV Perturbation step
        noisy_ratings = self.bbv.perturb_array(dataset.ratings, seed=seed)
        
        noisy_dataset = RatingDataset(
            user_indices=dataset.user_indices,
            item_indices=dataset.item_indices,
            ratings=noisy_ratings,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            domain=self.domain
        )
        
        # 2. Train GLMF
        self.glmf.fit(noisy_dataset, max_em_epochs=max_em_epochs,
                      sgd_inner_epochs=sgd_inner_epochs, seed=seed)
        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return self.glmf.predict(user_idx, item_idx)

    def predict_full(self) -> np.ndarray:
        return self.glmf.predict_full()

    def save(self, filepath: str):
        """Save trained MENTOR model parameters (latent matrices U, V and mixture parameters)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            U=self.glmf.U,
            V=self.glmf.V,
            pi=self.glmf.pi,
            sigma=self.glmf.sigma,
            b=self.glmf.b,
            num_users=self.num_users,
            num_items=self.num_items,
            epsilon=self.epsilon,
            domain=np.array(self.domain)
        )
        print(f"[OK] Trained model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MENTOR':
        """Load trained MENTOR model parameters from disk."""
        data = np.load(filepath)
        model = cls(
            num_users=int(data['num_users']),
            num_items=int(data['num_items']),
            epsilon=float(data['epsilon']),
            domain=tuple(data['domain'])
        )
        model.glmf.U = data['U']
        model.glmf.V = data['V']
        model.glmf.pi = data['pi']
        model.glmf.sigma = data['sigma']
        model.glmf.b = data['b']
        return model
