"""
LapMF (Laplace-noise Matrix Factorization) Baseline.
Applies standard Laplace noise injection to rating matrix before MF (Equation 8).
"""

import numpy as np
from typing import Tuple
from .pmf import PMF
from ..perturbation.laplace import LaplacePerturbation
from ..utils.data_loader import RatingDataset


class LapMF(PMF):
    """
    LapMF baseline: injects Laplace noise under LDP into ratings before PMF training.
    """
    def __init__(self, num_users: int, num_items: int, epsilon: float, latent_dim: int = 15,
                 lambda_reg: float = 0.001, learning_rate: float = 0.001,
                 domain: Tuple[float, float] = (1.0, 5.0), seed: int = 42):
        super().__init__(num_users=num_users, num_items=num_items, latent_dim=latent_dim,
                         lambda_reg=lambda_reg, learning_rate=learning_rate, domain=domain, seed=seed)
        self.epsilon = float(epsilon)
        self.laplace_mech = LaplacePerturbation(epsilon=self.epsilon, domain=domain)

    def fit(self, dataset: RatingDataset, num_epochs: int = 20, seed: int = 42) -> 'LapMF':
        """Perturb ratings via Laplace mechanism then train standard PMF."""
        noisy_ratings = self.laplace_mech.perturb_array(dataset.ratings, seed=seed)
        
        noisy_dataset = RatingDataset(
            user_indices=dataset.user_indices,
            item_indices=dataset.item_indices,
            ratings=np.clip(noisy_ratings, self.d1, self.d2),
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            domain=(self.d1, self.d2)
        )
        
        super().fit(noisy_dataset, num_epochs=num_epochs, seed=seed)
        return self
