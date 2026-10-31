"""
Gaussian-Laplacian Mixture Matrix Factorization (GLMF).
Implementation of Section IV-C (Equations 5 & 6, Algorithm 1).
"""

import numpy as np
from typing import Tuple, Dict
from ..utils.data_loader import RatingDataset


class GLMF:
    """
    Gaussian-Laplacian Mixture Model-based Matrix Factorization (GLMF).
    Combines Gaussian (L2) and Laplacian (L1) noise modeling via Expectation-Maximization (EM).
    """
    def __init__(self, num_users: int, num_items: int, num_components: int = 5,
                 alpha: float = 0.5, latent_dim: int = 15, lambda_reg: float = 0.001,
                 learning_rate: float = 0.001, em_threshold: float = 0.0001,
                 domain: Tuple[float, float] = (1.0, 5.0), seed: int = 42):
        self.num_users = num_users
        self.num_items = num_items
        self.K = num_components
        self.alpha = float(alpha)
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.lr = learning_rate
        self.em_threshold = em_threshold
        self.d1, self.d2 = domain
        
        rng = np.random.RandomState(seed)
        mean_rating = (self.d1 + self.d2) / 2.0
        init_val = np.sqrt(mean_rating / latent_dim)
        # Latent factor initialization centered around sqrt(mean_rating / latent_dim)
        self.U = rng.normal(loc=init_val, scale=0.05, size=(num_users, latent_dim))
        self.V = rng.normal(loc=init_val, scale=0.05, size=(num_items, latent_dim))
        
        # Mixture model parameters initialize: \pi_k, \sigma_k, b_k
        self.pi = np.full(self.K, 1.0 / self.K)
        self.sigma = rng.uniform(0.5, 1.5, size=self.K)
        self.b = rng.uniform(0.5, 1.5, size=self.K)

    def _gaussian_pdf(self, x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian PDF: N(x | mu, sigma^2)."""
        sigma = max(sigma, 1e-4)
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    def _laplace_pdf(self, x: np.ndarray, mu: np.ndarray, b: float) -> np.ndarray:
        """Laplace PDF: L(x | mu, b)."""
        b = max(b, 1e-4)
        return (1.0 / (2.0 * b)) * np.exp(-np.abs(x - mu) / b)

    def e_step(self, dataset: RatingDataset) -> np.ndarray:
        """
        E-step: Compute posterior responsibilities \gamma_{ijk} for each observed entry (i,j) and mixture component k.
        Returns gamma matrix of shape (num_observed, K).
        """
        n_obs = dataset.num_observed
        u_idx = dataset.user_indices
        i_idx = dataset.item_indices
        r_star = dataset.ratings
        
        # Compute predicted ratings mu = U_u^T V_i
        preds = np.sum(self.U[u_idx] * self.V[i_idx], axis=1)
        
        gamma = np.zeros((n_obs, self.K), dtype=float)
        for k in range(self.K):
            g_pdf = self._gaussian_pdf(r_star, preds, self.sigma[k])
            l_pdf = self._laplace_pdf(r_star, preds, self.b[k])
            mix_pdf = self.alpha * g_pdf + (1.0 - self.alpha) * l_pdf
            gamma[:, k] = self.pi[k] * mix_pdf
            
        # Normalize across components k (sum to 1)
        sum_gamma = np.sum(gamma, axis=1, keepdims=True)
        sum_gamma = np.where(sum_gamma == 0, 1e-12, sum_gamma)
        gamma /= sum_gamma
        return gamma

    def m_step(self, dataset: RatingDataset, gamma: np.ndarray):
        """
        M-step: Update mixture parameters \pi_k, \sigma_k^2, b_k per Equation (5).
        """
        n_obs = dataset.num_observed
        u_idx = dataset.user_indices
        i_idx = dataset.item_indices
        r_star = dataset.ratings
        preds = np.sum(self.U[u_idx] * self.V[i_idx], axis=1)
        
        diff = r_star - preds
        sq_diff = diff**2
        abs_diff = np.abs(diff)
        
        sum_gamma_k = np.sum(gamma, axis=0) # sum over i,j for each k
        sum_gamma_k_safe = np.where(sum_gamma_k == 0, 1e-12, sum_gamma_k)
        
        # Update \pi_k
        self.pi = sum_gamma_k / n_obs
        
        # Update \sigma_k^2 and b_k
        for k in range(self.K):
            sigma2_k = np.sum(gamma[:, k] * sq_diff) / sum_gamma_k_safe[k]
            self.sigma[k] = np.sqrt(max(sigma2_k, 1e-6))
            
            b_k = np.sum(gamma[:, k] * abs_diff) / sum_gamma_k_safe[k]
            self.b[k] = max(b_k, 1e-6)

    def sgd_step(self, dataset: RatingDataset, gamma: np.ndarray, seed: int = 42):
        """
        Latent Factor Update via SGD maximizing Equation (6).
        """
        rng = np.random.RandomState(seed)
        n_obs = dataset.num_observed
        indices = rng.permutation(n_obs)
        
        # Precompute weights for each observation: W_g = sum_k gamma_k / sigma_k^2, W_l = sum_k gamma_k / b_k
        w_g = np.sum(gamma / (self.sigma**2), axis=1)
        w_l = np.sum(gamma / self.b, axis=1)
        
        for idx in indices:
            u = dataset.user_indices[idx]
            i = dataset.item_indices[idx]
            r = dataset.ratings[idx]
            
            pred = float(np.dot(self.U[u], self.V[i]))
            err = pred - r  # pred - r_star
            
            grad_scalar = w_g[idx] * err + w_l[idx] * np.sign(err)
            
            u_grad = grad_scalar * self.V[i] + self.lambda_reg * self.U[u]
            v_grad = grad_scalar * self.U[u] + self.lambda_reg * self.V[i]
            
            self.U[u] -= self.lr * u_grad
            self.V[i] -= self.lr * v_grad

    def fit(self, dataset: RatingDataset, max_em_epochs: int = 15, sgd_inner_epochs: int = 3,
            seed: int = 42) -> 'GLMF':
        """Full EM + SGD training loop until convergence threshold or max epochs."""
        prev_pi = self.pi.copy()
        
        for epoch in range(max_em_epochs):
            # 1. E-step
            gamma = self.e_step(dataset)
            
            # 2. M-step
            self.m_step(dataset, gamma)
            
            # 3. Latent space updates via SGD
            for inner_ep in range(sgd_inner_epochs):
                self.sgd_step(dataset, gamma, seed=seed + epoch * 100 + inner_ep)
                
            # Convergence check on \pi
            delta_pi = np.max(np.abs(self.pi - prev_pi))
            if delta_pi < self.em_threshold and epoch > 3:
                break
            prev_pi = self.pi.copy()
            
        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Predict ratings clipped to domain [d1, d2]."""
        preds = np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
        return np.clip(preds, self.d1, self.d2)

    def predict_full(self) -> np.ndarray:
        """Predict full rating matrix."""
        R_hat = self.U @ self.V.T
        return np.clip(R_hat, self.d1, self.d2)
