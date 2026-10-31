"""
Laplace and Unbounded Perturbation Mechanisms for baseline comparisons (LapMF).
"""

import numpy as np
from typing import Tuple


class LaplacePerturbation:
    """
    Standard Laplace Noise Injection Mechanism under LDP.
    Noise scale b = (d2 - d1) / epsilon.
    """
    def __init__(self, epsilon: float, domain: Tuple[float, float] = (1.0, 5.0)):
        self.epsilon = float(epsilon)
        self.d1, self.d2 = float(domain[0]), float(domain[1])
        self.sensitivity = self.d2 - self.d1
        self.scale = self.sensitivity / self.epsilon

    def perturb_array(self, ratings: np.ndarray, seed: int = None) -> np.ndarray:
        """Inject Laplace noise into ratings array."""
        rng = np.random.RandomState(seed)
        noise = rng.laplace(loc=0.0, scale=self.scale, size=ratings.shape)
        # Note: Standard LapMF operates on noisy ratings, usually unclipped or clipped to domain
        return ratings + noise
