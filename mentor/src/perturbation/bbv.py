"""
Bidirectional Bounded LDP Perturbation (BBV Mechanism)
Implementation of Section IV-B (Equations 1-4).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, Union


class BBVPerturbation:
    """
    Bidirectional Bounded Input Perturbation Mechanism (BBV).
    Satisfies \epsilon-LDP while minimizing noise variance over bounded domain [d1, d2].
    """
    def __init__(self, epsilon: float, domain: Tuple[float, float] = (1.0, 5.0)):
        assert epsilon > 0, "Privacy budget epsilon must be positive"
        self.epsilon = float(epsilon)
        self.d1, self.d2 = float(domain[0]), float(domain[1])
        self.exp_eps = np.exp(self.epsilon)

    def _pdf_norm_p(self, d: float) -> float:
        """Compute probability density p in inner range [t-d, t+d] to ensure total probability = 1."""
        domain_len = self.d2 - self.d1
        denom = 2.0 * d * (self.exp_eps - 1.0) + domain_len
        return self.exp_eps / denom

    def _variance(self, d: float, t: float) -> float:
        """
        Compute Var[t'] for a given d and t.
        Eq. (2) from paper.
        """
        p = self._pdf_norm_p(d)
        d1, d2 = self.d1, self.d2
        exp_eps = self.exp_eps

        # Expectation E[t']
        E_t = p * (0.5 - 0.5 / exp_eps) * 4.0 * t * d + (p / (2.0 * exp_eps)) * (d2**2 - d1**2)

        # E[t'^2]
        term1 = (1.0 / 3.0 - 1.0 / (3.0 * exp_eps)) * (6.0 * d * t**2 + 2.0 * d**3)
        term2 = (1.0 / (3.0 * exp_eps)) * (d2**3 - d1**3)
        E_t2 = p * (term1 + term2)

        var = E_t2 - E_t**2
        return var

    def compute_params(self, t: float) -> Tuple[float, float]:
        """
        Find optimal d and p for a given true rating t in [d1, d2].
        Bounded by t - d >= d1 and t + d <= d2, d > 0.
        """
        # Range constraints on d
        max_d = min(t - self.d1, self.d2 - t)
        if max_d <= 1e-6:
            # Handle boundary points gracefully
            max_d = 1e-4

        # Minimize variance w.r.t. d in (1e-6, max_d)
        res = minimize_scalar(
            lambda d: self._variance(d, t),
            bounds=(1e-6, max_d),
            method='bounded'
        )
        opt_d = float(res.x)
        opt_p = self._pdf_norm_p(opt_d)
        return opt_d, opt_p

    def perturb_single(self, t: float, rng: np.random.RandomState = None) -> float:
        """Perturb a single rating value t."""
        if rng is None:
            rng = np.random.RandomState()
            
        t = float(np.clip(t, self.d1, self.d2))
        opt_d, opt_p = self.compute_params(t)

        prob_inner = 2.0 * opt_d * opt_p
        
        # Decide whether to sample from inner range [t-d, t+d] or outer range
        if rng.rand() < prob_inner:
            # Inner range [t - d, t + d]
            return float(rng.uniform(t - opt_d, t + opt_d))
        else:
            # Outer range [d1, t-d) U (t+d, d2]
            left_len = max(0.0, (t - opt_d) - self.d1)
            right_len = max(0.0, self.d2 - (t + opt_d))
            total_outer_len = left_len + right_len
            
            if total_outer_len <= 1e-9:
                return float(rng.uniform(self.d1, self.d2))
                
            prob_left = left_len / total_outer_len
            if rng.rand() < prob_left:
                return float(rng.uniform(self.d1, t - opt_d))
            else:
                return float(rng.uniform(t + opt_d, self.d2))

    def perturb_array(self, ratings: np.ndarray, seed: int = None) -> np.ndarray:
        """Vectorized/batched perturbation for an array of ratings."""
        rng = np.random.RandomState(seed)
        perturbed = np.zeros_like(ratings, dtype=float)
        for idx, r in enumerate(ratings):
            perturbed[idx] = self.perturb_single(r, rng=rng)
        return np.clip(perturbed, self.d1, self.d2)
