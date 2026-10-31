"""
Rationality of Laplace Distribution Evaluation (Section VI, Fig 2(i)/(j) analog).
Verifies approximation between BBV perturbed data and Laplace distribution.
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import Dict
from ..perturbation.bbv import BBVPerturbation
from ..perturbation.laplace import LaplacePerturbation


def evaluate_laplace_rationality(raw_ratings: np.ndarray, epsilon: float = 0.5,
                                 domain: tuple = (1.0, 5.0), seed: int = 42) -> Dict[str, float]:
    """
    Compares BBV perturbed data distribution vs true Laplace distribution.
    Returns KS-statistic, p-value, and RMSE difference between raw and perturbed.
    """
    bbv = BBVPerturbation(epsilon=epsilon, domain=domain)
    lap = LaplacePerturbation(epsilon=epsilon, domain=domain)
    
    bbv_perturbed = bbv.perturb_array(raw_ratings, seed=seed)
    lap_perturbed = lap.perturb_array(raw_ratings, seed=seed)
    
    bbv_diff = bbv_perturbed - raw_ratings
    lap_diff = lap_perturbed - raw_ratings
    
    ks_stat, p_val = ks_2samp(bbv_diff, lap_diff)
    
    rmse_bbv = float(np.sqrt(np.mean(bbv_diff**2)))
    rmse_lap = float(np.sqrt(np.mean(lap_diff**2)))
    
    return {
        "ks_stat": float(ks_stat),
        "p_value": float(p_val),
        "rmse_bbv_diff": rmse_bbv,
        "rmse_lap_diff": rmse_lap
    }
