"""
Unit tests for GLMF (Gaussian-Laplacian Mixture MF).
"""

import numpy as np
from src.utils.data_loader import generate_synthetic_data
from src.models.glmf import GLMF


def test_glmf_e_step_responsibilities_sum_to_one():
    """Verify that E-step produces posterior responsibilities gamma that sum to 1 across k."""
    dataset = generate_synthetic_data(num_users=30, num_items=20, density=0.2, seed=42)
    glmf = GLMF(num_users=30, num_items=20, num_components=5, seed=42)
    
    gamma = glmf.e_step(dataset)
    
    # Check shape
    assert gamma.shape == (dataset.num_observed, 5)
    
    # Check that responsibilities sum to 1.0 for every rating
    sum_per_rating = np.sum(gamma, axis=1)
    assert np.allclose(sum_per_rating, 1.0, atol=1e-5)


def test_glmf_m_step_updates():
    """Verify M-step updates mixture parameters pi, sigma, b correctly."""
    dataset = generate_synthetic_data(num_users=30, num_items=20, density=0.2, seed=42)
    glmf = GLMF(num_users=30, num_items=20, num_components=5, seed=42)
    
    gamma = glmf.e_step(dataset)
    glmf.m_step(dataset, gamma)
    
    # pi should sum to 1.0
    assert np.isclose(np.sum(glmf.pi), 1.0, atol=1e-5)
    # sigma and b should be positive
    assert np.all(glmf.sigma > 0)
    assert np.all(glmf.b > 0)
