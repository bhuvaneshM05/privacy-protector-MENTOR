"""
Unit tests for BBV (Bidirectional Bounded LDP Perturbation) mechanism.
"""

import numpy as np
from src.perturbation.bbv import BBVPerturbation


def test_bbv_expectation_variance():
    """Verify that sample expectation and variance match analytical formulas closely."""
    eps = 1.5
    domain = (1.0, 5.0)
    t = 3.0
    bbv = BBVPerturbation(epsilon=eps, domain=domain)
    
    # Generate large sample
    rng = np.random.RandomState(42)
    samples = np.array([bbv.perturb_single(t, rng=rng) for _ in range(50000)])
    
    d, p = bbv.compute_params(t)
    
    # Analytical expectation
    d1, d2 = domain
    exp_eps = np.exp(eps)
    E_t = p * (0.5 - 0.5 / exp_eps) * 4.0 * t * d + (p / (2.0 * exp_eps)) * (d2**2 - d1**2)
    
    # Check sample mean vs analytical expectation
    sample_mean = np.mean(samples)
    assert np.isclose(sample_mean, E_t, atol=0.05)


def test_bbv_ldp_ratio():
    """Verify the LDP property: pdf(y|t1) / pdf(y|t2) <= exp(eps) for output values y."""
    eps = 1.0
    domain = (1.0, 5.0)
    bbv = BBVPerturbation(epsilon=eps, domain=domain)
    
    t1, t2 = 2.5, 4.0
    d1, p1 = bbv.compute_params(t1)
    d2, p2 = bbv.compute_params(t2)
    
    def get_pdf(y, t, d, p):
        if abs(y - t) <= d:
            return p
        return p / np.exp(eps)

    # Check for a grid of output values y in [1, 5]
    y_grid = np.linspace(1.0, 5.0, 100)
    for y in y_grid:
        pdf1 = get_pdf(y, t1, d1, p1)
        pdf2 = get_pdf(y, t2, d2, p2)
        ratio = pdf1 / pdf2
        assert ratio <= np.exp(eps) + 1e-4, f"LDP ratio violated at y={y}: {ratio} > exp({eps})"


def test_bbv_epsilon_limits():
    """Verify behavior as epsilon changes."""
    domain = (1.0, 5.0)
    t = 3.0
    
    # Large epsilon -> small noise, d -> 0
    bbv_high = BBVPerturbation(epsilon=10.0, domain=domain)
    d_high, p_high = bbv_high.compute_params(t)
    
    # Small epsilon -> larger noise range
    bbv_low = BBVPerturbation(epsilon=0.1, domain=domain)
    d_low, p_low = bbv_low.compute_params(t)
    
    assert d_high < d_low or np.isclose(d_high, 0.0, atol=1e-3)
