"""
GauMF Baseline (GLMF variant with alpha = 1.0, pure Gaussian noise assumption).
"""

from .glmf import GLMF


class GauMF(GLMF):
    """
    GauMF baseline model: GLMF with alpha = 1.0 (pure Gaussian mixture modeling).
    """
    def __init__(self, num_users: int, num_items: int, num_components: int = 5,
                 latent_dim: int = 15, lambda_reg: float = 0.001,
                 learning_rate: float = 0.001, em_threshold: float = 0.0001,
                 domain: tuple = (1.0, 5.0), seed: int = 42):
        super().__init__(num_users=num_users, num_items=num_items, num_components=num_components,
                         alpha=1.0, latent_dim=latent_dim, lambda_reg=lambda_reg,
                         learning_rate=learning_rate, em_threshold=em_threshold,
                         domain=domain, seed=seed)
