"""
GMF Baseline (Gaussian Mixture used only for initialization, then standard PMF).
"""

from .pmf import PMF
from .glmf import GLMF
from ..utils.data_loader import RatingDataset


class GMF(PMF):
    """
    GMF baseline: uses Gaussian Mixture Model to initialize latent space, then fits via PMF.
    """
    def __init__(self, num_users: int, num_items: int, num_components: int = 5,
                 latent_dim: int = 15, lambda_reg: float = 0.001,
                 learning_rate: float = 0.001, domain: tuple = (1.0, 5.0), seed: int = 42):
        super().__init__(num_users=num_users, num_items=num_items, latent_dim=latent_dim,
                         lambda_reg=lambda_reg, learning_rate=learning_rate, domain=domain, seed=seed)
        self.num_components = num_components

    def fit(self, dataset: RatingDataset, num_epochs: int = 20, seed: int = 42) -> 'GMF':
        # Quick EM step to initialize U and V
        init_glmf = GLMF(num_users=self.num_users, num_items=self.num_items,
                         num_components=self.num_components, alpha=1.0,
                         latent_dim=self.latent_dim, lambda_reg=self.lambda_reg,
                         learning_rate=self.lr, domain=(self.d1, self.d2), seed=seed)
        init_glmf.fit(dataset, max_em_epochs=3, sgd_inner_epochs=1, seed=seed)
        self.U = init_glmf.U.copy()
        self.V = init_glmf.V.copy()
        
        # Then standard PMF fine-tuning
        super().fit(dataset, num_epochs=num_epochs, seed=seed)
        return self
