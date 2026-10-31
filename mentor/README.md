# MENTOR: Privacy-Preserving Recommendations via BBV + GLMF under LDP

Python implementation and benchmarking suite reproducing the paper:
**Zhang et al., "Privacy-Preserving Recommendations With Mixture Model-Based Matrix Factorization Under Local Differential Privacy," IEEE Transactions on Industrial Informatics, Vol. 21, No. 7, July 2025.**

---

## 🌟 Overview

MENTOR tackles the privacy-utility tradeoff, communication overhead, and complex mixed noise interactions in local differential privacy (LDP) recommendation systems through two core innovations:

1. **BBV Mechanism (Bidirectional Bounded LDP Perturbation)**:
   - Constrained variance-minimizing input perturbation.
   - Solves for optimal boundary offset $d$ and inner probability $p$ under strict $\varepsilon$-LDP guarantees.
   - Eliminates extreme outlier rating noise (unlike unbounded Laplace perturbation).

2. **GLMF (Gaussian–Laplacian Mixture Matrix Factorization)**:
   - Combines Gaussian $\mathcal{N}(r^* | u_i^T v_j, \sigma_k^2)$ (inherent user noise) and Laplacian $\mathcal{L}(r^* | u_i^T v_j, b_k)$ (privacy-injected noise) over $K=5$ mixture components.
   - Alternating Expectation-Maximization (EM) E-step / M-step for mixture parameters coupled with SGD latent space optimization.
   - Requires only a **single round of communication** between users and server.

---

## 📁 Repository Structure

```
mentor/
├── configs/
│   └── default_config.yaml
├── data/
│   └── data_loader.py
├── src/
│   ├── perturbation/
│   │   ├── bbv.py          # BBV Mechanism (Eqs. 1-4)
│   │   └── laplace.py      # Standard Laplace Mechanism
│   ├── models/
│   │   ├── pmf.py          # NoPriv (PMF) Baseline (Eq. 7)
│   │   ├── lapmf.py        # LapMF Baseline (Eq. 8)
│   │   ├── gaumf.py        # GauMF Baseline (alpha = 1.0)
│   │   ├── gmf.py          # GMF Baseline
│   │   ├── glmf.py         # GLMF Mixture Model (Eqs. 5-6)
│   │   └── mentor.py       # Full MENTOR Framework (BBV + GLMF)
│   ├── eval/
│   │   ├── metrics.py      # RMSE & F-Measure
│   │   └── rationality.py  # Laplace Approximation / KS-Test
│   └── utils/
│       └── data_loader.py  # Rating matrix splitting & dataset generation
├── experiments/
│   └── run_benchmarks.py   # Benchmark script across epsilon in [0.1, 1.0]
├── results/                # Metrics CSV, JSON, and performance plots
├── tests/                  # Unit test suite
├── run_tests.py            # Unit test runner
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
cd mentor
pip install -r requirements.txt
```

### Running Unit Tests

Run the numeric sanity test suite covering BBV expectation/variance, LDP probability ratio bounds, GLMF responsibility summation, and evaluation metrics:

```bash
python run_tests.py
```

### Running Benchmarks & Experiments

Execute the full benchmark suite sweeping $\varepsilon \in [0.1, 1.0]$ across NoPriv, LapMF, GauMF, GMF, GLMF, and MENTOR:

```bash
python experiments/run_benchmarks.py
```

---

## 📊 Evaluation Metrics

- **RMSE**: Root Mean Squared Error on unobserved test ratings $\Omega_{\text{test}}$.
- **F-Measure**: Precision-Recall harmonic mean using rating threshold $R_{ij} \ge 4.0$ as relevant.
- **Laplace Rationality Check**: KS-test and distribution distance between BBV perturbed data and theoretical Laplace noise.
