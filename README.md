# SOFC-NSR Hierarchical Bayesian Competing-Risk Analysis

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PyMC](https://img.shields.io/badge/PyMC-5.28.4-blue)](https://www.pymc.io/)
[![ArviZ](https://img.shields.io/badge/ArviZ-0.22.0-green)](https://www.arviz.org/)

This repository provides the source code and pre-computed inference results for the paper:

> **EunJoo Park, Hyochan Kwon, Jinkwang Lee.** *Start–Stop Cycle-Induced Failure-Mode Transition in SOFC-Powered Northern Sea Route Shipping: A Hierarchical Bayesian Competing-Risk Analysis.* **Journal of Marine Science and Engineering**, 2026 (under review).

---

## Overview

This work develops a hierarchical Bayesian competing-risk framework that:

1. **Decomposes** SOFC area-specific resistance (ASR) growth into cycle-induced fatigue and time-dependent electrochemical aging components.
2. **Identifies** a failure-mode transition between crack-dominated and ASR-dominated regimes at f ≈ 3–6 cycles/month across six Northern Sea Route (NSR) duty-cycle scenarios.
3. **Derives** route-specific predictive maintenance intervals expressed in both cycle-based (160–600 cycles) and calendar-based (10–160 months) metrics.

---

## Repository Structure

```
sofc-nsr-bayesian-competing-risk/
├── README.md                       This file
├── LICENSE                         CC BY 4.0
├── requirements.txt                Python dependencies
├── verify_results.py               Reproducibility verification script
│
├── code/
│   ├── common.py                   Model definitions, DGP, MCMC factories
│   ├── figures_all.py              All figure generation code
│   └── make_figure[1-6].py         Per-figure standalone scripts
│
├── results/
│   ├── idata_asr_v3.nc             ASR model posterior (4 × 1500 draws)
│   ├── idata_surv_v3.nc            Weibull survival posterior (4 × 2000 draws)
│   ├── idata_asr_noise_0_*_v3.nc   σ_e sensitivity studies (4 noise levels)
│   ├── recovery_arrays_v3.npz      90-cell parameter recovery (true vs posterior)
│   └── run_summary_v3.json         Execution metadata + diagnostics
│
└── docs/                           Supplementary documentation
```

---

## Quick Start

### Option A — Inspect cached results (fast, no MCMC)

The cached inference results in `results/` allow inspection of all reported posterior summaries without re-running MCMC. Verification is fast (under 1 minute):

```bash
pip install -r requirements.txt
python verify_results.py
```

This script reproduces the key values reported in the paper:
- **Table 2** (posterior summaries, R̂, ESS, MCSE)
- **Table 3** (parameter recovery within 94% HDI)
- **Table 6** (maintenance intervals)
- **Section 3.6** (frailty diagnostics, σ_e sensitivity)

### Option B — Reproduce from scratch (slow, full MCMC)

To re-run the entire pipeline including MCMC sampling (approximately 3–4 hours on a 4-core CPU):

```python
from code.common import (
    generate_synthetic_data, build_covariate_matrix,
    fit_asr_model, fit_survival_model,
    ASR_DRAWS, ASR_TUNE, SURV_DRAWS, SURV_TUNE,
    MCMC_COMMON, P, BASE_SEED
)

# 1. Generate synthetic dataset (90 cells × 6 scenarios)
obs_df, fail_df, gt_df = generate_synthetic_data(seed=BASE_SEED)

# 2. Build covariate matrix
Z_raw, Z_mean, Z_std, Z_norm, Z_cell_norm, cell_meta = build_covariate_matrix(obs_df)

# 3. Fit ASR degradation model
idata_asr = fit_asr_model(
    obs_df, Z_cell_norm, n_cells=90,
    draws=ASR_DRAWS, tune=ASR_TUNE,
    chains=4, cores=4, target_accept=0.97
)

# 4. Fit Weibull crack-failure model
idata_surv = fit_survival_model(
    fail_df, Z_cell_norm,
    draws=SURV_DRAWS, tune=SURV_TUNE,
    chains=4, cores=4, target_accept=0.97
)
```

---

## Software Stack

The following library versions were used to produce the published results (recorded in `results/run_summary_v3.json`):

| Component | Version | Purpose |
|---|---|---|
| Python | 3.12.13 | Runtime environment |
| PyMC | 5.28.4 | Probabilistic programming, NUTS sampling |
| ArviZ | 0.22.0 | Posterior diagnostics (R̂, ESS, MCSE, PSIS-LOO, HDI) |
| NumPy | 2.0.2 | Random sampling, linear algebra |
| SciPy | 1.16.3 | Brent-method root finding, prior distributions |
| PyTensor | 2.38.2 | PyMC computational backend |

Earlier major versions (Python 3.11, PyMC 5.16, ArviZ 0.18, NumPy 1.26) yield numerically equivalent results within the reported HDIs.

---

## Reproducibility Notes

### Random Seed Configuration

The master random seed is `BASE_SEED = 2024` (set in `code/common.py`). It is propagated to:
- `numpy.random.default_rng()` (synthetic data generation)
- `pymc.sample(random_seed=...)` (MCMC sampling)
- `scipy.stats.weibull_min.rvs(random_state=...)` (Weibull crack-failure times)

This propagation is documented in Section 2.7 of the main text.

### Expected Outputs

When running with the documented seed and library versions, the following diagnostics should be reproduced (within numerical precision):

```
ASR model
  Maximum R̂:       1.0100  (Section 3.1, Table 2)
  Minimum ESS_bulk:  479    (mu_ASR0)
  Minimum ESS_tail:  684    (mu_cyc)
  Divergent transitions:  0

Key posterior estimates
  alpha_cyc:  mean = 0.7289, 94% HDI = [0.598, 0.872]   (true = 0.720)
  alpha_time: mean = 0.6543, 94% HDI = [0.593, 0.696]   (true = 0.680)
  sigma_e:    mean = 0.0079, 94% HDI = [0.0076, 0.0081] (true = 0.008)

Survival model
  delta_fc:   mean = -0.686, 94% HDI = [-0.946, -0.412]
```

### Computational Resources

The published results were obtained on a 4-core machine running Linux. Approximate wallclock times:

| Stage | Time |
|---|---|
| Synthetic data generation | < 5 s |
| ASR model MCMC (4 × 1500 draws + 4000 tune) | ~ 35 min |
| Survival model MCMC (4 × 2000 draws + 6000 tune) | ~ 4 min |
| σ_e sensitivity studies (4 refits) | ~ 110 min total |
| Sobol sensitivity (post-MCMC) | < 5 min |
| Total | ~ 3 hours |

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@article{park2026sofc,
  title   = {Start--Stop Cycle-Induced Failure-Mode Transition in SOFC-Powered Northern Sea Route Shipping: A Hierarchical Bayesian Competing-Risk Analysis},
  author  = {Park, EunJoo and Kwon, Hyochan and Lee, Jinkwang},
  journal = {Journal of Marine Science and Engineering},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This work is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. See [LICENSE](LICENSE) for details. You are free to share and adapt the material for any purpose, provided you give appropriate credit and indicate if changes were made.

---

## Contact

For questions or issues, please contact the corresponding author:

**Jinkwang Lee** — Department of Mechanical Convergence Engineering, Gyeongsang National University, Changwon 51391, Republic of Korea (`jklee1@gnu.ac.kr`)
