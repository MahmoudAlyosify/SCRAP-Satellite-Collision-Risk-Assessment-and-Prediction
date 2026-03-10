# 🛰️ SCRAP — Satellite Collision Risk Assessment and Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.x-orange?logo=xgboost"/>
  <img src="https://img.shields.io/badge/LightGBM-4.x-green"/>
  <img src="https://img.shields.io/badge/Optuna-3.x-purple"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-red"/>
  <img src="https://img.shields.io/badge/Dataset-ESA%20CDMs-lightgrey"/>
  <img src="https://img.shields.io/badge/Status-Research%20Complete-brightgreen"/>
</p>

> **CSAI 801 Artificial Intel & Mach Learn · Group 14 · Queen's University · Winter 2026**  
> Mohamed Yahya · Mirna Embaby · Mahmoud Alyosify

---

## Overview

**SCRAP** is a supervised machine learning framework for predicting the final satellite collision risk at the Time of Closest Approach (TCA), using **only telemetry data available at least 48 hours in advance** — the minimum operational lead time required for maneuver planning.

The framework addresses two core challenges simultaneously:

- **Classification:** Identify whether a conjunction event will exceed the $10^{-6}$ high-risk threshold at TCA
- **Regression:** Accurately estimate the magnitude of risk for confirmed high-risk events

This decomposition mirrors the structure of the ESA official evaluation metric and is the architectural foundation of SCRAP's two-stage Sentinel–Specialist design.

---

## Results at a Glance

| Metric | Value |
|---|---|
| **F₂ Score** | **0.9464** |
| **Recall (High-Risk Events)** | **97.31%** |
| False Negatives | **9** / 334 true HR events |
| False Positives | 56 |
| Precision | 0.8530 |
| Specialist RMSE | **0.25 log₁₀ units** |
| Sentinel HR/LR Probability Separation | **26.6×** |
| LRP Naive Baseline | L = ∞ (F₂ = 0) |

> **Note:** The LRP baseline collapses because at t = 2 days, every observed risk is below 10⁻⁶. Any model beating this baseline has learned genuine predictive structure from pre-cutoff telemetry.

---

## Architecture

```
Raw CDM Time-Series (variable length per event)
            │
            ▼
┌───────────────────────────────────────────┐
│         2-Day Operational Cutoff          │
│   Drop all CDMs with time_to_tca < 2.0    │
│   Target ← last CDM in full sequence      │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│      Physics-Informed Feature Engine      │
│  • Mahalanobis Distance (covariance-aware)│
│  • Log-Covariance Determinant (both objs) │
│  • Uncertainty Volume                     │
│  • Orbital Geometry (apogee/perigee/inc)  │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│   Time-Series Flattening (11 stats/feat)  │
│  last · mean · std · min · max · delta    │
│  slope · last2_change · change_ratio      │
│  recent_vs_early · max_single_jump        │
│                                           │
│         → 1,208 features/event            │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│   STAGE 1: Sentinel (LightGBM Classifier) │
│   Objective: Maximise F₂                  │
│   Threshold: OOF-tuned (t* = 0.762)       │
│   Output: p(r ≥ 10⁻⁶) per event           │
└───────────────────────────────────────────┘
       │                    │
    p ≥ t*               p < t*
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│  STAGE 2:   │     │  Hard clip   │
│  Specialist │     │  → -6.001    │
│  (XGBoost   │     └──────────────┘
│  Regressor) │
│  HR-only    │
│  training   │
└─────────────┘
       │
       ▼
┌───────────────────────────────────────────┐
│      Borderline Promotion Module          │
│   If p ∈ [t*/2, t*) AND high uncertainty  │
│   → promote prediction to -5.99           │
│   (Jump-regime victim protection)         │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│        Global Bias Calibration            │
│   OOF-tuned scalar shift on all preds     │
└───────────────────────────────────────────┘
            │
            ▼
      Final Prediction
```



https://github.com/user-attachments/assets/8e383b17-3306-438d-b48f-6e6a33f082f7



---

## Repository Structure

```
SCRAP/
│
├── SCRAP_Satellite_Collision_Risk_Assessment_and_Prediction.ipynb   # ← Main notebook (full pipeline)
│
├── data/
│   └── (loaded automatically from HuggingFace — see Setup)
│
├── reports/
│   └── Project_Proposal__Satellite_Collision_Risk_Assessment_and_Prediction.pdf
│   └── Project_Proposal__Satellite_Collision_Risk_Assessment_and_Prediction.pdf
│
└── README.md                     # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended; CPU fallback is automatic)

### Install Dependencies

```bash
pip install xgboost lightgbm catboost optuna shap datasets scipy scikit-learn pandas numpy matplotlib
```

Or install all at once via the first notebook cell:

```python
!pip install xgboost lightgbm catboost optuna shap datasets scipy -q
```

### Dataset

The dataset is loaded automatically from HuggingFace. No manual download is required:

```python
from datasets import load_dataset
ds = load_dataset("mahmoudalyosify/SCRAP")
df_train = ds['train'].to_pandas()
df_test  = ds['test'].to_pandas()
```

> **Source:** ESA Historical Conjunction Data Messages (CDMs) Database  
> **Size:** 162,634 CDM records · 13,154 unique conjunction events · 103 features per CDM

---

## Running the Notebook

Open `SCRAP_FINAL_v10_fixed.ipynb` and run all cells in order. The notebook is fully self-contained and handles:

1. Hardware auto-detection (CUDA / MPS / CPU)
2. Dataset download and preprocessing
3. Physics feature engineering
4. Time-series flattening
5. Sentinel hyperparameter tuning (Optuna, 80 trials)
6. Specialist hyperparameter tuning (Optuna, 80 trials)
7. OOF threshold optimisation
8. Borderline promotion and bias calibration
9. Final evaluation with ablation table
10. SHAP explainability analysis
11. Jump-regime recall segmentation

> **Expected runtime:** ~9 hours on a CUDA GPU (80 + 80 Optuna trials × 5-fold CV × 11,942 events × 1,208 features). Reduce `N_TRIALS` to 25 for faster experimentation.

---

## Evaluation Metric

The official ESA competition loss:

$$L = \frac{1}{F_2} \times \text{MSE}_\text{HR}$$

where:
- $F_2$ is the F-beta score with $\beta = 2$ (recall weighted 4× over precision)
- $\text{MSE}_\text{HR}$ is the mean squared error on **high-risk events only**, computed in **probability space**
- Lower $L$ is better

For hyperparameter selection, we use the numerically stable composite score:

$$S = \frac{F_2}{1 + \text{MSE}_\text{HR}} \in [0, 1]$$

This avoids the division instability of $L$ when $F_2 \approx 0$ and is bounded, making it well-suited for Optuna maximisation.

---

## Key Design Decisions

### Why Two Stages?

The compound loss $L = (1/F_2) \times \text{MSE}_\text{HR}$ couples a classification objective with a regression objective. Training a single model to optimise both simultaneously is mathematically conflicted. SCRAP decomposes them:

- **Sentinel** learns the classification boundary ($r = 10^{-6}$) under F₂-weighted gradient
- **Specialist** learns the regression surface within the high-risk zone under RMSE gradient on HR-only data

### Why Dynamic Threshold?

At the 2-day boundary, raw model probabilities are naturally low ($\sim 0.001$). A fixed threshold of 0.5 classifies every event as low-risk (F₂ = 0). Sweeping the threshold on OOF predictions and selecting the value that maximises F₂ discovered $t^* = 0.762$, recovering +1.8% recall vs. default threshold.

### Why Calibrated Sample Weights?

Sample weights are derived from first principles rather than tuned as a hyperparameter:

$$W_\text{high} = \frac{N_\text{neg}}{N_\text{pos}} \times \beta^2 = 8.24 \times 4 \approx 32.94$$

The imbalance factor corrects class frequency; $\beta^2 = 4$ directly encodes the ESA metric's recall priority into the training objective.

### The Jump-Regime Problem

A subset of high-risk events undergoes covariance updates in the final 48 hours that are fundamentally unpredictable from pre-cutoff telemetry. Four momentum features target this regime:

| Feature | What It Captures |
|---|---|
| `risk_last2_change` | Instantaneous risk change at the 2-day boundary |
| `change_ratio` | Is the last CDM step anomalously large vs. history? |
| `recent_vs_early` | Medium-term risk drift (last 3 vs. first 3 CDMs) |
| `max_single_jump` | Largest single-step change in the observable series |

---

## Ablation Results

| Pipeline Stage | F₂ | Recall | FN |
|---|---|---|---|
| LRP Naive Baseline | 0.0000 | 0.0% | 334 |
| Sentinel (threshold = 0.5) | 0.9522 | 95.5% | 15 |
| + Dynamic Threshold (t* = 0.762) | 0.9464 | 97.3% | 9 |
| + Borderline Promotion | 0.9464 | 97.3% | 9 |
| **+ Global Bias (Final)** | **0.9464** | **97.3%** | **9** |

---

## SHAP Feature Importance (Top 5)

| Rank | Feature | Physical Meaning |
|---|---|---|
| 1 | `mahalanobis_distance_last` | Covariance-normalised spatial separation at 2-day boundary |
| 2 | `combined_sigma_t_last` | Transverse position uncertainty (along-track cross-section) |
| 3 | `t_log_cov_det_last` | Target covariance growth (jump-regime precursor) |
| 4 | `risk_last2_change` | Instantaneous risk momentum at the boundary |
| 5 | `f107_last` | Solar flux modulating atmospheric drag and covariance inflation |

---

## Known Limitations

| Limitation | Explanation |
|---|---|
| **Irreducible Bayes error** | The residual 2.69% miss rate (9 FN) likely reflects events that undergo covariance updates in the final 48 hours that no pre-cutoff telemetry can predict |
| **Test partition composition** | $L \approx 0$ is scale-dependent; ESA leaderboard comparison is invalid due to different test set curation |
| **Temporal non-stationarity** | Solar cycle variation shifts drag distributions; periodic retraining is required for sustained deployed performance |
| **Static 2-day boundary** | A production system should dynamically adjust the prediction horizon based on available CDM count and quality per event |

---

## Future Work

- **Uncertainty quantification** via quantile regression or NGBoost to report prediction intervals on borderline events
- **Transformer sequence modeling** on raw CDM streams to capture inter-CDM dynamics beyond single-step statistics
- **Adversarial validation** to detect deployment-time feature distribution shift before it degrades recall
- **Online learning** to adapt to solar cycle variation without full retraining

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{scrap2026,
  title   = {SCRAP: Satellite Collision Risk Assessment and Prediction},
  author  = {Yahya, Mohamed and Embaby, Mirna and Alyosify, Mahmoud},
  year    = {2026},
  note    = {CSAI 801 Project Report, Queen's University},
  url     = {https://github.com/mahmoudalyosify/SCRAP}
}
```

And the original ESA challenge dataset:

```bibtex
@article{uriot2020spacecraft,
  title   = {Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition},
  author  = {Uriot, T. and Izzo, D. and Sim{\~o}es, L. F. and others},
  journal = {arXiv preprint arXiv:2008.03069},
  year    = {2020}
}
```

---

## References

1. ESA Space Debris Office. *Conjunction Surveillance and Collision Avoidance*. ESA Space Safety Programme, 2020.
2. Chan, F. K. *Spacecraft Collision Probability*. The Aerospace Press, 2008.
3. Uriot, T., Izzo, D., Simões, L. F., et al. *Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition*. arXiv:2008.03069, 2020.
4. Alyosify, M. *SCRAP ESA CDM Dataset*. HuggingFace Datasets, 2026. https://huggingface.co/datasets/mahmoudalyosify/SCRAP

---

<p align="center">
  Made with ☕ and orbital mechanics · Queen's University · Winter 2026
</p>
