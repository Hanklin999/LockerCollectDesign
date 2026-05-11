# 📦 Causal Inference in Last-Mile Logistics
### Do Push Notifications Actually Change When People Pick Up Their Parcels?

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)
[![Methods](https://img.shields.io/badge/Methods-DiD%20%7C%20PSM%20%7C%20HTE%20%7C%20Sensitivity-6366F1?style=flat)]()
[![Data](https://img.shields.io/badge/Data-Calibrated%20Simulation-F59E0B?style=flat)]()

---

## Overview

This project applies a full causal inference pipeline to evaluate the impact of **push notification strategies** on parcel pickup efficiency across 2,000 smart locker locations.

The core question is deceptively simple:

> *Stores with more notification touches show faster pickup times — but is that because the notifications work, or because those stores were already different to begin with?*

This project answers that question rigorously using four complementary methods: **Difference-in-Differences**, **Propensity Score Matching**, **Heterogeneous Treatment Effects**, and **Sensitivity Analysis**.

---

## Background

Smart locker networks face a critical operational tension: parcels left uncollected tie up locker capacity, reduce throughput, and increase Return-to-Sender (RTS) rates. Push notifications are the primary lever to drive timely pickup — but designing the right notification strategy requires understanding *causal* effects, not just correlations.

This project simulates a nationwide experiment modelled on real operational constraints:

- **2,000 smart locker stores** across Taiwan
- **800,000+ parcels per day** at peak
- **5-day pickup deadline** for high-inventory (burst) stores
- **Three notification cadences** tested simultaneously

**Data**: Calibrated simulation. Parameters (collection hours, RTS rates, treatment effect sizes, store inventory distributions) are grounded in real operational observations. Raw company data is not used for confidentiality reasons.

---

## Experiment Design

```
1,000 burst stores (high inventory) → randomly assigned to:

  ┌─────────────────────────────────────────────────────────┐
  │  5D Control  │  D0 + D4          │  2 touches │ n=100  │
  │  5D G2       │  D0 + D2 + D4     │  3 touches │ n=100  │
  │  5D G4       │  D0–D4 (all)      │  5 touches │ n=100  │
  │  6D          │  D0 + D5          │  2 touches │ n=700  │
  └─────────────────────────────────────────────────────────┘

1,000 vacant stores (low inventory) → 7D, D0 + D6, 2 touches

Timeline:  BM week (2026-01-19) → 4 experiment weeks
           CNY week (2026-02-09) excluded: seasonal confound
```

**Primary outcome**: `collection_hrs` — hours from parcel arrival to pickup  
**Secondary outcome**: `rts_rate` — proportion of parcels returned to sender  
**Guardrail metrics**: `complaint_rate`, `opt_out_rate`

---

## Methods

| Notebook | Method | Research Question | Key Output |
|----------|--------|-------------------|------------|
| `00` | EDA + DAG | What does the data look like? What are the confounders? | Causal DAG, covariate distributions |
| `01` | Data Preparation | How is the simulation calibrated? | `store_panel.csv`, `store_metadata.csv` |
| `02` | Difference-in-Differences | Did the policy change *cause* faster pickup, controlling for time trends? | DiD coefficient, event study plot |
| `03` | Propensity Score Matching | After balancing store characteristics, what is the treatment effect? | ATT, Love plot, PS overlap |
| `04` | Heterogeneous Treatment Effects | Which stores benefit most? | CATE by subgroup, feature importance |
| `05` | Sensitivity Analysis | How robust is the result to hidden confounding? | Rosenbaum Γ, placebo tests, LOO |

---

## Key Findings

### 1. Notification cadence reduces collection time — but with diminishing returns

| Strategy | Avg Collection Hours | vs Control | Significant |
|----------|---------------------|------------|-------------|
| Control (2 touches) | 33.5h | — | — |
| G2 (3 touches) | ~32.0h | **−1.5h** | ✅ p < 0.05 |
| G4 (5 touches) | ~31.8h | **−1.7h** | ✅ p < 0.05 |
| G4 − G2 (marginal) | — | **−0.2h** | ❌ Not significant |

**Implication**: 3 touches captures most of the benefit. The 4th and 5th notification contribute negligible additional reduction at non-trivial cost and opt-out risk.

### 2. PSM confirms the DiD result

After matching stores on utilization rate, daily volume, capacity, and metro status, the ATT is consistent with the DiD estimate. This rules out simple selection bias as an explanation.

### 3. Treatment effects are heterogeneous

High-traffic metro stores show larger effects than low-traffic regional stores. This supports a targeted rollout strategy: upgrade notification cadence for burst metro stores first.

### 4. Results are robust to hidden confounding

Rosenbaum sensitivity bounds show the result holds against hidden confounders up to **Γ ≈ 1.8** — a confounder would need to make treated stores 1.8× more likely to be selected before it could explain away the finding.

---

## Repository Structure

```
causal-inference-locker-notifications/
│
├── data/
│   ├── processed/
│   │   ├── store_metadata.csv        # Static store characteristics
│   │   ├── store_panel.csv           # Weekly panel (store × week)
│   │   └── store_panel_highrisk.csv  # >96hr uncollected parcel cohort
│   └── README.md                     # Data dictionary
│
├── notebooks/
│   ├── 00_EDA_and_DAG.ipynb
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_DiD.ipynb
│   ├── 03_PSM.ipynb
│   ├── 04_HTE.ipynb
│   └── 05_Sensitivity_Analysis.ipynb
│
├── src/
│   ├── data_generation.py    # Calibrated DGP
│   ├── did.py                # DiD estimators
│   ├── psm.py                # PSM pipeline
│   ├── hte.py                # Meta-learners + Causal Forest DML
│   ├── sensitivity.py        # Rosenbaum bounds, placebo tests, LOO
│   └── visualization.py      # All plot functions (unified style)
│
├── outputs/
│   └── figures/              # All saved plots
│
├── requirements.txt
└── README.md
```

---

## Visualizations

| Plot | Method | What it shows |
|------|--------|---------------|
| Causal DAG | EDA | Confounders, treatment path, hidden bias risk |
| Collection hours trend | EDA | Weekly outcome by group, BM baseline |
| Parallel trends | DiD | Pre-treatment trend validation |
| Event study | DiD | Dynamic treatment effect trajectory |
| Coefficient stability | DiD | Result across 5 specifications |
| PS overlap | PSM | Common support before/after matching |
| Love plot | PSM | SMD balance before/after matching |
| ATT estimates | PSM | Point estimate + CI across comparisons |
| CATE distribution | HTE | Heterogeneity across meta-learners |
| Subgroup waterfall | HTE | Treatment effect by store type/location |
| Feature importance | HTE | What drives heterogeneity |
| Rosenbaum bounds | Sensitivity | Hidden confounding threshold |
| LOO robustness | Sensitivity | Week-by-week influence |
| Placebo comparison | Sensitivity | True vs placebo outcome effects |

---

## Setup

```bash
git clone https://github.com/yourusername/causal-inference-locker-notifications
cd causal-inference-locker-notifications

pip install -r requirements.txt

# Generate simulated data
python src/data_generation.py

# Run notebooks in order
jupyter notebook notebooks/
```

**requirements.txt**
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
statsmodels>=0.14
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## Methodology Notes

### Why simulation?

Real experimental data from the locker network is confidential. This simulation uses **calibrated parameters** — effect sizes, variance, store distributions, and seasonal patterns — grounded in real operational observations. The Data Generating Process (DGP) is fully documented in `src/data_generation.py` and `notebooks/01_Data_Preparation.ipynb`.

This approach lets us:
1. Demonstrate the full causal inference pipeline on a realistic problem
2. Validate estimators against known ground truth
3. Show how DGP assumptions affect conclusions (sensitivity)

### Why do we need causal inference if stores were randomly assigned?

Random assignment (within the 5D group) justifies a simple comparison for the 5D arms. But:

- The **6D group** was assigned by store inventory level, not randomly → PSM needed
- Even with randomisation, **store FE** in DiD removes residual covariate imbalance
- **HTE** requires meta-learners regardless of assignment mechanism
- **Sensitivity analysis** is always warranted with observational elements

### Limitations

- One pre-period (BM week) limits the parallel trends test
- Self-report: collection hours are system-logged, but store categorisation relies on historical thresholds which may shift
- External validity: results apply to burst-store conditions; vacant store dynamics differ
- CNY exclusion is justified but reduces post-treatment sample by one week

---

## About

Built as a portfolio project demonstrating applied causal inference for product and experimentation science roles.

**Methods**: Difference-in-Differences · Propensity Score Matching · T/S/X-Learner · Causal Forest DML · Rosenbaum Bounds  
**Tools**: Python · statsmodels · scikit-learn · matplotlib  
**Certificate**: HarvardX — Causal Inference (verified)  
**Industry context**: Based on experimentation work in last-mile logistics at a Southeast Asian e-commerce platform

---

*Questions or feedback? Open an issue or reach out via [LinkedIn](https://linkedin.com/in/yourprofile).*
