# 📦 How Many Notifications Should We Send?

### A Product Analytics Case Study on Notification Strategy and Parcel Pickup Behavior

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)
[![Methods](https://img.shields.io/badge/Methods-DiD%20%7C%20PSM%20%7C%20HTE%20%7C%20Sensitivity-6366F1?style=flat)]()
[![Data](https://img.shields.io/badge/Data-Calibrated%20Simulation-F59E0B?style=flat)]()

---

## TL;DR

**Problem**: Locker congestion at high-inventory stores reduces network throughput and increases return-to-sender rates.

**Question**: How many pickup-reminder notifications should we send, and to which stores?

**Finding**: Increasing notification touches from 2 to 3 reduces parcel collection time by **1.2–1.7 hours**, validated by two independent causal methods. A 4th and 5th touch add only **0.2–0.6 additional hours** — diminishing returns.

**Recommendation**: Roll out the 3-touch strategy network-wide, prioritizing non-metro and high-capacity stores where the effect is largest.

**Impact**: At 800K parcels/day, this reduces return-to-sender volume by an estimated **2,880 parcels/day** and frees locker capacity faster, improving throughput without added headcount.

---

## The Business Problem

Smart locker networks have a fixed asset: locker slots. When buyers don't collect parcels promptly, slots stay occupied, new parcels can't be processed, and stores hit capacity ("burst") — triggering order pauses and operational firefighting.

Push notifications are the cheapest lever available to drive faster pickup. But more notifications also mean more cost and a real risk of notification fatigue (complaints, opt-outs). The team needed a number: **what's the right number of touches?**

---

## Product Metrics Framework

| Tier | Metric | Why it matters |
|------|--------|-----------------|
| **North Star** | `collection_hrs` — hours from parcel arrival to pickup | Directly drives locker turnover and capacity availability |
| **Secondary** | `rts_rate` — % of parcels returned to sender | Captures the tail-risk failure mode of slow pickup |
| **Guardrail** | `complaint_rate` | Detects notification fatigue |
| **Guardrail** | `opt_out_rate` | Detects long-term channel erosion |

A strategy is only "good" if it moves the North Star **without** breaching the guardrails.

---

## Experiment Design

**Unit of randomization**: Store (not buyer) — notifications are configured at the store/locker level, so store-level randomization avoids cross-contamination between buyers at the same location.

```
1,000 burst stores (high inventory, >5% order-closure hours)
  → randomly split into:

  ┌──────────────────────────────────────────────────────────┐
  │  Control │ 5-day deadline │ D0, D4          │ 2 touches │ n=100 │
  │  G2      │ 5-day deadline │ D0, D2, D4      │ 3 touches │ n=100 │
  │  G4      │ 5-day deadline │ D0–D4 (daily)   │ 5 touches │ n=100 │
  │  6D      │ 6-day deadline │ D0, D5          │ 2 touches │ n=700 │
  └──────────────────────────────────────────────────────────┘

1,000 vacant stores (low inventory)
  → 7D: 7-day deadline, D0/D6, 2 touches (pilot-only comparison arm)

Timeline: 1 baseline week + 4 experiment weeks
Excluded: Chinese New Year eve week (seasonal pickup-speed confound)
```

Pilot sizing (100/100/100 vs 700/1000) reflects a real-world constraint: full network rollout wasn't feasible up front, so the team ran a smaller controlled pilot on the 5-day groups before scaling the deadline-only change to more stores. This pilot-then-scale pattern is a common compromise in operational experimentation.

---

## Decision Memo: Which Cadence Should We Ship?

| Option | Collection Time Impact | Cost | Recommendation |
|--------|------------------------|------|-----------------|
| **A — Keep 2 touches** | Baseline (33.7h) | Lowest | ❌ Leaves easy gains on the table |
| **B — 3 touches (G2)** | **−1.2 to −1.5h** (DiD / PSM) | +1 notification/parcel | ✅ **Ship this** |
| **C — 5 touches (G4)** | −1.8 to −1.9h | +3 notifications/parcel | ❌ Marginal gain (+0.3–0.6h over B) doesn't justify 3x the notification cost or opt-out risk |

**Decision: Ship Option B (3-touch cadence) network-wide.**

The 4th and 5th touches in Option C are not statistically distinguishable from Option B's *marginal* contribution once cost and guardrail risk are weighed — classic diminishing returns.

---

## Where Should We Roll Out First?

Heterogeneity analysis (4 independent estimators — T/S/X-Learner, Causal Forest DML) shows the effect is **not uniform** across stores. Rolling out everywhere at once is fine here (95% of stores benefit), but if rollout needs to be staged, prioritize by these findings:

| Segment | Effect Size | Take First? |
|---------|-------------|-------------|
| **Non-metro stores** | **−2.4h** | ✅ Yes — largest effect |
| Metro stores | −1.7h | Standard priority |
| Low-utilization stores | −2.1h | ✅ Yes |
| High-utilization stores | −1.6h | Standard priority |

**Counter-intuitive finding**: notification reminders work *better* in non-metro stores, not metro ones. Buyers there appear to have weaker baseline pickup habits, so a nudge moves the needle more. Geographic location explains almost none of the variation (feature importance: 0.003); **store capacity** is the dominant driver (feature importance: 1.33, 5x the next variable) — bigger stores, where pickup delay is more costly operationally, also see larger behavioral response to reminders.

**95% of stores (285/300)** show a meaningful individual effect (CATE < −0.5h), so a full rollout is justified without complex targeting logic. If budget is constrained, non-metro and high-capacity stores offer the best marginal ROI.

---

## Why We Can't Just Compare Group Averages

A naive comparison (3-touch stores vs. 2-touch stores) shows a **−1.75h gap**. But stores weren't randomly given a notification policy in a vacuum — store characteristics (inventory pressure, traffic, capacity) co-determine both the policy a store receives *and* its baseline pickup speed. Comparing raw averages would conflate "the notification worked" with "this store was always going to collect faster."

We address this with two independent causal designs that should agree if the effect is real:

**1. Difference-in-Differences** — removes store-level and time-level confounds by comparing the *change* in collection time for treated vs. control stores, before vs. after the policy change.

**2. Propensity Score Matching** — pairs each treated store with a statistically similar control store (matched on utilization rate, volume, capacity, metro status) before comparing outcomes.

| Method | Estimated Effect | 95% CI |
|--------|-------------------|--------|
| DiD (two-way fixed effects) | **−1.505h** | [−1.97, −1.04] |
| PSM (ATT, matched) | **−1.654h** | [−1.88, −1.43] |
| **Agreement** | within 9% of each other | — |

Two methods built on different assumptions land within 9% of each other — strong evidence the effect is real, not an artifact of store selection.

---

## Is This Result Trustworthy? (Sensitivity Checks)

| Check | Question Asked | Result |
|-------|------------------|--------|
| **Permutation test** | Could random chance produce this result? | 500 random reshuffles never came close (p < 0.001) |
| **Rosenbaum bounds** | How strong would an unmeasured confounder need to be to overturn this? | Holds up to **Γ ≥ 3.0** — far beyond the conventional Γ > 1.5 robustness bar |
| **Leave-one-week-out** | Is one unusual week driving the whole result? | Estimate varies by at most 0.1h when any single week is dropped |
| **Covariate stability** | Does adding more controls change the answer? | 0% change from simplest to fullest model specification |
| **Placebo outcomes** | Does the "effect" show up where it shouldn't? | Small mechanical increase in complaint/opt-out rate (expected — more touches = more messages), no evidence of broader confounding |

**Bottom line**: this is one of the more robust findings you'll see in an applied experiment — it survives every standard stress test.

---

## Business Impact Translation

| Statistical Result | Operational Translation |
|----------------------|---------------------------|
| −1.5h average collection time | Faster slot turnover → more locker capacity available per day without new hardware |
| −0.36pp RTS rate | ≈ **2,880 fewer returned parcels/day** at 800K parcels/day volume |
| Diminishing returns beyond 3 touches | Avoids ~40% extra notification spend (5 vs 3 touches) for <0.3h of additional benefit |
| 95% of stores benefit | No targeting infrastructure needed — simple network-wide rollout |

---

## Methods Reference

| Notebook | Method | Used To Answer |
|----------|--------|-----------------|
| `00` | EDA + Causal DAG | What confounds the naive comparison? |
| `01` | Data Preparation | How is the simulation calibrated to real ops data? |
| `02` | Difference-in-Differences | Did the policy *cause* faster pickup? |
| `03` | Propensity Score Matching | Does the result hold after balancing store characteristics? |
| `04` | Heterogeneous Treatment Effects | Where should we prioritize rollout? |
| `05` | Sensitivity Analysis | How much do we trust this? |

Full statistical detail, tables, and method documentation: see [**RESULTS.md**](RESULTS.md) (English) and [**RESULTS_zh.md**](RESULTS_zh.md) (繁體中文).

---

## Data Note

Real locker-network data is confidential. This project uses a **calibrated simulation**: every parameter (baseline collection time, effect sizes, store volume distribution, seasonal patterns) is set to match values actually observed in a real nationwide notification experiment. The full data-generating process is documented and auditable in `src/data_generation.py` and `notebooks/01_Data_Preparation.ipynb`.

---

## Repository Structure

```
LockerCollectDesign/
│
├── data/processed/          # Simulated store metadata + weekly panel
├── notebooks/                # 00–05, run in order
├── src/                       # Reusable estimator + plotting modules
├── outputs/figures/          # All 19 generated plots
├── README.md                 # You are here
├── RESULTS.md                 # Full technical writeup (English)
├── RESULTS_zh.md              # Full technical writeup (繁體中文)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Hanklin999/LockerCollectDesign
cd LockerCollectDesign
pip install -r requirements.txt

python src/data_generation.py        # generate simulated data
python -m jupytext --to notebook notebooks/*.py
jupyter notebook notebooks/          # run 01 → 00 → 02 → 03 → 04 → 05
```

---

## Limitations

- Single pre-treatment week limits formal pre-trend testing (mitigated by confirming baseline balance and an immediate, stable post-treatment effect)
- Simulated data — no novel real-world discovery; the goal is to demonstrate the analytical pipeline with operationally realistic parameters
- Small per-arm sample (n=100) leaves residual covariate imbalance after randomization, addressed via PSM
- Excluded one experiment week (Chinese New Year) for a valid seasonal confound; leave-one-out testing confirms this doesn't drive the result

---

## About

**Methods**: Difference-in-Differences · Propensity Score Matching · T/S/X-Learner · Causal Forest DML · Rosenbaum Bounds
**Tools**: Python · statsmodels · scikit-learn · matplotlib
**Background**: HarvardX Causal Inference (verified); 2.5 years designing and analyzing nationwide A/B experiments in e-commerce logistics

*Built as a portfolio project for Product Analytics / Experimentation Scientist roles.*
