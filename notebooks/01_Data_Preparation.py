# %% [markdown]
# # 01 · Data Preparation & DGP Documentation
#
# **Goal**: Document the Data Generating Process (DGP) transparently,
# generate the simulation datasets, and validate that the output
# matches the real operational parameters it was calibrated to.
#
# **Why simulation?**
# Real locker network data is confidential. This simulation uses
# calibrated parameters grounded in real operational observations:
# effect sizes, variance, store distributions, and seasonal patterns
# all reflect what was actually observed. The DGP is fully auditable here.
#
# **Calibration sources**:
# - Collection hours by notification group: real weekly experiment data
# - RTS rates: real experiment data (2025-10-27 to 2026-03-02)
# - Store count by city: actual Taiwan store distribution
# - Burst store definition: pct_closure_hours > 5% (real operational rule)
#
# ---

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.append("../src")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from data_generation import (
    generate_store_metadata,
    assign_treatment,
    generate_store_panel,
    generate_highrisk_cohort,
    generate_all_data,
    TAIWAN_CITIES,
    TREATMENT_GROUPS,
    COLLECTION_HRS_PARAMS,
    RTS_PARAMS,
    EXPERIMENT_WEEKS,
)
from visualization import set_style

set_style()
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../outputs/figures", exist_ok=True)

# %% [markdown]
# ## 1. DGP Overview
#
# The simulation has four layers:
#
# ```
# Layer 1: Store Metadata (static)
#     2,000 stores with characteristics drawn from real Taiwan distributions
#     Confounders: utilization rate, daily volume, capacity, city, region
#
# Layer 2: Treatment Assignment (selection on observables)
#     Burst stores (pct_closure_hours > 5%) → randomly split into 5D/6D groups
#     Vacant stores → all assigned to 7D
#     Assignment is NOT fully random → PSM needed
#
# Layer 3: Weekly Panel Outcomes
#     Collection hours = baseline + treatment_effect × post + store_FE + noise
#     Treatment effects calibrated to real observed data
#     One excluded week (CNY) with seasonal adjustment
#
# Layer 4: High-Risk Cohort
#     15% of parcels uncollected >96hrs → separate outcome distribution
#     D4 notification intervention effect modelled separately
# ```

# %% [markdown]
# ## 2. DGP Parameters (Calibration Table)

# %%
print("=" * 65)
print("CALIBRATION PARAMETERS")
print("=" * 65)

print("\nCollection Hours (calibrated from real BM week 2026-01-19):")
for group, params in COLLECTION_HRS_PARAMS.items():
    print(f"  {group:15s}: baseline={params['mean']}h  "
          f"effect={params['treatment_effect']:+.1f}h  "
          f"→ expected post={params['mean']+params['treatment_effect']:.1f}h")

print("\nRTS Rate:")
for group, params in RTS_PARAMS.items():
    print(f"  {group:15s}: baseline={params['mean']*100:.2f}%  "
          f"effect={params['treatment_effect']*100:+.3f}pp")

print("\nExperiment Timeline:")
for week in EXPERIMENT_WEEKS:
    label = "BM (pre-treatment)" if week["is_bm"] else "Experiment week"
    print(f"  {week['date']}  week_id={week['week_id']}  {label}")

print("\nNote: 2026-02-09 (CNY week) excluded from experiment timeline.")
print("      Seasonal effect: buyers collect faster before Chinese New Year.")

# %% [markdown]
# ## 3. Generate Data

# %%
metadata, panel, highrisk = generate_all_data(
    n_stores=2000,
    seed=42,
    output_dir="../data/processed",
)

# %% [markdown]
# ## 4. Validate Store Metadata

# %%
print("Store metadata shape:", metadata.shape)
print("\nStore type distribution:")
print(metadata["store_type"].value_counts())

# %%
print("\nTreatment group distribution:")
tg_dist = metadata["treatment_group"].value_counts().sort_index()
print(tg_dist)

# %%
# City distribution vs real Taiwan data
print("\nCity distribution (simulated vs real):")
city_counts = metadata["city"].value_counts()
city_compare = pd.DataFrame({
    "simulated":  city_counts,
    "real_total": {c: v["count"] for c, v in TAIWAN_CITIES.items()},
}).fillna(0).astype(int)
print(city_compare.head(10))

# %%
# Utilization rate distribution (key confounder)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, label in zip(
    axes,
    ["avg_utilization_rate", "avg_daily_volume"],
    ["Utilization Rate", "Daily Volume (pkgs/day)"],
):
    for stype, grp in metadata.groupby("store_type"):
        ax.hist(grp[col], bins=30, alpha=0.55,
                label=stype.capitalize(), edgecolor="white")
    ax.set_title(f"{label} by Store Type")
    ax.set_xlabel(label)
    ax.legend()

plt.suptitle("Key Confounder Distributions", fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/00_confounder_distributions.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Validate Panel Outcomes
#
# The post-period mean collection hours should match the calibration targets.

# %%
post = panel[panel["is_post"] == 1]
print("Post-period collection hours vs calibration targets:")
print(f"{'Group':15s}  {'Simulated':>10}  {'Target':>10}  {'Diff':>8}")
print("-" * 50)
for group, params in COLLECTION_HRS_PARAMS.items():
    target = params["mean"] + params["treatment_effect"]
    sim = post[post["treatment_group"] == group]["collection_hrs"].mean()
    if not pd.isna(sim):
        diff = sim - target
        print(f"{group:15s}  {sim:>10.3f}  {target:>10.3f}  {diff:>+8.3f}")

# %%
# Check BM week values match baseline
bm = panel[panel["is_bm"] == 1]
print("\nBM week collection hours vs calibration baseline (33.5h for 5D groups):")
print(
    bm[bm["treatment_group"].isin(["5D_Control", "5D_G2", "5D_G4"])]
    .groupby("treatment_group")["collection_hrs"]
    .mean()
    .round(3)
)

# %% [markdown]
# ## 6. Validate Treatment Effect Direction

# %%
# Check that more touches = lower collection hours
group_order = ["5D_Control", "5D_G2", "5D_G4"]
post_means = post[post["treatment_group"].isin(group_order)].groupby("treatment_group")["collection_hrs"].mean()
print("Collection hours by group (post-period):")
for g in group_order:
    if g in post_means:
        print(f"  {g:15s}: {post_means[g]:.3f}h")

print("\nExpected ordering: Control > G2 > G4 (more touches = faster pickup)")
is_ordered = post_means["5D_Control"] > post_means["5D_G2"] > post_means["5D_G4"]
print(f"Ordering correct: {is_ordered}")

# %%
# Visualise panel outcomes
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Collection hours
weekly = (
    panel[panel["treatment_group"].isin(["5D_Control", "5D_G2", "5D_G4"])]
    .groupby(["date", "treatment_group"])["collection_hrs"]
    .mean()
    .reset_index()
)
palette = {"5D_Control": "#4878CF", "5D_G2": "#6ACC65", "5D_G4": "#D65F5F"}
for group, gdf in weekly.groupby("treatment_group"):
    axes[0].plot(gdf["date"], gdf["collection_hrs"], marker="o",
                 color=palette[group], linewidth=2, label=group)
axes[0].set_title("Collection Hours Over Time")
axes[0].set_xlabel("Week")
axes[0].set_ylabel("Avg Hours")
axes[0].legend()
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

# RTS rate
weekly_rts = (
    panel[panel["treatment_group"].isin(["5D_Control", "5D_G2", "5D_G4"])]
    .groupby(["date", "treatment_group"])["rts_rate"]
    .mean()
    .reset_index()
)
for group, gdf in weekly_rts.groupby("treatment_group"):
    axes[1].plot(gdf["date"], gdf["rts_rate"] * 100, marker="s",
                 color=palette[group], linewidth=2, label=group)
axes[1].set_title("RTS Rate (%) Over Time")
axes[1].set_xlabel("Week")
axes[1].set_ylabel("RTS Rate (%)")
axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)

plt.suptitle("Simulated Outcomes: Calibration Validation", fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/00b_outcome_validation.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. High-Risk Cohort

# %%
print("High-risk cohort (>96hr uncollected parcels):")
print(f"  Rows              : {len(highrisk):,}")
print(f"  % of all parcels  : ~15% (by design)")
print(f"  Avg collection hrs: {highrisk['collection_hrs'].mean():.1f}h")
print(f"  Avg RTS rate      : {highrisk['rts_rate'].mean()*100:.2f}%")

# %%
# D4 intervention effect on high-risk cohort
hr_post = highrisk[highrisk["is_post"] == 1]
print("\nD4 intervention effect on high-risk cohort (post-period):")
for group in ["5D_Control", "5D_G2", "5D_G4"]:
    sub = hr_post[hr_post["treatment_group"] == group]
    if len(sub) > 0:
        print(f"  {group:15s}: {sub['collection_hrs'].mean():.1f}h  "
              f"RTS={sub['rts_rate'].mean()*100:.2f}%")

# %% [markdown]
# ## 8. Data Dictionary

# %%
print("""
DATA DICTIONARY
===============

store_metadata.csv (one row per store, static)
  store_id              : unique store identifier (0–1999)
  city                  : Taiwan city name
  region_type           : metro / regional / rural
  is_metro              : 1 if six major cities (六都), else 0
  capacity              : max parcel capacity (pkgs), ~N(400, 40)
  avg_daily_volume      : historical avg daily parcels
  avg_utilization_rate  : daily_volume / capacity
  pct_closure_hours     : fraction of day store closes new orders
  store_type            : burst (high inventory) / vacant (low inventory)
  treatment_group       : 5D_Control / 5D_G2 / 5D_G4 / 6D / 7D

store_panel.csv (one row per store per week)
  store_id              : matches store_metadata
  week_id               : 0=BM, 1–4=experiment weeks
  date                  : calendar date of week
  is_bm                 : 1 if BM (pre-treatment) week, else 0
  is_post               : 1 if post-treatment week, else 0
  treatment_group       : same as metadata
  store_type            : same as metadata
  [store characteristics]: same as metadata (denormalised for convenience)
  collection_hrs        : PRIMARY OUTCOME — hours from arrival to pickup
  rts_rate              : SECONDARY OUTCOME — fraction of parcels returned
  complaint_rate        : GUARDRAIL — notification complaint rate
  opt_out_rate          : GUARDRAIL — notification unsubscribe rate
  is_treated_5d         : 1 if 5D_G2 or 5D_G4, used for DiD
  is_g2 / is_g4         : group indicators for multi-arm DiD
  cohort                : 'normal' (used in sensitivity join)

store_panel_highrisk.csv
  Same schema as store_panel.csv, but for parcels uncollected >96 hours.
  collection_hrs here starts at 96h+ (mean ~123h before D4 intervention).
  rts_rate is ~15–18% for this cohort (vs ~1.5% overall).
""")

# %% [markdown]
# ## 9. Reproduce the Data
#
# To regenerate all datasets from scratch with a different seed:

# %%
# Uncomment to regenerate:
# metadata, panel, highrisk = generate_all_data(
#     n_stores=2000,
#     seed=123,            # change seed for different random draw
#     output_dir="../data/processed",
# )

print("Data generation complete.")
print("Files saved to ../data/processed/")
print("\nNext: Run notebooks in order:")
print("  00_EDA_and_DAG → 02_DiD → 03_PSM → 04_HTE → 05_Sensitivity")
