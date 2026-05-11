# %% [markdown]
# # 00 · EDA & Causal DAG
#
# **Goal**: Understand the data structure, identify confounders,
# and formalise the causal assumptions as a DAG before any modelling.
#
# **Key questions**:
# - What does the outcome distribution look like across groups?
# - Are stores systematically different across treatment arms?
# - What is the assumed causal structure?
#
# ---

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from visualization import (
    plot_collection_hrs_trend,
    plot_rts_trend,
    plot_store_characteristics,
    plot_dag,
    set_style,
)

set_style()

# %%
# Load data
panel    = pd.read_csv("../data/processed/store_panel.csv")
metadata = pd.read_csv("../data/processed/store_metadata.csv")

print(f"Panel shape     : {panel.shape}")
print(f"Metadata shape  : {metadata.shape}")
print(f"\nWeeks in panel  : {sorted(panel['date'].unique())}")
print(f"Treatment groups: {panel['treatment_group'].unique()}")

# %% [markdown]
# ## 1. Data Overview

# %%
# Treatment group counts
print("Store count by treatment group:")
print(
    panel.groupby("treatment_group")["store_id"]
    .nunique()
    .rename("n_stores")
    .to_frame()
)

# %%
# Outcome distributions by group (post-period only)
post = panel[panel["is_post"] == 1]
print("\nCollection hours by group (post-period):")
print(
    post.groupby("treatment_group")["collection_hrs"]
    .agg(["mean", "std", "min", "max"])
    .round(2)
)

# %%
# RTS rate by group
print("\nRTS rate (%) by group (post-period):")
print(
    (post.groupby("treatment_group")["rts_rate"].mean() * 100)
    .round(3)
    .rename("rts_rate_pct")
    .to_frame()
)

# %% [markdown]
# ## 2. Outcome Trends Over Time

# %%
fig = plot_collection_hrs_trend(
    panel,
    groups=["5D_Control", "5D_G2", "5D_G4"],
    save_path="../outputs/figures/01_collection_hrs_trend.png",
)
plt.show()

# %%
fig = plot_rts_trend(
    panel,
    groups=["5D_Control", "5D_G2", "5D_G4"],
    save_path="../outputs/figures/02_rts_trend.png",
)
plt.show()

# %% [markdown]
# **Observation**:
# - G2 and G4 show lower collection hours than Control post-treatment
# - G4 shows minimal additional improvement over G2 → diminishing returns
# - RTS rate mirrors collection hours pattern

# %% [markdown]
# ## 3. Store Characteristics (Confounder Check)

# %%
fig = plot_store_characteristics(
    metadata,
    save_path="../outputs/figures/03_store_characteristics.png",
)
plt.show()

# %%
# Check if burst stores are concentrated in metro areas
print("Store type by region:")
print(
    pd.crosstab(
        metadata["store_type"],
        metadata["region_type"],
        normalize="index",
    ).round(3)
)

# %%
# Compare key confounders across treatment groups
confounder_cols = [
    "avg_utilization_rate", "avg_daily_volume",
    "capacity", "pct_closure_hours", "is_metro",
]

five_d = panel[
    panel["treatment_group"].isin(["5D_Control", "5D_G2", "5D_G4"])
    & (panel["is_bm"] == 1)
].drop_duplicates("store_id")

print("\nConfounder balance across 5D groups (BM week, store-level):")
print(
    five_d.groupby("treatment_group")[confounder_cols]
    .mean()
    .round(4)
)

# %% [markdown]
# **Note**: The 5D groups were randomly assigned within burst stores,
# so we expect reasonable balance. Small imbalances remain by chance —
# PSM in notebook 03 will formally test and adjust for this.

# %% [markdown]
# ## 4. Causal DAG

# %%
fig = plot_dag(save_path="../outputs/figures/04_dag.png")
plt.show()

# %% [markdown]
# ### DAG interpretation
#
# **Treatment path**:
# `Historical Inventory → Treatment Assignment → Notification Frequency → Collection Hours`
#
# **Confounders** (affect both treatment assignment AND outcome):
# - `avg_utilization_rate` — busy stores get more aggressive notification strategy
# - `avg_daily_volume` — high-volume stores have different buyer behaviour
# - `is_metro` — metro buyers may respond differently to notifications
# - `capacity` — larger stores have more slack, different urgency
#
# **Hidden bias risk** (dashed red):
# - Unobserved store traits (e.g. local buyer demographics, store staff engagement)
#   could confound even after controlling for observed covariates
# - Addressed in notebook 05 via Rosenbaum bounds
#
# **Identifying assumption for DiD**:
# In the absence of treatment, treated and control stores would have followed
# the same time trend in collection hours (parallel trends).

# %% [markdown]
# ## 5. Pre-Treatment Balance Summary

# %%
from sklearn.preprocessing import StandardScaler

bm_stores = panel[panel["is_bm"] == 1].drop_duplicates("store_id")
five_d_bm = bm_stores[bm_stores["treatment_group"].isin(
    ["5D_Control", "5D_G2", "5D_G4"]
)]

# SMD between G2 and Control in BM week
def quick_smd(df, col, group_a, group_b, group_col="treatment_group"):
    a = df[df[group_col] == group_a][col]
    b = df[df[group_col] == group_b][col]
    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0

print("Pre-treatment SMD (BM week):")
for col in confounder_cols:
    smd_g2 = quick_smd(five_d_bm, col, "5D_G2", "5D_Control")
    smd_g4 = quick_smd(five_d_bm, col, "5D_G4", "5D_Control")
    print(f"  {col:30s}  G2 vs Control: {smd_g2:.3f}   G4 vs Control: {smd_g4:.3f}")

# %% [markdown]
# SMD < 0.1 indicates good pre-treatment balance.
# Values close to 0 confirm the random assignment within burst stores worked.

# %% [markdown]
# ---
# ## Summary
#
# | Finding | Implication |
# |---------|-------------|
# | G2/G4 show lower collection hrs than Control from week 1 | Consistent with treatment effect |
# | Burst stores concentrated in metro areas | Metro is a confounder — control in PSM |
# | 5D group balance is good pre-treatment | Random assignment within burst worked |
# | Causal DAG formalises confounders | PSM and DiD target the right variables |
#
# **Next**: Notebook 02 — Difference-in-Differences
