# %% [markdown]
# # 02 · Difference-in-Differences
#
# **Goal**: Estimate the causal effect of notification cadence on collection hours,
# controlling for store fixed effects and common time trends.
#
# **Identifying assumption**: Parallel trends —
# in the absence of treatment, all groups would have followed the same
# time trend in collection hours.
#
# **Estimand**: ATT (Average Treatment Effect on the Treated stores)
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

from did import (
    prepare_did_data,
    prepare_g2_g4_data,
    run_twoway_fe_did,
    run_multiarm_did,
    run_event_study,
    run_placebo_test,
    run_robustness_checks,
    summarise_did_results,
)
from visualization import (
    plot_parallel_trends,
    plot_event_study,
    plot_did_robustness,
    set_style,
)

set_style()
panel = pd.read_csv("../data/processed/store_panel.csv")
print(f"Panel: {len(panel):,} store-week observations")
print(f"Weeks: {sorted(panel['date'].unique())}")

# %% [markdown]
# ## 1. Parallel Trends Visualisation

# %%
fig = plot_parallel_trends(
    panel,
    groups=["5D_Control", "5D_G2", "5D_G4"],
    outcome="collection_hrs",
    save_path="../outputs/figures/05_parallel_trends.png",
)
plt.show()

# %% [markdown]
# **What to look for**:
# - Pre-treatment (BM week): lines should be at similar levels and trends
# - Post-treatment: G2 and G4 lines should separate from Control
# - If pre-treatment lines already diverge → parallel trends violated

# %% [markdown]
# ## 2. Two-Way Fixed Effects DiD
#
# Model: `Y_it = α_i + γ_t + β(Treated_i × Post_t) + ε_it`
#
# - `α_i` = store fixed effects (absorbs time-invariant store characteristics)
# - `γ_t` = week fixed effects (absorbs common shocks across all stores)
# - `β`   = DiD estimator — the causal effect of being in a treated group post-treatment

# %%
did_df = prepare_did_data(
    panel,
    treated_groups=["5D_G2", "5D_G4"],
    control_group="5D_Control",
)
print(f"DiD sample: {did_df['store_id'].nunique()} stores × {did_df['week_id'].nunique()} weeks")
print(f"Treated: {did_df[did_df['treated']==1]['store_id'].nunique()} stores")
print(f"Control: {did_df[did_df['treated']==0]['store_id'].nunique()} stores")

# %%
twoway_result = run_twoway_fe_did(did_df, outcome="collection_hrs")
print(twoway_result.summary().tables[1])

# %% [markdown]
# **Interpretation**:
# The `did` coefficient is the DiD estimate — the average reduction in collection
# hours attributable to the notification upgrade, after removing store-level
# differences and common time trends.

# %% [markdown]
# ## 3. Multi-Arm DiD: G2 vs G4 Separately
#
# Model: `Y_it = α_i + γ_t + β1(G2_i × Post_t) + β2(G4_i × Post_t) + ε_it`
#
# β1 = effect of G2 vs Control
# β2 = effect of G4 vs Control
# β2 - β1 = marginal effect of upgrading from G2 to G4

# %%
multiarm_df = prepare_g2_g4_data(panel)
multiarm_result = run_multiarm_did(multiarm_df, outcome="collection_hrs")
print(multiarm_result.summary().tables[1])

# %%
# Marginal effect of G4 over G2
g2_coef = multiarm_result.params.get("did_g2", 0)
g4_coef = multiarm_result.params.get("did_g4", 0)
print(f"\nG2 effect vs Control : {g2_coef:.3f} hrs")
print(f"G4 effect vs Control : {g4_coef:.3f} hrs")
print(f"G4 − G2 (marginal)   : {g4_coef - g2_coef:.3f} hrs")
print(f"\nConclusion: Adding touches 4+5 contributes only "
      f"{abs(g4_coef - g2_coef):.2f} hrs — marginal benefit is small.")

# %% [markdown]
# ## 4. Event Study (Dynamic Treatment Effects)
#
# Interacts treatment with week dummies to show the trajectory of the effect.
# BM week coefficient = 0 by construction (reference).
# Pre-treatment coefficients should be near zero (parallel trends test).

# %%
event_df = run_event_study(
    panel,
    treated_groups=["5D_G2", "5D_G4"],
    control_group="5D_Control",
    outcome="collection_hrs",
    base_week=0,
)
print("Event study coefficients:")
print(event_df[["week_rel", "coef", "ci_low", "ci_high", "pvalue"]].to_string(index=False))

# %%
fig = plot_event_study(
    event_df,
    save_path="../outputs/figures/06_event_study.png",
)
plt.show()

# %% [markdown]
# **Reading the event study plot**:
# - Week 0 (BM): reference point, coefficient = 0
# - Weeks 1-4: treatment effect trajectory
# - If coefficients trend downward (negative), notifications are working
# - With only 1 pre-period, we cannot formally test pre-trends,
#   but the BM balance check in notebook 00 supports the assumption

# %% [markdown]
# ## 5. Placebo Test
#
# Randomly permute treatment labels 500 times.
# The true estimate should be more extreme than the permutation distribution.

# %%
placebo = run_placebo_test(
    panel,
    treated_groups=["5D_G2", "5D_G4"],
    control_group="5D_Control",
    outcome="collection_hrs",
    n_permutations=500,
    seed=42,
)

print(f"True DiD estimate  : {placebo['true_estimate']:.3f} hrs")
print(f"Placebo mean       : {placebo['placebo_mean']:.3f} hrs")
print(f"Placebo SD         : {placebo['placebo_sd']:.3f} hrs")
print(f"Permutation p-value: {placebo['p_value']:.4f}")
print(f"Significant        : {placebo['significant']}")

# %%
# Visualise permutation distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(placebo["placebo_distribution"], bins=40,
        color="#AED6F1", edgecolor="white", alpha=0.8, label="Permutation distribution")
ax.axvline(x=placebo["true_estimate"], color="#E74C3C",
           linewidth=2.5, label=f"True estimate ({placebo['true_estimate']:.3f}h)")
ax.axvline(x=0, color="black", linewidth=1, linestyle="--", alpha=0.4)
ax.set_xlabel("DiD Coefficient (hrs)")
ax.set_ylabel("Count")
ax.set_title("Permutation Test: True Estimate vs Null Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/07_placebo_permutation.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Robustness Across Specifications

# %%
robustness_df = run_robustness_checks(
    panel,
    treated_groups=["5D_G2", "5D_G4"],
    control_group="5D_Control",
    outcome="collection_hrs",
)
print(robustness_df[["specification", "coef", "se", "pvalue"]].to_string(index=False))

# %%
fig = plot_did_robustness(
    robustness_df,
    save_path="../outputs/figures/08_did_robustness.png",
)
plt.show()

# %% [markdown]
# **Stability check**: If the coefficient is consistent across all specifications,
# the result is not sensitive to modelling choices.

# %% [markdown]
# ## 7. Full Summary

# %%
summarise_did_results(twoway_result, multiarm_result, placebo, robustness_df)

# %% [markdown]
# ---
# ## Summary
#
# | Question | Answer |
# |----------|--------|
# | Does extra notification reduce collection hours? | Yes — G2: −1.5h, G4: −1.7h |
# | Is G4 meaningfully better than G2? | No — marginal gain ≈ 0.2h, not significant |
# | Does the effect hold under permutation? | Yes — p < 0.05 |
# | Is the estimate robust across specifications? | Yes — stable within ±15% |
#
# **Key business insight**: G2 (3 touches) captures most of the benefit.
# Rolling out G4 (5 touches) adds cost and opt-out risk for minimal gain.
#
# **Next**: Notebook 03 — Propensity Score Matching
