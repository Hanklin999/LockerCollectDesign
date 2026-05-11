"""
psm.py
======
Propensity Score Matching for Shopee smart locker notification experiment.

Research Question:
    After controlling for store characteristics that drove treatment
    assignment, what is the true causal effect of notification strategy
    (G2/G4 vs Control) on collection hours?

Why PSM is needed here:
    Treatment assignment was NOT random. Stores were grouped based on
    historical inventory levels (pct_closure_hours > 5% = burst store).
    Within burst stores, the 5D groups (Control/G2/G4) were randomly
    assigned — but the 6D group received a different deadline policy.
    PSM lets us:
        1. Verify the 5D random assignment actually balanced covariates
        2. Compare 5D vs 6D controlling for store characteristics
        3. Provide an alternative ATT estimate to cross-validate DiD

Design:
    - Unit: store (one observation per store, using post-period average)
    - Treatment: G2 or G4 (3 or 5 touches) vs Control (2 touches)
    - Covariates: store characteristics known before experiment
      [avg_utilization_rate, avg_daily_volume, capacity,
       is_metro, pct_closure_hours]
    - Matching: 1:1 nearest-neighbor within caliper on logit(propensity)
    - Estimand: ATT (Average Treatment Effect on the Treated)

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from typing import Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

COVARIATE_COLS = [
    "avg_utilization_rate",
    "avg_daily_volume",
    "capacity",
    "is_metro",
    "pct_closure_hours",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_psm_data(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    use_post_avg: bool = True,
) -> pd.DataFrame:
    """
    Collapse panel to store-level cross-section for PSM.

    PSM operates on one observation per store. We use the post-period
    average outcome to represent each store's result under its assigned
    treatment.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of generate_store_panel().
    treated_groups : list
        Groups to label as treated (T=1).
    control_group : str
        Group to label as control (T=0).
    use_post_avg : bool
        If True, average over post-period weeks only.
        If False, use all weeks (including BM).

    Returns
    -------
    pd.DataFrame
        One row per store with columns:
        [store_id, treatment_group, treated, collection_hrs,
         rts_rate, complaint_rate, opt_out_rate, + COVARIATE_COLS]
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()

    if use_post_avg:
        df = df[df["is_post"] == 1]

    # Aggregate to store level
    agg_dict = {
        "collection_hrs":       "mean",
        "rts_rate":             "mean",
        "complaint_rate":       "mean",
        "opt_out_rate":         "mean",
        "treatment_group":      "first",
        "avg_utilization_rate": "first",   # static store characteristics
        "avg_daily_volume":     "first",
        "capacity":             "first",
        "is_metro":             "first",
        "pct_closure_hours":    "first",
        "region_type":          "first",
        "city":                 "first",
    }

    store_df = (
        df.groupby("store_id")
        .agg(agg_dict)
        .reset_index()
    )

    store_df["treated"] = store_df["treatment_group"].isin(treated_groups).astype(int)

    return store_df


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------

def estimate_propensity_scores(
    store_df: pd.DataFrame,
    covariate_cols: list = COVARIATE_COLS,
    random_state: int = 42,
) -> Tuple[np.ndarray, LogisticRegression, StandardScaler]:
    """
    Estimate P(T=1 | X) via logistic regression.

    Uses standardised covariates for numerical stability.
    Reports AUC as a check: too-high AUC (>0.8) suggests
    near-perfect separation, which would make matching unreliable.
    Too-low AUC (~0.5) suggests no confounding, PSM is unnecessary
    but harmless.

    Parameters
    ----------
    store_df : pd.DataFrame
        Output of prepare_psm_data().
    covariate_cols : list
        Columns to use as covariates.
    random_state : int
        Random seed for LogisticRegression.

    Returns
    -------
    Tuple of:
        propensity_scores : np.ndarray, shape (n_stores,)
        fitted_model      : LogisticRegression
        scaler            : StandardScaler (for later use)
    """
    X = store_df[covariate_cols].values
    T = store_df["treated"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        C=1.0,
    )
    model.fit(X_scaled, T)
    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    auc = roc_auc_score(T, propensity_scores)
    print(f"  Propensity model AUC: {auc:.3f}")
    if auc > 0.85:
        print("  WARNING: High AUC suggests strong separation. "
              "Matching quality may be poor — check common support.")
    elif auc < 0.55:
        print("  NOTE: Low AUC suggests minimal confounding. "
              "PSM and naive comparison should yield similar estimates.")

    return propensity_scores, model, scaler


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_stores(
    store_df: pd.DataFrame,
    propensity_scores: np.ndarray,
    caliper: float = 0.05,
    ratio: int = 1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1:k nearest-neighbour matching on logit(propensity score).

    Matching on logit(PS) rather than PS itself is standard practice
    (Rosenbaum & Rubin 1985). The caliper is expressed in logit units.
    A caliper of 0.05 is approximately 0.2 * SD(logit PS), which is
    the commonly recommended rule of thumb.

    Parameters
    ----------
    store_df : pd.DataFrame
        Store-level cross-section with 'treated' column.
    propensity_scores : np.ndarray
        Output of estimate_propensity_scores().
    caliper : float
        Maximum allowed logit(PS) distance for a valid match.
        Default 0.05 (≈ 0.2 SD rule of thumb).
    ratio : int
        Number of control stores matched per treated store.
    random_state : int
        Seed for breaking ties.

    Returns
    -------
    Tuple of:
        matched_df   : pd.DataFrame with matched treated + control rows
        match_report : pd.DataFrame summarising match quality
    """
    np.random.seed(random_state)

    # Logit transform (clip to avoid log(0))
    ps_clipped = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
    logit_ps = np.log(ps_clipped / (1 - ps_clipped))

    store_df = store_df.copy()
    store_df["propensity_score"] = propensity_scores
    store_df["logit_ps"] = logit_ps

    treated_mask = store_df["treated"] == 1
    control_mask = store_df["treated"] == 0

    treated_df = store_df[treated_mask].reset_index(drop=True)
    control_df = store_df[control_mask].reset_index(drop=True)

    # Fit NN on control logit PS
    nn = NearestNeighbors(n_neighbors=ratio, algorithm="ball_tree")
    nn.fit(control_df["logit_ps"].values.reshape(-1, 1))

    distances, indices = nn.kneighbors(
        treated_df["logit_ps"].values.reshape(-1, 1)
    )

    # Apply caliper: drop treated units with no match within caliper
    within_caliper = distances[:, 0] <= caliper
    n_matched = within_caliper.sum()
    n_dropped = (~within_caliper).sum()

    print(f"  Treated stores       : {len(treated_df)}")
    print(f"  Matched (in caliper) : {n_matched}")
    print(f"  Dropped (out caliper): {n_dropped} "
          f"({100 * n_dropped / len(treated_df):.1f}%)")

    matched_treated = treated_df[within_caliper].copy()
    matched_treated["match_id"] = np.arange(n_matched)

    matched_control_rows = []
    for i, (in_cal, idx_row) in enumerate(zip(within_caliper, indices)):
        if not in_cal:
            continue
        for j in range(ratio):
            row = control_df.iloc[idx_row[j]].copy()
            row["match_id"] = i
            matched_control_rows.append(row)

    matched_control = pd.DataFrame(matched_control_rows).reset_index(drop=True)

    matched_df = pd.concat(
        [matched_treated, matched_control], ignore_index=True
    )

    # Match report
    match_report = pd.DataFrame({
        "metric": [
            "n_treated_total",
            "n_matched_treated",
            "n_dropped_treated",
            "pct_matched",
            "caliper_used",
            "matching_ratio",
        ],
        "value": [
            len(treated_df),
            n_matched,
            n_dropped,
            round(100 * n_matched / len(treated_df), 1),
            caliper,
            ratio,
        ],
    })

    return matched_df, match_report


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------

def compute_smd(
    store_df: pd.DataFrame,
    covariate_cols: list = COVARIATE_COLS,
) -> pd.DataFrame:
    """
    Compute Standardized Mean Difference (SMD) for each covariate.

    SMD = |mean_treated - mean_control| / sqrt((var_t + var_c) / 2)

    Interpretation:
        SMD < 0.1  : good balance (conventional threshold)
        SMD < 0.25 : acceptable
        SMD >= 0.25: imbalance, matching may be insufficient

    Parameters
    ----------
    store_df : pd.DataFrame
        Either the full unmatched dataset or the matched dataset.
    covariate_cols : list
        Covariates to check balance on.

    Returns
    -------
    pd.DataFrame
        Columns: [covariate, smd, mean_treated, mean_control,
                  var_treated, var_control, balanced]
    """
    treated = store_df[store_df["treated"] == 1]
    control = store_df[store_df["treated"] == 0]

    records = []
    for col in covariate_cols:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        var_t  = treated[col].var()
        var_c  = control[col].var()
        pooled_sd = np.sqrt((var_t + var_c) / 2)

        smd = abs(mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0

        records.append({
            "covariate":    col,
            "smd":          round(smd, 4),
            "mean_treated": round(mean_t, 4),
            "mean_control": round(mean_c, 4),
            "var_treated":  round(var_t, 4),
            "var_control":  round(var_c, 4),
            "balanced":     smd < 0.1,
        })

    return pd.DataFrame(records).sort_values("smd", ascending=False)


def check_common_support(
    store_df: pd.DataFrame,
    propensity_scores: np.ndarray,
) -> Dict:
    """
    Check common support (overlap) assumption.

    Common support requires that for every treated store, there exists
    at least one control store with a similar propensity score.
    Extreme propensity scores (near 0 or 1) indicate lack of overlap.

    Returns
    -------
    dict with:
        ps_min_treated, ps_max_treated,
        ps_min_control, ps_max_control,
        overlap_min, overlap_max,
        pct_treated_in_support : % of treated stores in overlap region
    """
    store_df = store_df.copy()
    store_df["ps"] = propensity_scores

    treated_ps = store_df[store_df["treated"] == 1]["ps"]
    control_ps = store_df[store_df["treated"] == 0]["ps"]

    overlap_min = max(treated_ps.min(), control_ps.min())
    overlap_max = min(treated_ps.max(), control_ps.max())

    pct_in_support = (
        (treated_ps >= overlap_min) & (treated_ps <= overlap_max)
    ).mean()

    return {
        "ps_min_treated":         round(treated_ps.min(), 4),
        "ps_max_treated":         round(treated_ps.max(), 4),
        "ps_min_control":         round(control_ps.min(), 4),
        "ps_max_control":         round(control_ps.max(), 4),
        "overlap_min":            round(overlap_min, 4),
        "overlap_max":            round(overlap_max, 4),
        "pct_treated_in_support": round(pct_in_support, 4),
    }


# ---------------------------------------------------------------------------
# ATT estimation
# ---------------------------------------------------------------------------

def estimate_att(
    matched_df: pd.DataFrame,
    outcome: str = "collection_hrs",
) -> Dict:
    """
    Estimate Average Treatment Effect on the Treated (ATT) from matched sample.

    ATT = E[Y(1) - Y(0) | T=1]
        = mean(outcome_treated) - mean(outcome_matched_control)

    Standard error via matched-pairs variance estimator.

    Parameters
    ----------
    matched_df : pd.DataFrame
        Output of match_stores(). Must contain 'match_id' and 'treated'.
    outcome : str
        Outcome variable.

    Returns
    -------
    dict with:
        att, se, ci_low, ci_high, pvalue, n_pairs
    """
    treated = (
        matched_df[matched_df["treated"] == 1]
        .set_index("match_id")[outcome]
    )
    control = (
        matched_df[matched_df["treated"] == 0]
        .set_index("match_id")[outcome]
    )

    # Align on match_id
    common_ids = treated.index.intersection(control.index)
    diffs = treated.loc[common_ids].values - control.loc[common_ids].values

    att = diffs.mean()
    se  = diffs.std() / np.sqrt(len(diffs))
    ci_low  = att - 1.96 * se
    ci_high = att + 1.96 * se

    # t-test
    from scipy import stats
    t_stat, pvalue = stats.ttest_1samp(diffs, 0)

    return {
        "att":      round(att, 4),
        "se":       round(se, 4),
        "ci_low":   round(ci_low, 4),
        "ci_high":  round(ci_high, 4),
        "t_stat":   round(t_stat, 4),
        "pvalue":   round(pvalue, 4),
        "n_pairs":  len(diffs),
        "significant": pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_psm_pipeline(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
    caliper: float = 0.05,
    covariate_cols: list = COVARIATE_COLS,
    seed: int = 42,
) -> Dict:
    """
    End-to-end PSM pipeline.

    Steps:
        1. Prepare store-level cross-section
        2. Estimate propensity scores
        3. Check common support
        4. Compute pre-match SMD
        5. Match treated to control stores
        6. Compute post-match SMD
        7. Estimate ATT

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel from generate_store_panel().
    treated_groups : list
        Treatment groups to compare against control.
    control_group : str
        Control group name.
    outcome : str
        Outcome variable for ATT estimation.
    caliper : float
        Caliper for nearest-neighbour matching.
    covariate_cols : list
        Covariates for propensity model and balance checks.
    seed : int
        Random seed.

    Returns
    -------
    dict with all intermediate outputs:
        store_df, ps_scores, support_check,
        smd_before, matched_df, match_report,
        smd_after, att_result
    """
    print("=" * 55)
    print("PSM PIPELINE")
    print("=" * 55)

    print("\n[1] Preparing store-level data...")
    store_df = prepare_psm_data(panel, treated_groups, control_group)
    print(f"    Treated stores : {store_df['treated'].sum()}")
    print(f"    Control stores : {(store_df['treated'] == 0).sum()}")

    print("\n[2] Estimating propensity scores...")
    ps_scores, ps_model, scaler = estimate_propensity_scores(
        store_df, covariate_cols, random_state=seed
    )

    print("\n[3] Checking common support...")
    support = check_common_support(store_df, ps_scores)
    print(f"    Treated PS range : [{support['ps_min_treated']}, {support['ps_max_treated']}]")
    print(f"    Control PS range : [{support['ps_min_control']}, {support['ps_max_control']}]")
    print(f"    Overlap region   : [{support['overlap_min']}, {support['overlap_max']}]")
    print(f"    % Treated in support: {support['pct_treated_in_support'] * 100:.1f}%")

    print("\n[4] Balance BEFORE matching...")
    smd_before = compute_smd(store_df, covariate_cols)
    print(smd_before[["covariate", "smd", "balanced"]].to_string(index=False))

    print("\n[5] Matching...")
    matched_df, match_report = match_stores(
        store_df, ps_scores, caliper=caliper, seed=seed
    )

    print("\n[6] Balance AFTER matching...")
    smd_after = compute_smd(matched_df, covariate_cols)
    print(smd_after[["covariate", "smd", "balanced"]].to_string(index=False))

    n_balanced_before = smd_before["balanced"].sum()
    n_balanced_after  = smd_after["balanced"].sum()
    print(f"\n    Covariates balanced (SMD<0.1): "
          f"{n_balanced_before}/{len(covariate_cols)} → "
          f"{n_balanced_after}/{len(covariate_cols)}")

    print(f"\n[7] Estimating ATT on '{outcome}'...")
    att = estimate_att(matched_df, outcome)
    print(f"    ATT    : {att['att']:.3f} hrs")
    print(f"    SE     : {att['se']:.3f}")
    print(f"    95% CI : [{att['ci_low']:.3f}, {att['ci_high']:.3f}]")
    print(f"    P-value: {att['pvalue']:.4f}")
    print(f"    N pairs: {att['n_pairs']}")
    print(f"    Significant: {att['significant']}")

    print("\n" + "=" * 55)

    return {
        "store_df":     store_df,
        "ps_scores":    ps_scores,
        "ps_model":     ps_model,
        "support":      support,
        "smd_before":   smd_before,
        "matched_df":   matched_df,
        "match_report": match_report,
        "smd_after":    smd_after,
        "att":          att,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    data_path = "data/processed/store_panel.csv"
    if not os.path.exists(data_path):
        print("Run data_generation.py first.")
        sys.exit(1)

    panel = pd.read_csv(data_path)
    print(f"Loaded panel: {len(panel):,} store-week observations\n")

    # --- Comparison 1: G2+G4 (any extra touch) vs Control ---
    print(">>> Comparison 1: Any extra touch (G2+G4) vs Control")
    results_main = run_psm_pipeline(
        panel,
        treated_groups=["5D_G2", "5D_G4"],
        control_group="5D_Control",
        outcome="collection_hrs",
    )

    # --- Comparison 2: G2 only vs Control ---
    print("\n>>> Comparison 2: G2 (3 touches) vs Control (2 touches)")
    results_g2 = run_psm_pipeline(
        panel,
        treated_groups=["5D_G2"],
        control_group="5D_Control",
        outcome="collection_hrs",
    )

    # --- Comparison 3: G4 only vs Control ---
    print("\n>>> Comparison 3: G4 (5 touches) vs Control (2 touches)")
    results_g4 = run_psm_pipeline(
        panel,
        treated_groups=["5D_G4"],
        control_group="5D_Control",
        outcome="collection_hrs",
    )

    # --- ATT comparison table ---
    print("\n>>> ATT Summary Table")
    summary = pd.DataFrame([
        {"comparison": "G2+G4 vs Control", **results_main["att"]},
        {"comparison": "G2 vs Control",    **results_g2["att"]},
        {"comparison": "G4 vs Control",    **results_g4["att"]},
    ])
    print(summary[["comparison", "att", "se", "ci_low", "ci_high", "pvalue", "significant"]].to_string(index=False))
