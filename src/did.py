"""
did.py
======
Difference-in-Differences estimation for Shopee smart locker
notification experiment.

Research Question:
    Did the notification strategy change (G2/G4 vs Control) causally
    reduce collection hours, controlling for pre-existing store differences
    and common time trends?

Design:
    - Unit of analysis: store (store-level randomization)
    - Pre-period: BM week (2026-01-19)
    - Post-period: 4 experiment weeks (excluding CNY week 2026-02-09)
    - Treatment: 5D_G2 and 5D_G4 vs 5D_Control
    - Identifying assumption: parallel trends
      (treated and control stores would have followed the same trend
       absent the notification change)

Key outputs:
    1. DiD point estimates with clustered standard errors
    2. Event study plot coefficients (parallel trends test)
    3. Placebo test (fake treatment in pre-period)
    4. Robustness check (with/without store covariates)

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_did_data(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
) -> pd.DataFrame:
    """
    Subset and prepare panel data for DiD estimation.

    Filters to treated and control groups only (excludes 6D, 7D),
    creates DiD interaction term, and encodes time as relative
    distance from BM week.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of generate_store_panel().
    treated_groups : list
        Treatment group names to include as treated.
    control_group : str
        Control group name.

    Returns
    -------
    pd.DataFrame
        Filtered panel with DiD variables:
        [treated, post, did, week_rel, ...]
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()

    # Binary treatment indicator
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)

    # Post indicator (already in panel, but make explicit)
    df["post"] = df["is_post"].astype(int)

    # DiD interaction term
    df["did"] = df["treated"] * df["post"]

    # Relative week (0 = BM, 1-4 = experiment weeks)
    week_map = {wid: i for i, wid in enumerate(sorted(df["week_id"].unique()))}
    df["week_rel"] = df["week_id"].map(week_map)

    return df


def prepare_g2_g4_data(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare data for separate G2 vs Control and G4 vs Control comparisons.

    Returns panel with treatment_group as categorical for multi-arm DiD.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of generate_store_panel().

    Returns
    -------
    pd.DataFrame
        Filtered to 5D groups only, treatment_group as category.
    """
    df = panel[panel["treatment_group"].isin(
        ["5D_Control", "5D_G2", "5D_G4"]
    )].copy()

    df["group_cat"] = pd.Categorical(
        df["treatment_group"],
        categories=["5D_Control", "5D_G2", "5D_G4"],
    )
    df["post"] = df["is_post"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Core DiD estimators
# ---------------------------------------------------------------------------

def run_twoway_fe_did(
    df: pd.DataFrame,
    outcome: str = "collection_hrs",
    covariates: Optional[list] = None,
) -> object:
    """
    Two-way fixed effects DiD (store FE + week FE).

    Model:
        Y_it = alpha_i + gamma_t + beta * (treated_i * post_t) + X_it * delta + e_it

    beta is the DiD estimator (ATT under parallel trends assumption).
    Standard errors clustered at store level.

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_did_data(). Must be indexed by [store_id, week_id].
    outcome : str
        Outcome variable name.
    covariates : list, optional
        Time-varying store covariates to control for.

    Returns
    -------
    statsmodels OLS result object.
    """
    # Reset index if panel was indexed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    exog_vars = ["did", "treated", "post"]
    if covariates:
        exog_vars += covariates

    # Two-way FE via store + week dummies (within-estimator via OLS)
    formula = (
        outcome + " ~ "
        + " + ".join(exog_vars)
        + " + C(store_id) + C(week_id)"
    )
    result = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store_id"]},
    )
    return result


def run_multiarm_did(
    df: pd.DataFrame,
    outcome: str = "collection_hrs",
    covariates: Optional[list] = None,
) -> object:
    """
    Multi-arm DiD comparing G2 and G4 simultaneously against Control.

    Model:
        Y_it = alpha_i + gamma_t
               + beta1 * (G2_i * post_t)
               + beta2 * (G4_i * post_t)
               + e_it

    beta1 = ATT for G2 vs Control
    beta2 = ATT for G4 vs Control
    beta2 - beta1 = marginal effect of 5th notification touch

    Parameters
    ----------
    df : pd.DataFrame
        Output of prepare_g2_g4_data(). Indexed by [store_id, week_id].
    outcome : str
        Outcome variable name.
    covariates : list, optional
        Additional covariates.

    Returns
    -------
    statsmodels OLS result object.
    """
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = df.copy()
    df["is_g2"] = (df["treatment_group"] == "5D_G2").astype(int)
    df["is_g4"] = (df["treatment_group"] == "5D_G4").astype(int)
    df["did_g2"] = df["is_g2"] * df["post"]
    df["did_g4"] = df["is_g4"] * df["post"]

    exog_vars = ["did_g2", "did_g4", "is_g2", "is_g4", "post"]
    if covariates:
        exog_vars += covariates

    formula = (
        outcome + " ~ "
        + " + ".join(exog_vars)
        + " + C(store_id) + C(week_id)"
    )
    result = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store_id"]},
    )
    return result


# ---------------------------------------------------------------------------
# Parallel trends test: event study
# ---------------------------------------------------------------------------

def run_event_study(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
    base_week: int = 0,
) -> pd.DataFrame:
    """
    Event study regression to test the parallel trends assumption.

    Interacts treatment indicator with week dummies, omitting the BM
    week (base_week=0) as reference. Pre-treatment coefficients should
    be statistically indistinguishable from zero.

    In this experiment we only have one pre-period (BM week), so the
    test is limited. The plot will show:
        week 0 (BM): reference (0 by construction)
        weeks 1-4:   treatment effect trajectory

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel (not indexed).
    treated_groups : list
        Groups to treat as treated.
    control_group : str
        Control group.
    outcome : str
        Outcome variable.
    base_week : int
        Reference week id (omitted category).

    Returns
    -------
    pd.DataFrame
        Columns: [week_rel, coef, ci_low, ci_high, is_pre]
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)

    # Relative week
    week_ids = sorted(df["week_id"].unique())
    week_map = {wid: i for i, wid in enumerate(week_ids)}
    df["week_rel"] = df["week_id"].map(week_map)

    # Create week dummies interacted with treated
    week_rels = sorted(df["week_rel"].unique())
    interaction_terms = []

    for w in week_rels:
        if w == base_week:
            continue
        col = f"treat_w{w}"
        df[col] = df["treated"] * (df["week_rel"] == w).astype(int)
        interaction_terms.append(col)

    # Week dummies (for time FE)
    week_dummies = [f"week_rel_{w}" for w in week_rels if w != base_week]
    for w in week_rels:
        if w != base_week:
            df[f"week_rel_{w}"] = (df["week_rel"] == w).astype(int)

    formula = (
        outcome
        + " ~ treated + "
        + " + ".join(interaction_terms)
        + " + "
        + " + ".join(week_dummies)
        + " + C(store_id)"   # store FE via dummies (small N = feasible)
    )

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store_id"]},
    )

    # Extract interaction coefficients
    records = []
    for w in week_rels:
        if w == base_week:
            records.append({
                "week_rel": w,
                "coef":     0.0,
                "ci_low":   0.0,
                "ci_high":  0.0,
                "is_pre":   True,
                "se":       0.0,
                "pvalue":   1.0,
            })
        else:
            col = f"treat_w{w}"
            coef = model.params.get(col, np.nan)
            ci = model.conf_int()
            records.append({
                "week_rel": w,
                "coef":     coef,
                "ci_low":   ci.loc[col, 0] if col in ci.index else np.nan,
                "ci_high":  ci.loc[col, 1] if col in ci.index else np.nan,
                "is_pre":   False,
                "se":       model.bse.get(col, np.nan),
                "pvalue":   model.pvalues.get(col, np.nan),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Placebo test
# ---------------------------------------------------------------------------

def run_placebo_test(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
    n_permutations: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Permutation-based placebo test.

    Randomly shuffles treatment labels among stores within the BM week
    and re-estimates the DiD coefficient. If the true estimate is
    extreme relative to the permutation distribution, it supports
    the causal interpretation.

    Parameters
    ----------
    n_permutations : int
        Number of random permutations. Default 500.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        true_estimate : float
            DiD coefficient from real treatment assignment.
        placebo_distribution : np.ndarray
            DiD coefficients under permuted labels.
        p_value : float
            Proportion of placebo estimates more extreme than true estimate.
        significant : bool
            True if p_value < 0.05.
    """
    rng = np.random.default_rng(seed)

    # True estimate
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)
    df["post"] = df["is_post"].astype(int)
    df["did"] = df["treated"] * df["post"]

    true_model = smf.ols(
        f"{outcome} ~ did + treated + post + C(week_id)",
        data=df,
    ).fit()
    true_estimate = true_model.params.get("did", np.nan)

    # Permutation distribution
    placebo_estimates = []
    store_treatments = df[["store_id", "treated"]].drop_duplicates()

    for _ in range(n_permutations):
        # Shuffle treatment at store level
        shuffled = store_treatments.copy()
        shuffled["treated"] = rng.permutation(shuffled["treated"].values)

        df_perm = df.drop(columns=["treated", "did"]).merge(
            shuffled, on="store_id"
        )
        df_perm["did"] = df_perm["treated"] * df_perm["post"]

        perm_model = smf.ols(
            f"{outcome} ~ did + treated + post + C(week_id)",
            data=df_perm,
        ).fit()
        placebo_estimates.append(perm_model.params.get("did", np.nan))

    placebo_estimates = np.array(placebo_estimates)

    # Two-sided p-value
    p_value = np.mean(np.abs(placebo_estimates) >= np.abs(true_estimate))

    return {
        "true_estimate":        true_estimate,
        "placebo_distribution": placebo_estimates,
        "p_value":              p_value,
        "significant":          p_value < 0.05,
        "placebo_mean":         np.mean(placebo_estimates),
        "placebo_sd":           np.std(placebo_estimates),
    }


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def run_robustness_checks(
    panel: pd.DataFrame,
    treated_groups: list = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
) -> pd.DataFrame:
    """
    Run DiD under multiple specifications to test robustness.

    Specifications:
        1. No controls, no FE (naive)
        2. Store FE only
        3. Time FE only
        4. Two-way FE (preferred)
        5. Two-way FE + store covariates

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel data.

    Returns
    -------
    pd.DataFrame
        One row per specification with columns:
        [spec, coef, se, ci_low, ci_high, pvalue, n_obs]
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)
    df["post"] = df["is_post"].astype(int)
    df["did"] = df["treated"] * df["post"]

    specs = [
        ("Naive OLS",
         f"{outcome} ~ did + treated + post"),
        ("Time FE only",
         f"{outcome} ~ did + treated + post + C(week_id)"),
        ("Store FE only",
         f"{outcome} ~ did + treated + post + C(store_id)"),
        ("Two-way FE",
         f"{outcome} ~ did + treated + post + C(store_id) + C(week_id)"),
        ("Two-way FE + covariates",
         f"{outcome} ~ did + treated + post + C(store_id) + C(week_id)"
         f" + avg_utilization_rate + avg_daily_volume + is_metro"),
    ]

    records = []
    for spec_name, formula in specs:
        try:
            model = smf.ols(formula, data=df).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["store_id"]},
            )
            coef = model.params.get("did", np.nan)
            se = model.bse.get("did", np.nan)
            ci = model.conf_int()
            records.append({
                "specification": spec_name,
                "coef":          round(coef, 4),
                "se":            round(se, 4),
                "ci_low":        round(ci.loc["did", 0], 4) if "did" in ci.index else np.nan,
                "ci_high":       round(ci.loc["did", 1], 4) if "did" in ci.index else np.nan,
                "pvalue":        round(model.pvalues.get("did", np.nan), 4),
                "n_obs":         int(model.nobs),
            })
        except Exception as e:
            records.append({
                "specification": spec_name,
                "coef": np.nan, "se": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "pvalue": np.nan, "n_obs": 0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_did_results(
    twoway_result: object,
    multiarm_result: object,
    placebo: Dict,
    robustness: pd.DataFrame,
) -> None:
    """
    Print a formatted summary of all DiD results.

    Parameters
    ----------
    twoway_result : statsmodels result
        Two-way FE DiD (treated vs control).
    multiarm_result : statsmodels result
        Multi-arm DiD (G2 vs Control, G4 vs Control).
    placebo : dict
        Output of run_placebo_test().
    robustness : pd.DataFrame
        Output of run_robustness_checks().
    """
    print("=" * 60)
    print("DiD RESULTS SUMMARY")
    print("=" * 60)

    print("\n[1] Two-way FE DiD (Treated vs Control)")
    print(f"    DiD estimate : {twoway_result.params.get('did', 'N/A'):.3f} hrs")
    print(f"    Std Error    : {twoway_result.bse.get('did', 'N/A'):.3f}")
    print(f"    P-value      : {twoway_result.pvalues.get('did', 'N/A'):.4f}")

    print("\n[2] Multi-arm DiD")
    print(f"    G2 vs Control: {multiarm_result.params.get('did_g2', 'N/A'):.3f} hrs")
    print(f"    G4 vs Control: {multiarm_result.params.get('did_g4', 'N/A'):.3f} hrs")
    g2 = multiarm_result.params.get("did_g2", 0)
    g4 = multiarm_result.params.get("did_g4", 0)
    print(f"    G4 - G2 (marginal 4th+5th touch): {g4 - g2:.3f} hrs")

    print("\n[3] Placebo Test")
    print(f"    True estimate : {placebo['true_estimate']:.3f} hrs")
    print(f"    Placebo mean  : {placebo['placebo_mean']:.3f} hrs")
    print(f"    Placebo SD    : {placebo['placebo_sd']:.3f} hrs")
    print(f"    P-value       : {placebo['p_value']:.4f}")
    print(f"    Significant   : {placebo['significant']}")

    print("\n[4] Robustness Checks")
    print(robustness[["specification", "coef", "se", "pvalue"]].to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # Load data
    data_path = "data/processed/store_panel.csv"
    if not os.path.exists(data_path):
        print("Run data_generation.py first to create the panel data.")
        sys.exit(1)

    panel = pd.read_csv(data_path)
    print(f"Loaded panel: {len(panel):,} store-week observations")

    # 1. Two-way FE DiD
    print("\nRunning two-way FE DiD...")
    did_df = prepare_did_data(panel)
    twoway_result = run_twoway_fe_did(did_df)

    # 2. Multi-arm DiD
    print("Running multi-arm DiD...")
    multiarm_df = prepare_g2_g4_data(panel)
    multiarm_result = run_multiarm_did(multiarm_df)

    # 3. Event study
    print("Running event study...")
    event_study_df = run_event_study(panel)
    print("\nEvent study coefficients:")
    print(event_study_df[["week_rel", "coef", "ci_low", "ci_high", "pvalue"]].to_string(index=False))

    # 4. Placebo test
    print("\nRunning placebo test (500 permutations)...")
    placebo = run_placebo_test(panel, n_permutations=500)

    # 5. Robustness checks
    print("Running robustness checks...")
    robustness = run_robustness_checks(panel)

    # Summary
    summarise_did_results(twoway_result, multiarm_result, placebo, robustness)
