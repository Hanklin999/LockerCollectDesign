"""
sensitivity.py
==============
Sensitivity Analysis for Shopee smart locker notification experiment.

Research Question:
    Our DiD and PSM results show notification strategy reduces collection
    hours. But how robust are these findings?

    Three threats to validity we test:
        1. Hidden confounding  - is there an unobserved store characteristic
                                 that both drove treatment assignment AND
                                 caused the outcome to change?
        2. Placebo tests       - does the effect appear where it shouldn't?
        3. Specification       - does the result depend on arbitrary
                                 modelling choices?

Methods:
    1. Rosenbaum Bounds
       How strong would hidden confounding need to be (Gamma) to
       explain away our PSM result? If Gamma > 1.5, the result is
       considered robust in most applied work.

    2. Placebo Outcome Test
       Test treatment effect on an outcome that SHOULD NOT be affected:
       complaint_rate and opt_out_rate in the pre-period.
       If treatment predicts these → confounding or spillover.

    3. Placebo Treatment Test
       Randomly reassign treatment labels and re-estimate.
       True effect should be more extreme than the null distribution.
       (Mirrors the permutation test in did.py but applied to PSM ATT.)

    4. Pre-trend Test (Coefficient Stability)
       Does the DiD estimate change when we add more covariates?
       A large change suggests omitted variable bias.

    5. Leave-One-Out (LOO) Robustness
       Drop one week at a time and re-estimate DiD.
       Checks whether any single week is driving the result
       (e.g., the CNY week we already excluded).

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Rosenbaum Bounds
# ---------------------------------------------------------------------------

def rosenbaum_bounds(
    matched_df: pd.DataFrame,
    outcome: str = "collection_hrs",
    treatment_col: str = "treated",
    gamma_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute Rosenbaum sensitivity bounds via Wilcoxon signed-rank test.

    Tests how strong an unobserved confounder (Gamma) would need to be
    to overturn the study conclusion at alpha = 0.05.

    Gamma interpretation:
        Gamma = 1.0 : no hidden bias (standard assumption)
        Gamma = 1.5 : one store could be 1.5x more likely to receive
                      treatment due to unobserved characteristics
        Gamma = 2.0 : 2x more likely
        Rule of thumb: Gamma > 1.5 is considered robust

    Method (normal approximation to Wilcoxon statistic):
        Under Gamma, the test statistic W+ (sum of positive-difference ranks)
        has bounds on its expectation and variance.
        We report the worst-case (upper bound) p-value at each Gamma.

    Parameters
    ----------
    matched_df : pd.DataFrame
        Output of psm.match_stores(). Must have 'match_id' and treatment_col.
    outcome : str
        Outcome variable used for the Wilcoxon test.
    treatment_col : str
        Binary treatment indicator column.
    gamma_range : np.ndarray, optional
        Gamma values to test. Defaults to 1.0 to 3.0 in steps of 0.1.

    Returns
    -------
    pd.DataFrame
        Columns: [gamma, p_upper, p_lower, reject_upper, reject_lower]
        p_upper = worst-case p-value (most conservative)
        p_lower = best-case p-value (most optimistic)
    """
    if gamma_range is None:
        gamma_range = np.round(np.arange(1.0, 3.1, 0.1), 1)

    # Align treated and control on match_id
    treated_rows = (
        matched_df[matched_df[treatment_col] == 1]
        .set_index("match_id")[outcome]
    )
    control_rows = (
        matched_df[matched_df[treatment_col] == 0]
        .set_index("match_id")[outcome]
    )

    common_ids = treated_rows.index.intersection(control_rows.index)
    diffs = control_rows.loc[common_ids].values - treated_rows.loc[common_ids].values
    n = len(diffs)

    # Wilcoxon signed-rank statistic on observed data
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    w_plus = ranks[diffs > 0].sum()   # observed test statistic

    results = []
    for gamma in gamma_range:
        p_plus  = gamma / (1 + gamma)   # max P(T=1 in pair) under Gamma
        p_minus = 1 / (1 + gamma)       # min P(T=1 in pair) under Gamma

        # E[W+] and Var[W+] under Gamma bounds
        e_upper = p_plus  * n * (n + 1) / 2
        e_lower = p_minus * n * (n + 1) / 2
        var_w   = p_plus * p_minus * n * (n + 1) * (2 * n + 1) / 6

        sd_w = np.sqrt(var_w)

        # Upper bound p-value (worst case: treatment least likely)
        z_upper = (w_plus - e_upper) / sd_w
        p_upper = stats.norm.sf(z_upper)   # one-sided

        # Lower bound p-value (best case)
        z_lower = (w_plus - e_lower) / sd_w
        p_lower = stats.norm.sf(z_lower)

        results.append({
            "gamma":        gamma,
            "p_upper":      round(p_upper, 6),
            "p_lower":      round(p_lower, 6),
            "reject_upper": p_upper < 0.05,   # still significant under worst case?
            "reject_lower": p_lower < 0.05,
            "z_upper":      round(z_upper, 4),
            "z_lower":      round(z_lower, 4),
        })

    bounds_df = pd.DataFrame(results)

    # Find critical Gamma: highest Gamma where upper p-value < 0.05
    robust_gammas = bounds_df[bounds_df["reject_upper"]]["gamma"]
    critical_gamma = robust_gammas.max() if len(robust_gammas) > 0 else 1.0

    print(f"  Observed W+        : {w_plus:.1f}")
    print(f"  N matched pairs    : {n}")
    print(f"  Critical Gamma     : {critical_gamma}")
    print(f"  Interpretation     : Result holds against hidden confounders "
          f"up to {critical_gamma}x treatment odds ratio")

    return bounds_df


# ---------------------------------------------------------------------------
# 2. Placebo outcome test
# ---------------------------------------------------------------------------

def placebo_outcome_test(
    panel: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    placebo_outcomes: List[str] = ["complaint_rate", "opt_out_rate"],
    true_outcome: str = "collection_hrs",
    use_pre_period: bool = True,
) -> pd.DataFrame:
    """
    Test treatment effect on outcomes that should NOT be affected.

    Logic:
        If notification strategy ONLY affects collection behaviour,
        it should NOT cause systematic differences in complaint_rate
        or opt_out_rate before the experiment begins.

        If we find significant effects on placebo outcomes, it suggests:
        - Confounding: treated stores were already systematically different
        - Spillover: the treatment affected more than the intended outcome
        - Bad luck: (can happen; check effect sizes)

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel data.
    placebo_outcomes : list
        Outcomes that should not be affected by treatment.
    true_outcome : str
        The real outcome, used as a reference for comparison.
    use_pre_period : bool
        If True, use only BM week (pre-period) for placebo test.
        This is the strictest test.

    Returns
    -------
    pd.DataFrame
        One row per outcome (placebo + true), with DiD estimates.
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)
    df["post"] = df["is_post"].astype(int)
    df["did"] = df["treated"] * df["post"]

    if use_pre_period:
        # For placebo outcomes, only use BM week
        # Treatment shouldn't affect pre-period outcomes at all
        df_placebo = df[df["is_bm"] == 1].copy()
        df_placebo["fake_post"] = 0  # no post in pre-period
        df_placebo["fake_did"] = 0
        note = "BM week only"
    else:
        df_placebo = df.copy()
        note = "Full panel"

    all_outcomes = placebo_outcomes + [true_outcome]
    records = []

    for outcome in all_outcomes:
        is_placebo = outcome in placebo_outcomes

        if is_placebo and use_pre_period:
            # Can't run DiD on pre-period only (no variation in post)
            # Instead: simple difference between treated and control in BM
            treated_bm = df_placebo[df_placebo["treated"] == 1][outcome]
            control_bm = df_placebo[df_placebo["treated"] == 0][outcome]
            diff = treated_bm.mean() - control_bm.mean()
            se = np.sqrt(
                treated_bm.var() / len(treated_bm)
                + control_bm.var() / len(control_bm)
            )
            t_stat = diff / se if se > 0 else 0
            pvalue = 2 * stats.t.sf(abs(t_stat), df=len(treated_bm) + len(control_bm) - 2)
            records.append({
                "outcome":   outcome,
                "is_placebo": is_placebo,
                "estimand":  "BM difference (treated - control)",
                "coef":      round(diff, 6),
                "se":        round(se, 6),
                "pvalue":    round(pvalue, 4),
                "significant": pvalue < 0.05,
                "note":      note,
            })
        else:
            # Full DiD for the true outcome
            try:
                model = smf.ols(
                    f"{outcome} ~ did + treated + post + C(week_id)",
                    data=df,
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": df["store_id"]},
                )
                coef   = model.params.get("did", np.nan)
                se_val = model.bse.get("did", np.nan)
                pvalue = model.pvalues.get("did", np.nan)
                records.append({
                    "outcome":    outcome,
                    "is_placebo": is_placebo,
                    "estimand":   "DiD (treated × post)",
                    "coef":       round(coef, 6),
                    "se":         round(se_val, 6),
                    "pvalue":     round(pvalue, 4),
                    "significant": pvalue < 0.05,
                    "note":       "Full panel DiD",
                })
            except Exception as e:
                records.append({
                    "outcome": outcome, "is_placebo": is_placebo,
                    "estimand": "DiD", "coef": np.nan,
                    "se": np.nan, "pvalue": np.nan,
                    "significant": False, "note": str(e),
                })

    result_df = pd.DataFrame(records)

    # Summary
    placebo_significant = result_df[result_df["is_placebo"] & result_df["significant"]]
    print(f"  Placebo outcomes tested    : {len(placebo_outcomes)}")
    print(f"  Significant placebo effects: {len(placebo_significant)}")
    if len(placebo_significant) > 0:
        print("  WARNING: Significant effects found on placebo outcomes:")
        print(placebo_significant[["outcome", "coef", "pvalue"]].to_string(index=False))
    else:
        print("  PASS: No significant effects on placebo outcomes.")

    return result_df


# ---------------------------------------------------------------------------
# 3. Leave-one-out robustness
# ---------------------------------------------------------------------------

def leave_one_out_robustness(
    panel: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
) -> pd.DataFrame:
    """
    Re-estimate DiD leaving out one week at a time.

    Checks whether any single week is driving the result.
    Particularly important for our dataset because:
        - We only have 5 weeks (1 BM + 4 post)
        - The CNY week (02-09) was already excluded
        - Each week has non-trivial influence

    If the estimate is stable across LOO iterations,
    it is not driven by any single week.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel (already with CNY week excluded).

    Returns
    -------
    pd.DataFrame
        Columns: [excluded_week, coef, se, ci_low, ci_high, pvalue]
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)
    df["post"]    = df["is_post"].astype(int)
    df["did"]     = df["treated"] * df["post"]

    all_weeks = sorted(df["week_id"].unique())
    records   = []

    # Full-sample estimate first
    full_model = smf.ols(
        f"{outcome} ~ did + treated + post + C(week_id)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["store_id"]})
    full_coef = full_model.params.get("did", np.nan)

    records.append({
        "excluded_week": "None (full sample)",
        "n_weeks":       len(all_weeks),
        "coef":          round(full_coef, 4),
        "se":            round(full_model.bse.get("did", np.nan), 4),
        "ci_low":        round(full_model.conf_int().loc["did", 0], 4)
                         if "did" in full_model.conf_int().index else np.nan,
        "ci_high":       round(full_model.conf_int().loc["did", 1], 4)
                         if "did" in full_model.conf_int().index else np.nan,
        "pvalue":        round(full_model.pvalues.get("did", np.nan), 4),
    })

    for drop_week in all_weeks:
        df_loo = df[df["week_id"] != drop_week]

        # Skip if only BM week remains
        if df_loo["post"].sum() == 0 or df_loo["is_bm"].sum() == 0:
            continue

        try:
            model = smf.ols(
                f"{outcome} ~ did + treated + post + C(week_id)",
                data=df_loo,
            ).fit(cov_type="cluster", cov_kwds={"groups": df_loo["store_id"]})

            coef = model.params.get("did", np.nan)
            ci   = model.conf_int()
            records.append({
                "excluded_week": f"week_id={drop_week}",
                "n_weeks":       len(all_weeks) - 1,
                "coef":          round(coef, 4),
                "se":            round(model.bse.get("did", np.nan), 4),
                "ci_low":        round(ci.loc["did", 0], 4) if "did" in ci.index else np.nan,
                "ci_high":       round(ci.loc["did", 1], 4) if "did" in ci.index else np.nan,
                "pvalue":        round(model.pvalues.get("did", np.nan), 4),
            })
        except Exception:
            continue

    loo_df = pd.DataFrame(records)

    # Stability check
    post_loo = loo_df[loo_df["excluded_week"] != "None (full sample)"]
    coef_sd  = post_loo["coef"].std()
    print(f"  Full-sample estimate : {full_coef:.4f} hrs")
    print(f"  LOO estimates range  : [{post_loo['coef'].min():.4f}, {post_loo['coef'].max():.4f}]")
    print(f"  LOO estimates SD     : {coef_sd:.4f}")
    if coef_sd < 0.3:
        print("  STABLE: Estimate does not depend heavily on any single week.")
    else:
        print("  WARNING: Large variation across LOO estimates — check specific weeks.")

    return loo_df


# ---------------------------------------------------------------------------
# 4. Coefficient stability (covariate sensitivity)
# ---------------------------------------------------------------------------

def coefficient_stability(
    panel: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
) -> pd.DataFrame:
    """
    Test how stable the DiD estimate is as we add more covariates.

    A large change in the coefficient when adding covariates suggests
    that the omitted variables matter — i.e., there may be residual
    confounding in the simpler specifications.

    Stability criterion (Oster 2019 rule of thumb):
        If |coef_with_controls - coef_without_controls| /
           |coef_without_controls| < 0.15 (15%), the result is stable.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel data.

    Returns
    -------
    pd.DataFrame
        One row per specification, ordered from simplest to most complex.
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups)].copy()
    df["treated"] = df["treatment_group"].isin(treated_groups).astype(int)
    df["post"]    = df["is_post"].astype(int)
    df["did"]     = df["treated"] * df["post"]

    specifications = [
        ("No controls",
         f"{outcome} ~ did + treated + post"),
        ("+ Time FE",
         f"{outcome} ~ did + treated + post + C(week_id)"),
        ("+ Store FE",
         f"{outcome} ~ did + treated + post + C(store_id)"),
        ("+ Time & Store FE",
         f"{outcome} ~ did + treated + post + C(week_id) + C(store_id)"),
        ("+ Utilization",
         f"{outcome} ~ did + treated + post + C(week_id) + C(store_id) + avg_utilization_rate"),
        ("+ Volume",
         f"{outcome} ~ did + treated + post + C(week_id) + C(store_id) + avg_utilization_rate + avg_daily_volume"),
        ("+ Metro + Capacity (full)",
         f"{outcome} ~ did + treated + post + C(week_id) + C(store_id) + avg_utilization_rate + avg_daily_volume + is_metro + capacity"),
    ]

    records = []
    for spec_name, formula in specifications:
        try:
            model = smf.ols(formula, data=df).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["store_id"]},
            )
            coef   = model.params.get("did", np.nan)
            se_val = model.bse.get("did", np.nan)
            ci     = model.conf_int()
            records.append({
                "specification": spec_name,
                "coef":          round(coef, 4),
                "se":            round(se_val, 4),
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

    stability_df = pd.DataFrame(records)

    # Oster stability check
    coef_no_controls = stability_df.iloc[0]["coef"]
    coef_full        = stability_df.iloc[-1]["coef"]
    if not np.isnan(coef_no_controls) and coef_no_controls != 0:
        pct_change = abs(coef_full - coef_no_controls) / abs(coef_no_controls)
        print(f"  Coef (no controls) : {coef_no_controls:.4f}")
        print(f"  Coef (full)        : {coef_full:.4f}")
        print(f"  % change           : {pct_change * 100:.1f}%")
        if pct_change < 0.15:
            print("  STABLE: <15% change across specifications (Oster criterion).")
        else:
            print("  NOTE: >15% change — covariates matter. Report full specification.")

    return stability_df


# ---------------------------------------------------------------------------
# Full sensitivity pipeline
# ---------------------------------------------------------------------------

def run_sensitivity_pipeline(
    panel: pd.DataFrame,
    matched_df: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
    gamma_range: Optional[np.ndarray] = None,
) -> Dict:
    """
    End-to-end sensitivity analysis pipeline.

    Runs all four sensitivity tests and returns a unified result dict.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel from generate_store_panel().
    matched_df : pd.DataFrame
        Matched sample from psm.match_stores() — needed for Rosenbaum bounds.
    treated_groups : list
        Treatment groups.
    control_group : str
        Control group.
    outcome : str
        Primary outcome variable.
    gamma_range : np.ndarray, optional
        Gamma values for Rosenbaum bounds.

    Returns
    -------
    dict with keys:
        rosenbaum_df, placebo_df, loo_df, stability_df
    """
    print("=" * 60)
    print("SENSITIVITY ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1] Rosenbaum Bounds (hidden confounding threshold)...")
    rosenbaum_df = rosenbaum_bounds(matched_df, outcome, gamma_range=gamma_range)
    print(rosenbaum_df[["gamma", "p_upper", "reject_upper"]].head(15).to_string(index=False))

    print("\n[2] Placebo Outcome Test...")
    placebo_df = placebo_outcome_test(
        panel, treated_groups, control_group,
        placebo_outcomes=["complaint_rate", "opt_out_rate"],
        true_outcome=outcome,
    )
    print(placebo_df[["outcome", "is_placebo", "coef", "pvalue", "significant"]].to_string(index=False))

    print("\n[3] Leave-One-Out Week Robustness...")
    loo_df = leave_one_out_robustness(panel, treated_groups, control_group, outcome)
    print(loo_df[["excluded_week", "coef", "se", "pvalue"]].to_string(index=False))

    print("\n[4] Coefficient Stability (covariate sensitivity)...")
    stability_df = coefficient_stability(panel, treated_groups, control_group, outcome)
    print(stability_df[["specification", "coef", "se", "pvalue"]].to_string(index=False))

    # Overall verdict
    print("\n" + "=" * 60)
    print("SENSITIVITY SUMMARY")
    print("=" * 60)

    robust_gammas = rosenbaum_df[rosenbaum_df["reject_upper"]]["gamma"]
    critical_gamma = robust_gammas.max() if len(robust_gammas) > 0 else 1.0
    placebo_pass = not placebo_df[placebo_df["is_placebo"] & placebo_df["significant"]].shape[0] > 0
    loo_stable = loo_df[loo_df["excluded_week"] != "None (full sample)"]["coef"].std() < 0.3
    stability_pass = (
        abs(stability_df.iloc[-1]["coef"] - stability_df.iloc[0]["coef"])
        / abs(stability_df.iloc[0]["coef"]) < 0.15
        if stability_df.iloc[0]["coef"] != 0 else True
    )

    print(f"  Rosenbaum critical Gamma : {critical_gamma} {'✓' if critical_gamma >= 1.5 else '✗'}")
    print(f"  Placebo outcome test     : {'PASS ✓' if placebo_pass else 'FAIL ✗'}")
    print(f"  LOO week stability       : {'STABLE ✓' if loo_stable else 'UNSTABLE ✗'}")
    print(f"  Coefficient stability    : {'STABLE ✓' if stability_pass else 'CHECK ✗'}")

    return {
        "rosenbaum_df": rosenbaum_df,
        "placebo_df":   placebo_df,
        "loo_df":       loo_df,
        "stability_df": stability_df,
        "critical_gamma": critical_gamma,
        "all_pass":     placebo_pass and loo_stable and stability_pass,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # Requires data_generation.py and psm.py to have been run first
    data_path   = "data/processed/store_panel.csv"
    if not os.path.exists(data_path):
        print("Run data_generation.py first.")
        sys.exit(1)

    panel = pd.read_csv(data_path)
    print(f"Loaded panel: {len(panel):,} store-week observations")

    # Load matched data from PSM (or re-run PSM inline)
    from psm import prepare_psm_data, estimate_propensity_scores, match_stores, COVARIATE_COLS

    print("\nRunning PSM to get matched sample for Rosenbaum bounds...")
    store_df = prepare_psm_data(panel)
    ps_scores, _, _ = estimate_propensity_scores(store_df, COVARIATE_COLS)
    matched_df, _ = match_stores(store_df, ps_scores)

    # Run full sensitivity pipeline
    results = run_sensitivity_pipeline(
        panel=panel,
        matched_df=matched_df,
        treated_groups=["5D_G2", "5D_G4"],
        control_group="5D_Control",
        outcome="collection_hrs",
    )

    print(f"\nOverall verdict: {'All tests passed ✓' if results['all_pass'] else 'Some tests failed — review above'}")
