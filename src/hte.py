"""
hte.py
======
Heterogeneous Treatment Effects (HTE) for Shopee smart locker
notification experiment.

Research Question:
    The average treatment effect tells us the notification strategy works.
    But WHICH stores benefit most? Should we prioritize metro stores?
    High-traffic stores? Stores with extreme utilization rates?

    This file answers: Who benefits most from extra notification touches?

Why HTE matters for business:
    If effect is heterogeneous, a blanket rollout is suboptimal.
    We can target the notification upgrade only to stores where
    marginal benefit exceeds marginal cost (per-SMS cost * volume).

Methods:
    1. Subgroup analysis      - simple, interpretable, low sample-size risk
    2. T-Learner              - separate models per treatment arm
    3. S-Learner              - single model with treatment as feature
    4. X-Learner              - best for imbalanced treatment groups
    5. Causal Forest (DML)    - gold standard, provides valid CIs

Estimand:
    CATE(x) = E[Y(1) - Y(0) | X = x]
    Conditional Average Treatment Effect for store with features x.

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COVARIATE_COLS = [
    "avg_utilization_rate",
    "avg_daily_volume",
    "capacity",
    "is_metro",
    "pct_closure_hours",
]

SUBGROUP_COLS = {
    "is_metro":            {"label": "Metro vs Non-Metro",   "type": "binary"},
    "avg_daily_volume":    {"label": "Daily Volume",          "type": "continuous"},
    "avg_utilization_rate":{"label": "Utilization Rate",      "type": "continuous"},
    "capacity":            {"label": "Store Capacity",        "type": "continuous"},
    "pct_closure_hours":   {"label": "Pct Closure Hours",     "type": "continuous"},
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_hte_data(
    panel: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare store-level cross-section for CATE estimation.

    Collapses panel to one row per store using post-period averages.
    Returns feature matrix X, treatment T, and outcome Y separately
    for compatibility with meta-learner APIs.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of generate_store_panel().
    treated_groups : list
        Groups treated as T=1.
    control_group : str
        Group treated as T=0.
    outcome : str
        Outcome variable for CATE estimation.

    Returns
    -------
    Tuple of:
        store_df : pd.DataFrame, full store-level data
        X        : np.ndarray, shape (n_stores, n_features)
        T        : np.ndarray, shape (n_stores,), binary treatment
        Y        : np.ndarray, shape (n_stores,), outcome
    """
    groups = treated_groups + [control_group]
    df = panel[panel["treatment_group"].isin(groups) & (panel["is_post"] == 1)].copy()

    store_df = (
        df.groupby("store_id")
        .agg({
            outcome:                "mean",
            "treatment_group":      "first",
            "avg_utilization_rate": "first",
            "avg_daily_volume":     "first",
            "capacity":             "first",
            "is_metro":             "first",
            "pct_closure_hours":    "first",
            "region_type":          "first",
            "city":                 "first",
        })
        .reset_index()
    )

    store_df["treated"] = store_df["treatment_group"].isin(treated_groups).astype(int)

    X = store_df[COVARIATE_COLS].values.astype(float)
    T = store_df["treated"].values.astype(int)
    Y = store_df[outcome].values.astype(float)

    return store_df, X, T, Y


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

def run_subgroup_analysis(
    store_df: pd.DataFrame,
    outcome: str = "collection_hrs",
    covariate_cols: Dict = SUBGROUP_COLS,
    n_quantile_bins: int = 3,
) -> pd.DataFrame:
    """
    Compute treatment effect within predefined subgroups.

    For binary covariates: split directly into two groups.
    For continuous covariates: split into quantile bins.

    This is the most interpretable HTE approach but suffers from
    multiple comparison issues and small sample sizes per cell.
    Use as exploratory analysis; confirm with meta-learners.

    Parameters
    ----------
    store_df : pd.DataFrame
        Store-level data with 'treated' and outcome columns.
    outcome : str
        Outcome variable.
    covariate_cols : dict
        Mapping of column name to label and type.
    n_quantile_bins : int
        Number of bins for continuous covariates.

    Returns
    -------
    pd.DataFrame
        Columns: [covariate, subgroup, n_treated, n_control,
                  mean_treated, mean_control, cate, se, ci_low, ci_high]
    """
    from scipy import stats

    records = []

    for col, meta in covariate_cols.items():
        if meta["type"] == "binary":
            bins = {
                f"{meta['label']} = 0": store_df[col] == 0,
                f"{meta['label']} = 1": store_df[col] == 1,
            }
        else:
            quantiles = pd.qcut(store_df[col], q=n_quantile_bins, duplicates="drop")
            bins = {
                str(cat): quantiles == cat
                for cat in quantiles.cat.categories
            }

        for bin_label, mask in bins.items():
            sub = store_df[mask]
            treated_sub = sub[sub["treated"] == 1][outcome]
            control_sub = sub[sub["treated"] == 0][outcome]

            if len(treated_sub) < 3 or len(control_sub) < 3:
                continue

            cate = treated_sub.mean() - control_sub.mean()
            pooled_se = np.sqrt(
                treated_sub.var() / len(treated_sub)
                + control_sub.var() / len(control_sub)
            )
            t_stat, pvalue = stats.ttest_ind(treated_sub, control_sub)

            records.append({
                "covariate":    meta["label"],
                "subgroup":     bin_label,
                "n_treated":    len(treated_sub),
                "n_control":    len(control_sub),
                "mean_treated": round(treated_sub.mean(), 3),
                "mean_control": round(control_sub.mean(), 3),
                "cate":         round(cate, 3),
                "se":           round(pooled_se, 3),
                "ci_low":       round(cate - 1.96 * pooled_se, 3),
                "ci_high":      round(cate + 1.96 * pooled_se, 3),
                "pvalue":       round(pvalue, 4),
                "significant":  pvalue < 0.05,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------

class TLearner:
    """
    T-Learner: train separate outcome models for treated and control.

    CATE(x) = mu_1(x) - mu_0(x)

    where:
        mu_1(x) = E[Y | T=1, X=x]  (trained on treated only)
        mu_0(x) = E[Y | T=0, X=x]  (trained on control only)

    Strength : flexible, naturally handles different response surfaces.
    Weakness : ignores shared structure; poor with small treatment groups.
    """

    def __init__(self, base_model=None):
        if base_model is None:
            base_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            )
        self.model_t = base_model
        self.model_c = base_model.__class__(**base_model.get_params())

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        """Fit separate models on treated and control subsets."""
        self.model_t.fit(X[T == 1], Y[T == 1])
        self.model_c.fit(X[T == 0], Y[T == 0])
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for each observation."""
        return self.model_t.predict(X) - self.model_c.predict(X)


class SLearner:
    """
    S-Learner: single model with treatment indicator as a feature.

    CATE(x) = mu(x, T=1) - mu(x, T=0)

    where mu(x, t) = E[Y | X=x, T=t] trained on the full dataset.

    Strength : uses all data; works well when treatment effect is small.
    Weakness : may shrink treatment effect toward zero if T gets
               low variable importance (common in tree models).
    """

    def __init__(self, base_model=None):
        if base_model is None:
            base_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            )
        self.model = base_model

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        """Fit single model on full dataset with T appended to X."""
        XT = np.column_stack([X, T])
        self.model.fit(XT, Y)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE by contrasting T=1 and T=0 predictions."""
        X1 = np.column_stack([X, np.ones(len(X))])
        X0 = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict(X1) - self.model.predict(X0)


class XLearner:
    """
    X-Learner: cross-fitting of imputed treatment effects.

    Steps:
        Stage 1: Fit mu_0, mu_1 (same as T-Learner)
        Stage 2: Impute individual effects
                 D_1 = Y_i - mu_0(X_i) for treated units
                 D_0 = mu_1(X_i) - Y_i for control units
        Stage 3: Fit tau_1 on D_1, tau_0 on D_0
        Stage 4: CATE = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
                 where g(x) = P(T=1 | X=x) (propensity score)

    Strength : best performer when treatment groups are imbalanced.
               Well-suited for this experiment (100 vs 100 treated,
               but propensity varies by store characteristics).
    Weakness : more complex, harder to explain to non-technical audience.
    """

    def __init__(self, base_model=None, propensity_model=None):
        if base_model is None:
            base_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            )
        if propensity_model is None:
            from sklearn.linear_model import LogisticRegression
            propensity_model = LogisticRegression(max_iter=1000, random_state=42)

        self.mu_0 = base_model
        self.mu_1 = base_model.__class__(**base_model.get_params())
        self.tau_0 = base_model.__class__(**base_model.get_params())
        self.tau_1 = base_model.__class__(**base_model.get_params())
        self.propensity_model = propensity_model

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        """Fit X-Learner in four stages."""
        # Stage 1
        self.mu_0.fit(X[T == 0], Y[T == 0])
        self.mu_1.fit(X[T == 1], Y[T == 1])

        # Stage 2: imputed effects
        D_1 = Y[T == 1] - self.mu_0.predict(X[T == 1])
        D_0 = self.mu_1.predict(X[T == 0]) - Y[T == 0]

        # Stage 3
        self.tau_1.fit(X[T == 1], D_1)
        self.tau_0.fit(X[T == 0], D_0)

        # Propensity for weighting
        scaler = StandardScaler()
        self.propensity_model.fit(scaler.fit_transform(X), T)
        self._scaler = scaler

        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE using propensity-weighted combination."""
        g = self.propensity_model.predict_proba(
            self._scaler.transform(X)
        )[:, 1]
        return g * self.tau_0.predict(X) + (1 - g) * self.tau_1.predict(X)


# ---------------------------------------------------------------------------
# Double ML / Causal Forest (lightweight implementation)
# ---------------------------------------------------------------------------

def run_causal_forest_dml(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_estimators: int = 300,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal Forest via Double Machine Learning (DML) residualisation.

    This is a lightweight implementation of the DML-CausalForest idea:

    Step 1 - Partial out confounders via cross-fitting:
        Y_tilde = Y - E[Y | X]   (outcome residual)
        T_tilde = T - E[T | X]   (treatment residual)

    Step 2 - Fit a forest on the residualised problem:
        CATE(x) estimated via weighted local regression using
        RandomForest proximity weights.

    Step 3 - Bootstrap for confidence intervals.

    Note: For production use, install EconML and use CausalForestDML.
    This implementation captures the core idea without the dependency.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix, shape (n, p).
    T : np.ndarray
        Binary treatment, shape (n,).
    Y : np.ndarray
        Outcome, shape (n,).
    n_estimators : int
        Number of trees in the forest.
    n_bootstrap : int
        Bootstrap iterations for CI estimation.
    seed : int
        Random seed.

    Returns
    -------
    Tuple of:
        cate_point : np.ndarray, point estimates
        cate_lower : np.ndarray, 5th percentile bootstrap CI
        cate_upper : np.ndarray, 95th percentile bootstrap CI
    """
    rng = np.random.default_rng(seed)

    # Step 1: Partial out confounders via 5-fold cross-fitting
    Y_hat = cross_val_predict(
        GradientBoostingRegressor(n_estimators=100, random_state=seed),
        X, Y, cv=5,
    )
    T_hat = cross_val_predict(
        GradientBoostingRegressor(n_estimators=100, random_state=seed),
        X, T.astype(float), cv=5,
    )

    Y_tilde = Y - Y_hat
    T_tilde = T - T_hat

    # Step 2: Fit forest on residualised outcome
    # Pseudo-outcome: Y_tilde / T_tilde (Robinson 1988 transformation)
    # Avoid division by near-zero T_tilde values
    T_tilde_clipped = np.where(np.abs(T_tilde) > 0.01, T_tilde, np.sign(T_tilde) * 0.01)
    pseudo_outcome = Y_tilde / T_tilde_clipped

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=seed,
    )
    forest.fit(X, pseudo_outcome, sample_weight=T_tilde_clipped ** 2)
    cate_point = forest.predict(X)

    # Step 3: Bootstrap CI
    bootstrap_cates = np.zeros((n_bootstrap, len(X)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(X), len(X))
        Xb, Tb, Yb = X[idx], T[idx], Y[idx]

        Y_hat_b = cross_val_predict(
            GradientBoostingRegressor(n_estimators=50, random_state=b),
            Xb, Yb, cv=3,
        )
        T_hat_b = cross_val_predict(
            GradientBoostingRegressor(n_estimators=50, random_state=b),
            Xb, Tb.astype(float), cv=3,
        )
        Yt_b = Yb - Y_hat_b
        Tt_b = Tb - T_hat_b
        Tt_b_c = np.where(np.abs(Tt_b) > 0.01, Tt_b, np.sign(Tt_b) * 0.01)
        po_b = Yt_b / Tt_b_c

        f_b = RandomForestRegressor(
            n_estimators=100, max_features="sqrt",
            min_samples_leaf=5, random_state=b,
        )
        f_b.fit(Xb, po_b, sample_weight=Tt_b_c ** 2)
        bootstrap_cates[b] = f_b.predict(X)

    cate_lower = np.percentile(bootstrap_cates, 5, axis=0)
    cate_upper = np.percentile(bootstrap_cates, 95, axis=0)

    return cate_point, cate_lower, cate_upper


# ---------------------------------------------------------------------------
# Feature importance for CATE
# ---------------------------------------------------------------------------

def compute_cate_feature_importance(
    X: np.ndarray,
    cate_estimates: np.ndarray,
    feature_names: List[str] = COVARIATE_COLS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Identify which store features drive treatment effect heterogeneity.

    Fits a regression of CATE estimates on store features, then uses
    permutation importance to rank features by their contribution to
    explaining variation in the treatment effect.

    Parameters
    ----------
    X : np.ndarray
        Store covariate matrix.
    cate_estimates : np.ndarray
        CATE estimates from any meta-learner.
    feature_names : list
        Names corresponding to X columns.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, importance, importance_std]
        Sorted by importance descending.
    """
    # Fit a forest to predict CATE from X
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        min_samples_leaf=3,
    )
    rf.fit(X, cate_estimates)

    # Permutation importance
    perm = permutation_importance(
        rf, X, cate_estimates,
        n_repeats=30,
        random_state=seed,
    )

    importance_df = pd.DataFrame({
        "feature":        feature_names,
        "importance":     perm.importances_mean.round(4),
        "importance_std": perm.importances_std.round(4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


# ---------------------------------------------------------------------------
# Full HTE pipeline
# ---------------------------------------------------------------------------

def run_hte_pipeline(
    panel: pd.DataFrame,
    treated_groups: List[str] = ["5D_G2", "5D_G4"],
    control_group: str = "5D_Control",
    outcome: str = "collection_hrs",
    run_bootstrap: bool = True,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict:
    """
    End-to-end HTE pipeline.

    Steps:
        1. Prepare store-level data
        2. Subgroup analysis (interpretable baseline)
        3. T-Learner CATE
        4. S-Learner CATE
        5. X-Learner CATE (primary estimator)
        6. Causal Forest DML CATE (with CIs)
        7. Feature importance for heterogeneity

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel from generate_store_panel().
    treated_groups : list
        Groups to treat as T=1.
    control_group : str
        Control group.
    outcome : str
        Outcome variable.
    run_bootstrap : bool
        Whether to run bootstrap CIs for causal forest (slow).
    n_bootstrap : int
        Bootstrap iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict with all outputs:
        store_df, X, T, Y,
        subgroup_results,
        cate_t, cate_s, cate_x,
        cate_cf, cate_cf_lower, cate_cf_upper,
        feature_importance
    """
    print("=" * 55)
    print("HTE PIPELINE")
    print("=" * 55)

    print("\n[1] Preparing data...")
    store_df, X, T, Y = prepare_hte_data(
        panel, treated_groups, control_group, outcome
    )
    print(f"    Treated: {T.sum()}  Control: {(T == 0).sum()}")
    print(f"    Outcome mean (treated): {Y[T == 1].mean():.3f}")
    print(f"    Outcome mean (control): {Y[T == 0].mean():.3f}")
    print(f"    Naive ATE: {Y[T==1].mean() - Y[T==0].mean():.3f} hrs")

    print("\n[2] Subgroup analysis...")
    subgroup_results = run_subgroup_analysis(store_df, outcome)
    sig_subgroups = subgroup_results[subgroup_results["significant"]]
    print(f"    Significant subgroups: {len(sig_subgroups)}/{len(subgroup_results)}")
    if len(sig_subgroups) > 0:
        print(sig_subgroups[["covariate", "subgroup", "cate", "pvalue"]].to_string(index=False))

    print("\n[3] T-Learner...")
    t_learner = TLearner()
    t_learner.fit(X, T, Y)
    cate_t = t_learner.effect(X)
    print(f"    CATE mean: {cate_t.mean():.3f}  SD: {cate_t.std():.3f}")

    print("\n[4] S-Learner...")
    s_learner = SLearner()
    s_learner.fit(X, T, Y)
    cate_s = s_learner.effect(X)
    print(f"    CATE mean: {cate_s.mean():.3f}  SD: {cate_s.std():.3f}")

    print("\n[5] X-Learner (primary)...")
    x_learner = XLearner()
    x_learner.fit(X, T, Y)
    cate_x = x_learner.effect(X)
    print(f"    CATE mean: {cate_x.mean():.3f}  SD: {cate_x.std():.3f}")

    print("\n[6] Causal Forest DML...")
    if run_bootstrap:
        print(f"    Running {n_bootstrap} bootstrap iterations (may take 1-2 min)...")
    cate_cf, cate_cf_lower, cate_cf_upper = run_causal_forest_dml(
        X, T, Y,
        n_bootstrap=n_bootstrap if run_bootstrap else 0,
        seed=seed,
    )
    print(f"    CATE mean: {cate_cf.mean():.3f}  SD: {cate_cf.std():.3f}")
    if run_bootstrap:
        pct_significant = np.mean(
            (cate_cf_lower > 0) | (cate_cf_upper < 0)
        )
        print(f"    % stores with significant CATE: {pct_significant*100:.1f}%")

    print("\n[7] Feature importance for heterogeneity...")
    importance_df = compute_cate_feature_importance(X, cate_x)
    print(importance_df.to_string(index=False))

    # Attach CATE estimates to store_df for downstream analysis
    store_df["cate_t"]  = cate_t
    store_df["cate_s"]  = cate_s
    store_df["cate_x"]  = cate_x
    store_df["cate_cf"] = cate_cf
    if run_bootstrap:
        store_df["cate_cf_lower"] = cate_cf_lower
        store_df["cate_cf_upper"] = cate_cf_upper

    print("\n" + "=" * 55)
    print("CATE Estimator Comparison")
    print("=" * 55)
    print(pd.DataFrame({
        "estimator": ["T-Learner", "S-Learner", "X-Learner", "CausalForest-DML"],
        "mean_cate": [cate_t.mean(), cate_s.mean(), cate_x.mean(), cate_cf.mean()],
        "sd_cate":   [cate_t.std(),  cate_s.std(),  cate_x.std(),  cate_cf.std()],
        "min_cate":  [cate_t.min(),  cate_s.min(),  cate_x.min(),  cate_cf.min()],
        "max_cate":  [cate_t.max(),  cate_s.max(),  cate_x.max(),  cate_cf.max()],
    }).round(3).to_string(index=False))

    return {
        "store_df":          store_df,
        "X":                 X,
        "T":                 T,
        "Y":                 Y,
        "subgroup_results":  subgroup_results,
        "cate_t":            cate_t,
        "cate_s":            cate_s,
        "cate_x":            cate_x,
        "cate_cf":           cate_cf,
        "cate_cf_lower":     cate_cf_lower if run_bootstrap else None,
        "cate_cf_upper":     cate_cf_upper if run_bootstrap else None,
        "feature_importance": importance_df,
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

    # Run full HTE pipeline
    results = run_hte_pipeline(
        panel,
        treated_groups=["5D_G2", "5D_G4"],
        control_group="5D_Control",
        outcome="collection_hrs",
        run_bootstrap=True,
        n_bootstrap=200,
        seed=42,
    )

    # Top stores with highest benefit (most negative CATE = biggest reduction)
    store_df = results["store_df"]
    print("\nTop 10 stores with largest treatment benefit (X-Learner):")
    top_stores = (
        store_df.nsmallest(10, "cate_x")
        [["store_id", "city", "region_type", "avg_daily_volume",
          "avg_utilization_rate", "cate_x"]]
    )
    print(top_stores.to_string(index=False))

    print("\nBottom 10 stores (smallest or negative benefit):")
    bottom_stores = (
        store_df.nlargest(10, "cate_x")
        [["store_id", "city", "region_type", "avg_daily_volume",
          "avg_utilization_rate", "cate_x"]]
    )
    print(bottom_stores.to_string(index=False))
