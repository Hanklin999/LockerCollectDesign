"""
data_generation.py
==================
Calibrated simulation of Shopee smart locker notification experiment.

Data Generating Process (DGP) is informed by real experimental observations:
- 2,000 stores (simplified from 2,913)
- 1,000 high-inventory (burst) stores, 1,000 low-inventory (vacant) stores
- Treatment assignment based on historical inventory level (not random)
- Outcomes calibrated to real observed collection hours and RTS rates

Experiment Design:
    Burst stores (1,000):
        5D Control : ~100 stores, D0+D4 (2 touches)
        5D G2      : ~100 stores, D0+D2+D4 (3 touches)
        5D G4      : ~100 stores, D0+D1+D2+D3+D4 (5 touches)
        6D         : ~700 stores, D0+D4 (2 touches, 6-day deadline)
    Vacant stores (1,000):
        7D         : ~1,000 stores, D0+D4 (2 touches, 7-day deadline)

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants calibrated from real experimental data
# ---------------------------------------------------------------------------

TAIWAN_CITIES = {
    "台北市":  {"count": 342, "type": "metro", "weight": 0.171},
    "新北市":  {"count": 497, "type": "metro", "weight": 0.248},
    "桃園市":  {"count": 284, "type": "metro", "weight": 0.142},
    "台中市":  {"count": 334, "type": "metro", "weight": 0.167},
    "台南市":  {"count": 178, "type": "metro", "weight": 0.089},
    "高雄市":  {"count": 325, "type": "metro", "weight": 0.163},
    "新竹縣市": {"count": 145, "type": "regional", "weight": 0.072},
    "基隆市":  {"count": 46,  "type": "regional", "weight": 0.023},
    "彰化縣":  {"count": 111, "type": "regional", "weight": 0.055},
    "屏東縣":  {"count": 93,  "type": "regional", "weight": 0.046},
    "其他":    {"count": 558, "type": "rural",    "weight": 0.279},
}

# Six major metros (六都) - burst stores concentrated here
SIX_METROS = ["台北市", "新北市", "桃園市", "台中市", "台南市", "高雄市"]

# Treatment group definitions
TREATMENT_GROUPS = {
    "5D_Control": {
        "n_stores":       100,
        "deadline_days":  5,
        "n_touches":      2,
        "touch_days":     [0, 4],
        "store_type":     "burst",
    },
    "5D_G2": {
        "n_stores":       100,
        "deadline_days":  5,
        "n_touches":      3,
        "touch_days":     [0, 2, 4],
        "store_type":     "burst",
    },
    "5D_G4": {
        "n_stores":       100,
        "deadline_days":  5,
        "n_touches":      5,
        "touch_days":     [0, 1, 2, 3, 4],
        "store_type":     "burst",
    },
    "6D": {
        "n_stores":       700,
        "deadline_days":  6,
        "n_touches":      2,
        "touch_days":     [0, 5],
        "store_type":     "burst",
    },
    "7D": {
        "n_stores":       1000,
        "deadline_days":  7,
        "n_touches":      2,
        "touch_days":     [0, 6],
        "store_type":     "vacant",
    },
}

# Collection hours: calibrated from real BM (2026-01-19) observations
COLLECTION_HRS_PARAMS = {
    "5D_Control": {"mean": 33.5, "treatment_effect": 0.0},
    "5D_G2":      {"mean": 33.5, "treatment_effect": -1.5},
    "5D_G4":      {"mean": 33.5, "treatment_effect": -1.7},
    "6D":         {"mean": 33.5, "treatment_effect": +0.5},
    "7D":         {"mean": 35.5, "treatment_effect": +2.0},
}

# RTS rate calibrated from real data
RTS_PARAMS = {
    "5D_Control": {"mean": 0.017, "treatment_effect": 0.0},
    "5D_G2":      {"mean": 0.017, "treatment_effect": -0.004},
    "5D_G4":      {"mean": 0.017, "treatment_effect": -0.003},
    "6D":         {"mean": 0.015, "treatment_effect": -0.001},
    "7D":         {"mean": 0.013, "treatment_effect": -0.002},
}

# Experiment timeline
EXPERIMENT_WEEKS = [
    {"week_id": 0, "date": "2026-01-19", "is_bm": True,  "seasonal_effect": 0.0},
    {"week_id": 1, "date": "2026-01-26", "is_bm": False, "seasonal_effect": 0.0},
    {"week_id": 2, "date": "2026-02-02", "is_bm": False, "seasonal_effect": 0.0},
    # 2026-02-09 excluded: CNY eve, buyers pickup faster due to holiday behavior
    {"week_id": 3, "date": "2026-02-23", "is_bm": False, "seasonal_effect": 0.0},
    {"week_id": 4, "date": "2026-03-02", "is_bm": False, "seasonal_effect": 0.0},
]


# ---------------------------------------------------------------------------
# Store metadata generation
# ---------------------------------------------------------------------------

def generate_store_metadata(
    n_stores: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate static store-level characteristics.

    Each store has fixed attributes determined before the experiment.
    These attributes act as confounders: they influence both treatment
    assignment (burst vs vacant) and the outcome (collection hours).

    Parameters
    ----------
    n_stores : int
        Total number of stores to simulate. Default 2,000.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per store with columns:
        [store_id, city, region_type, is_metro, capacity,
         avg_daily_volume, avg_utilization_rate, store_type,
         pct_closure_hours]
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Assign cities proportional to real Taiwan store distribution
    # ------------------------------------------------------------------
    city_names = list(TAIWAN_CITIES.keys())
    city_weights = np.array([v["weight"] for v in TAIWAN_CITIES.values()])
    city_weights = city_weights / city_weights.sum()  # normalise

    cities = rng.choice(city_names, size=n_stores, p=city_weights)
    is_metro = np.array([c in SIX_METROS for c in cities])
    region_type = np.where(
        is_metro,
        "metro",
        np.where(
            np.isin(cities, ["新竹縣市", "基隆市", "彰化縣", "屏東縣"]),
            "regional",
            "rural",
        ),
    )

    # ------------------------------------------------------------------
    # Store capacity: metro stores are smaller (300 pkgs), others ~400
    # Normal distribution, clipped to reasonable range
    # ------------------------------------------------------------------
    capacity_mean = np.where(is_metro, 300, 420)
    capacity = rng.normal(loc=capacity_mean, scale=40).clip(150, 600).astype(int)

    # ------------------------------------------------------------------
    # Daily volume: bimodal - low (~100) and high (~450) traffic stores
    # High traffic concentrated in metros
    # ------------------------------------------------------------------
    is_high_traffic = rng.binomial(1, p=np.where(is_metro, 0.55, 0.25))
    daily_volume = np.where(
        is_high_traffic,
        rng.normal(450, 60, n_stores).clip(300, 700),
        rng.normal(100, 25, n_stores).clip(30, 200),
    ).astype(int)

    # ------------------------------------------------------------------
    # Utilization rate: daily_volume / capacity
    # This is the KEY confounder - determines burst vs vacant assignment
    # ------------------------------------------------------------------
    avg_utilization_rate = (daily_volume / capacity).clip(0.1, 1.5)

    # ------------------------------------------------------------------
    # Pct closure hours: hours per day the store closes new orders
    # Burst definition: pct_closure_hours > 5%
    # Derived from utilization + noise
    # ------------------------------------------------------------------
    closure_base = (avg_utilization_rate - 0.6) * 0.3
    pct_closure_hours = (closure_base + rng.normal(0, 0.03, n_stores)).clip(0, 0.5)

    # ------------------------------------------------------------------
    # Store type: burst if pct_closure_hours > 5%, else vacant
    # This mirrors the real experiment's assignment mechanism
    # ------------------------------------------------------------------
    store_type = np.where(pct_closure_hours > 0.05, "burst", "vacant")

    df = pd.DataFrame({
        "store_id":            np.arange(n_stores),
        "city":                cities,
        "region_type":         region_type,
        "is_metro":            is_metro.astype(int),
        "capacity":            capacity,
        "avg_daily_volume":    daily_volume,
        "avg_utilization_rate": avg_utilization_rate.round(4),
        "pct_closure_hours":   pct_closure_hours.round(4),
        "store_type":          store_type,
    })

    return df


# ---------------------------------------------------------------------------
# Treatment assignment
# ---------------------------------------------------------------------------

def assign_treatment(
    store_metadata: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign treatment groups to stores.

    Assignment logic mirrors real experiment:
    - Burst stores (1,000) randomly split into 5D_Control/G2/G4/6D
      with sizes 100/100/100/700
    - Vacant stores (1,000) all assigned to 7D

    Parameters
    ----------
    store_metadata : pd.DataFrame
        Output of generate_store_metadata().
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        store_metadata with added 'treatment_group' column.
    """
    rng = np.random.default_rng(seed)
    df = store_metadata.copy()
    df["treatment_group"] = None

    # ------------------------------------------------------------------
    # Burst stores: randomly assign to 4 groups
    # Pilot design: small n for 5D groups (cost constraint),
    # larger n for 6D (closer to current policy, safer to scale)
    # ------------------------------------------------------------------
    burst_idx = df[df["store_type"] == "burst"].index.tolist()
    burst_idx = rng.permutation(burst_idx).tolist()

    # Take top 1,000 burst stores if more exist
    burst_idx = burst_idx[:1000]

    group_sizes = {
        "5D_Control": 100,
        "5D_G2":      100,
        "5D_G4":      100,
        "6D":         700,
    }

    pointer = 0
    for group, size in group_sizes.items():
        assigned = burst_idx[pointer: pointer + size]
        df.loc[assigned, "treatment_group"] = group
        pointer += size

    # ------------------------------------------------------------------
    # Vacant stores: all assigned to 7D
    # ------------------------------------------------------------------
    vacant_idx = df[df["store_type"] == "vacant"].index.tolist()
    vacant_idx = vacant_idx[:1000]
    df.loc[vacant_idx, "treatment_group"] = "7D"

    # Drop unassigned stores (if total burst > 1000 or vacant > 1000)
    df = df[df["treatment_group"].notna()].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Panel data generation (store × week)
# ---------------------------------------------------------------------------

def generate_store_panel(
    store_metadata: pd.DataFrame,
    noise_sd: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate weekly panel data for each store across experiment weeks.

    Outcome model:
        collection_hrs = base_mean
                       + treatment_effect * is_post
                       + store_fixed_effect
                       + week_noise
                       + seasonal_effect

    The treatment effect only applies post-BM (is_post = 1).
    BM week serves as the pre-treatment baseline for DiD.

    Parameters
    ----------
    store_metadata : pd.DataFrame
        Output of assign_treatment(), must contain 'treatment_group'.
    noise_sd : float
        Standard deviation of weekly noise. Calibrated to ±1.5hr
        observed in real data.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        [store_id, week_id, date, is_bm, is_post, treatment_group,
         store_type, city, region_type, is_metro, capacity,
         avg_daily_volume, avg_utilization_rate, pct_closure_hours,
         collection_hrs, rts_rate, complaint_rate, opt_out_rate]
    """
    rng = np.random.default_rng(seed)
    records = []

    for _, store in store_metadata.iterrows():
        group = store["treatment_group"]
        hrs_params = COLLECTION_HRS_PARAMS[group]
        rts_params = RTS_PARAMS[group]

        # Store fixed effect: slight variation across stores
        store_fe = rng.normal(0, 0.8)

        for week in EXPERIMENT_WEEKS:
            is_post = int(not week["is_bm"])

            # ----------------------------------------------------------
            # Collection hours outcome
            # Treatment effect only kicks in post-BM
            # ----------------------------------------------------------
            treatment_effect = hrs_params["treatment_effect"] * is_post
            seasonal = week["seasonal_effect"]
            week_noise = rng.normal(0, noise_sd)

            collection_hrs = float(np.clip(
                hrs_params["mean"]
                + treatment_effect
                + store_fe
                + seasonal
                + week_noise,
                10, 120,
            ))

            # ----------------------------------------------------------
            # RTS rate outcome (binary proportion, use beta-like noise)
            # ----------------------------------------------------------
            rts_base = rts_params["mean"] + rts_params["treatment_effect"] * is_post
            rts_noise = rng.normal(0, 0.002)
            rts_rate = float(np.clip(rts_base + rts_noise, 0.001, 0.25))

            # ----------------------------------------------------------
            # Guardrail metrics
            # More touches → slight increase in complaints and opt-outs
            # ----------------------------------------------------------
            n_touches = TREATMENT_GROUPS[group]["n_touches"]
            complaint_rate = float(np.clip(0.002 + 0.0003 * n_touches + rng.normal(0, 0.0005), 0, 0.02))
            opt_out_rate = float(np.clip(0.001 + 0.0002 * n_touches + rng.normal(0, 0.0003), 0, 0.01))

            records.append({
                "store_id":             int(store["store_id"]),
                "week_id":              week["week_id"],
                "date":                 week["date"],
                "is_bm":                int(week["is_bm"]),
                "is_post":              is_post,
                "treatment_group":      group,
                "store_type":           store["store_type"],
                "city":                 store["city"],
                "region_type":          store["region_type"],
                "is_metro":             int(store["is_metro"]),
                "capacity":             int(store["capacity"]),
                "avg_daily_volume":     int(store["avg_daily_volume"]),
                "avg_utilization_rate": float(store["avg_utilization_rate"]),
                "pct_closure_hours":    float(store["pct_closure_hours"]),
                "collection_hrs":       round(collection_hrs, 2),
                "rts_rate":             round(rts_rate, 5),
                "complaint_rate":       round(complaint_rate, 5),
                "opt_out_rate":         round(opt_out_rate, 5),
            })

    panel = pd.DataFrame(records)

    # Convenience columns for DiD
    panel["is_treated_5d"] = panel["treatment_group"].isin(
        ["5D_G2", "5D_G4"]
    ).astype(int)
    panel["is_g2"] = (panel["treatment_group"] == "5D_G2").astype(int)
    panel["is_g4"] = (panel["treatment_group"] == "5D_G4").astype(int)

    return panel


# ---------------------------------------------------------------------------
# High-risk parcel cohort (>96hr未取)
# ---------------------------------------------------------------------------

def generate_highrisk_cohort(
    store_panel: pd.DataFrame,
    pct_highrisk: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate high-risk parcel cohort (packages uncollected > 96 hours).

    This cohort represents ~15% of all parcels and is the primary
    driver of RTS. D4 notification timing is most impactful here.

    Real data reference:
        Control (no D4 intervention): ~124hr, RTS ~18%
        D4 17:00 intervention:        ~122hr, RTS ~15%

    Parameters
    ----------
    store_panel : pd.DataFrame
        Output of generate_store_panel().
    pct_highrisk : float
        Proportion of parcels that become high-risk. Default 0.15.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        High-risk cohort panel with same structure as store_panel
        but different outcome distributions.
    """
    rng = np.random.default_rng(seed)

    highrisk = store_panel.copy()

    # High-risk parcels have much longer collection hours
    highrisk["collection_hrs"] = rng.normal(
        loc=123.0,
        scale=8.0,
        size=len(highrisk)
    ).clip(96, 200).round(2)

    # D4 17:00 intervention effect
    d4_effect = np.where(
        highrisk["treatment_group"].isin(["5D_G2", "5D_G4", "5D_Control"]) &
        (highrisk["is_post"] == 1),
        -2.0,
        0.0,
    )
    highrisk["collection_hrs"] = (highrisk["collection_hrs"] + d4_effect).clip(96, 200)

    # RTS rate much higher for this cohort
    highrisk["rts_rate"] = rng.normal(
        loc=0.165,
        scale=0.015,
        size=len(highrisk)
    ).clip(0.10, 0.25).round(5)

    # D4 intervention reduces RTS
    rts_effect = np.where(
        highrisk["treatment_group"].isin(["5D_G2", "5D_G4", "5D_Control"]) &
        (highrisk["is_post"] == 1),
        -0.03,
        0.0,
    )
    highrisk["rts_rate"] = (highrisk["rts_rate"] + rts_effect).clip(0.05, 0.25).round(5)
    highrisk["cohort"] = "high_risk"
    store_panel["cohort"] = "normal"

    return highrisk


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def generate_all_data(
    n_stores: int = 2000,
    seed: int = 42,
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full data generation pipeline.

    Steps:
        1. Generate store metadata (static characteristics)
        2. Assign treatment groups
        3. Generate weekly panel outcomes
        4. Generate high-risk parcel cohort

    Parameters
    ----------
    n_stores : int
        Total stores to simulate.
    seed : int
        Master random seed.
    output_dir : str
        Directory to save CSV outputs.

    Returns
    -------
    Tuple of (store_metadata, store_panel, highrisk_panel)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1/4: Generating store metadata...")
    metadata = generate_store_metadata(n_stores=n_stores, seed=seed)

    print("Step 2/4: Assigning treatment groups...")
    metadata = assign_treatment(metadata, seed=seed)

    print("Step 3/4: Generating weekly panel data...")
    panel = generate_store_panel(metadata, seed=seed)

    print("Step 4/4: Generating high-risk parcel cohort...")
    highrisk = generate_highrisk_cohort(panel, seed=seed)

    # Save outputs
    metadata_path = f"{output_dir}/store_metadata.csv"
    panel_path = f"{output_dir}/store_panel.csv"
    highrisk_path = f"{output_dir}/store_panel_highrisk.csv"

    metadata.to_csv(metadata_path, index=False)
    panel.to_csv(panel_path, index=False)
    highrisk.to_csv(highrisk_path, index=False)

    print(f"\nDone. Files saved to {output_dir}/")
    print(f"  store_metadata.csv    : {len(metadata):,} stores")
    print(f"  store_panel.csv       : {len(panel):,} store-week observations")
    print(f"  store_panel_highrisk.csv : {len(highrisk):,} high-risk observations")

    # Summary
    print("\nTreatment group distribution:")
    print(
        metadata.groupby("treatment_group")
        .agg(n_stores=("store_id", "count"))
        .to_string()
    )

    print("\nCollection hours by group (post-BM average):")
    post = panel[panel["is_post"] == 1]
    print(
        post.groupby("treatment_group")["collection_hrs"]
        .agg(["mean", "std"])
        .round(2)
        .to_string()
    )

    return metadata, panel, highrisk


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metadata, panel, highrisk = generate_all_data(
        n_stores=2000,
        seed=42,
        output_dir="data/processed",
    )
