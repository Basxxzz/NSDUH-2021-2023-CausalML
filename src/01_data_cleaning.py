"""
01_data_cleaning.py

Prepare NSDUH 2021–2023 pooled data for the CausalML pipeline.

Steps (directly adapted from the original notebook Step 1–5):
1. Read pooled NSDUH_2021_2023.dta.
2. Keep relevant columns: outcome, exposures, covariates, weights, screening vars.
3. Convert all variables to numeric.
4. Restrict to women aged 18–49 (using SEXAGE / CATAG7 / AGE3 fallback).
5. Create derived variables (Y, exposure binaries, ANY_CANNA_EVER, IRPREG, ANY_NIC_EVER, ALCMON_bin).
6. Construct pooled survey weight W_NORM from ANALWT2_C1/C2/C3.
7. Build basic modeling subsets for QC (per exposure) and print summaries.
8. Save a single analysis-ready CSV for downstream scripts: nsduh_analysis.csv
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Paths

PROJECT_ROOT = Path("/Users/zach/Downloads/Phd_Prerequisites/Paper/Code/Code_Cell_repo")
RAW_DTA = PROJECT_ROOT / "NSDUH_2021_2023.dta"
OUTPUT_CSV = PROJECT_ROOT / "nsduh_analysis.csv"

SPECIAL_MISSINGS = {-9, 97, 98, 99}

# Helper functions

def to_numeric(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def vc(s: pd.Series, n: int = 8) -> dict:
    """Value counts (top n) as a small dict, including NaN."""
    return s.value_counts(dropna=False).head(n).to_dict()


def binarize_01(series: pd.Series,
                valid_pos=[1],
                valid_neg=[2, 0],
                specials=SPECIAL_MISSINGS) -> pd.Series:
    """
    Convert NSDUH categorical variables into 0/1 with NaN for special codes.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(~s.isin(specials))
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin(valid_pos)] = 1.0
    out[s.isin(valid_neg)] = 0.0
    return out


def choose_weight(row: pd.Series) -> float:
    """
    Pick the appropriate ANALWT2_* based on YEAR.
    """
    y = int(row["YEAR"]) if not pd.isna(row["YEAR"]) else np.nan
    if y == 2021:
        return row.get("ANALWT2_C1", np.nan)
    elif y == 2022:
        return row.get("ANALWT2_C2", np.nan)
    elif y == 2023:
        return row.get("ANALWT2_C3", np.nan)
    else:
        return np.nan


def normalize_weights(w: pd.Series) -> pd.Series:
    """
    Divide weights by 3 (for pooling 3 years) and rescale to mean ≈ 1.
    """
    w_div3 = w / 3.0
    mean_w = np.nanmean(w_div3)
    if (mean_w is not None) and (mean_w > 0):
        return w_div3 / mean_w
    return w_div3

# Step 1: Main pipeline

def main():
    # === Step 1: read pooled .dta and keep relevant columns ===
    if not RAW_DTA.exists():
        raise FileNotFoundError(f"Cannot find data file: {RAW_DTA}")

    print(f"[LOAD] Reading pooled NSDUH file:\n  {RAW_DTA}")
    df = pd.read_stata(RAW_DTA, convert_categoricals=False)

    # 1) Column lists: outcome, exposures, covariates, weights, screening
    Y_cols = ["IRAMDEYR"]  # outcome
    T_cols = ["ILLYR", "PNRNMYR", "STMNMYR", "MJEVER", "CBDHMPEVR"]  # exposures (including components)
    X_cols = [
        "AGE3", "IRPREG", "PREGNANT", "NEWRACE2", "EDUHIGHCAT", "INCOME",
        "IRINSUR4", "TOBFLAG", "NICVAPFLAG", "ALCMON", "YEAR"
    ]
    weight_cols = ["ANALWT2_C1", "ANALWT2_C2", "ANALWT2_C3"]
    screen_cols = ["IRSEX", "SEXAGE", "CATAG7"]

    keep_cols = list(dict.fromkeys(
        Y_cols + T_cols + X_cols + weight_cols + screen_cols
    ))

    present = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    print(f"[Step 1] Keep {len(present)} columns; missing {len(missing)} columns: "
          f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")

    df_small = df[present].copy()

    # 2) Convert to numeric
    for c in df_small.columns:
        df_small[c] = to_numeric(df_small[c])

    print("Shape after column selection:", df_small.shape)
    print(df_small.iloc[:2, :10])

    print("\n[QC] Key column distributions:")
    for c in ["IRSEX", "SEXAGE", "AGE3", "YEAR", "IRAMDEYR", "ILLYR", "PNRNMYR", "STMNMYR"]:
        if c in df_small.columns:
            print(c, vc(df_small[c]))

    # Step 2: Construct analysis sample (women 18–49)

    print("\n[Step 2] Build analysis sample (female 18–49)")

    # 1) Female only
    female_mask = (df_small["IRSEX"] == 2)
    print("Count of females:", int(female_mask.sum()), "/", len(df_small))

    # 2) Age selection: priority SEXAGE; fallback CATAG7; fallback AGE3
    age_mask = pd.Series(False, index=df_small.index)

    # 2.1 Check if SEXAGE behaves like a real age
    if "SEXAGE" in df_small.columns and df_small["SEXAGE"].notna().any():
        sa = df_small["SEXAGE"]
        is_yearlike = (
            sa.dropna().median() >= 12
            and sa.dropna().median() <= 80
            and sa.dropna().nunique() >= 10
        )
        print("Does SEXAGE look like actual age?", bool(is_yearlike),
              " | distribution:", vc(sa, n=10))
        if is_yearlike:
            age_mask = (sa >= 18) & (sa <= 49)

    # 2.2 Fallback to CATAG7: 2=18–25, 3=26–34, 4=35–49
    if not age_mask.any():
        if "CATAG7" in df_small.columns:
            print("Using CATAG7 as fallback. Distribution:", vc(df_small["CATAG7"], n=10))
            age_mask = df_small["CATAG7"].isin([2, 3, 4])

    # 2.3 Fallback to AGE3: 1=18–25, 2=26–34, 3=35–49
    if not age_mask.any():
        if "AGE3" in df_small.columns:
            print("Using AGE3 as fallback. Distribution:", vc(df_small["AGE3"], n=10))
            age_mask = df_small["AGE3"].isin([1, 2, 3])

    # 2.4 If still empty, warn
    if not age_mask.any():
        print("[Warning] Could not identify 18–49 age mapping. Please verify AGE variables.")

    # 3) Combine filters
    sample_mask = female_mask & age_mask
    df_analysis = df_small.loc[sample_mask].copy()
    print("Analysis sample size (female 18–49):", df_analysis.shape)

    print("\n[QC] Analysis sample basic:")
    for c in ["IRSEX", "SEXAGE", "CATAG7", "AGE3", "YEAR"]:
        if c in df_analysis.columns:
            print(c, vc(df_analysis[c], n=10))
    print("Sample size by YEAR:", df_analysis["YEAR"].value_counts(dropna=False).to_dict())

    # Pregnant-age women (for context)
    female_mask_all = (df_small["IRSEX"] == 2)
    age_mask_cat = df_small["CATAG7"].isin([2, 3, 4]) if "CATAG7" in df_small.columns else pd.Series(False, index=df_small.index)
    preg_age_mask = female_mask_all & age_mask_cat
    df_preg_age = df_small.loc[preg_age_mask]
    print("Women of reproductive age (18–49) count:", df_preg_age.shape[0])
    print("Proportion among total:", round(df_preg_age.shape[0] / df_small.shape[0] * 100, 2), "%")

    # Step 3: Derived variables (Y, T, X components)

    df_analysis = df_analysis.copy()

    # 2) Outcome Y
    df_analysis["Y"] = binarize_01(df_analysis["IRAMDEYR"], valid_pos=[1], valid_neg=[2, 0])

    # 3) Exposures
    df_analysis["ILLYR_bin"]   = binarize_01(df_analysis["ILLYR"])
    df_analysis["PNRNMYR_bin"] = binarize_01(df_analysis["PNRNMYR"])
    df_analysis["STMNMYR_bin"] = binarize_01(df_analysis["STMNMYR"])

    # 4) ANY_CANNA_EVER from MJEVER / CBDHMPEVR
    if ("MJEVER" in df_analysis.columns) and ("CBDHMPEVR" in df_analysis.columns):
        mj = binarize_01(df_analysis["MJEVER"], valid_pos=[1], valid_neg=[2])
        cbd = binarize_01(df_analysis["CBDHMPEVR"], valid_pos=[1], valid_neg=[2])
        df_analysis["ANY_CANNA_EVER"] = ((mj == 1) | (cbd == 1)).astype("float")
        df_analysis.loc[mj.isna() & cbd.isna(), "ANY_CANNA_EVER"] = np.nan
    else:
        df_analysis["ANY_CANNA_EVER"] = np.nan

    # 5) IRPREG from PREGNANT
    if "PREGNANT" in df_analysis.columns:
        df_analysis["IRPREG"] = binarize_01(df_analysis["PREGNANT"], valid_pos=[1], valid_neg=[2])
    else:
        df_analysis["IRPREG"] = np.nan

    # 6) ANY_NIC_EVER = TOBFLAG==1 or NICVAPFLAG==1
    if ("TOBFLAG" in df_analysis.columns) and ("NICVAPFLAG" in df_analysis.columns):
        tb = binarize_01(df_analysis["TOBFLAG"])
        vap = binarize_01(df_analysis["NICVAPFLAG"])
        df_analysis["ANY_NIC_EVER"] = ((tb == 1) | (vap == 1)).astype("float")
        df_analysis.loc[tb.isna() & vap.isna(), "ANY_NIC_EVER"] = np.nan
    else:
        df_analysis["ANY_NIC_EVER"] = np.nan

    # 7) ALCMON_bin
    if "ALCMON" in df_analysis.columns:
        df_analysis["ALCMON_bin"] = binarize_01(df_analysis["ALCMON"], valid_pos=[1], valid_neg=[2, 0])
    else:
        df_analysis["ALCMON_bin"] = np.nan

    print("\n[Step 3] Derived variable QC:")
    for c in [
        "Y", "ILLYR_bin", "PNRNMYR_bin", "STMNMYR_bin",
        "ANY_CANNA_EVER", "IRPREG", "ANY_NIC_EVER", "ALCMON_bin"
    ]:
        if c in df_analysis.columns:
            print(c, vc(df_analysis[c], n=8))

    # Step 4: Survey weights (RAW_W + W_NORM)

    df_analysis = df_analysis.copy()
    df_analysis["RAW_W"] = df_analysis.apply(choose_weight, axis=1)
    df_analysis["W_NORM"] = normalize_weights(df_analysis["RAW_W"])

    print("\n[Step 4] Weight QC:")
    print("RAW_W describe:")
    print(df_analysis["RAW_W"].describe())
    print("\nW_NORM describe (mean should be ≈ 1):")
    print(df_analysis["W_NORM"].describe())
    print("\nProportion with non-missing weights by YEAR:")
    print(df_analysis.groupby("YEAR")["RAW_W"].apply(lambda x: x.notna().mean()).to_dict())

    #Step 5: Build modeling datasets per exposure (for QC)

    covariate_cols = [
        "AGE3", "IRPREG", "NEWRACE2", "EDUHIGHCAT", "INCOME",
        "IRINSUR4", "ANY_NIC_EVER", "ALCMON_bin", "YEAR"
    ]

    exposure_map = {
        "ILLYR_bin": "ILLYR",
        "PNRNMYR_bin": "PNRNMYR",
        "STMNMYR_bin": "STMNMYR",
        "ANY_CANNA_EVER": "ANY_CANNA_EVER"
    }

    analysis_dfs = {}

    for t_col, label in exposure_map.items():
        if t_col not in df_analysis.columns:
            print(f"\n[{label}] skipped: {t_col} not found in df_analysis.")
            continue

        mask = (
            df_analysis["Y"].isin([0, 1]) &
            df_analysis[t_col].isin([0, 1]) &
            df_analysis["W_NORM"].notna()
        )

        cols = ["Y", t_col, "W_NORM"] + [c for c in covariate_cols if c in df_analysis.columns]
        dsub = df_analysis.loc[mask, cols].copy()

        if dsub.empty:
            print(f"\n[{label}] no rows after filtering for modeling subset.")
            continue

        dsub["Y"] = dsub["Y"].astype(int)
        dsub[t_col] = dsub[t_col].astype(int)
        analysis_dfs[label] = dsub

        print(f"\n[{label}] rows: {dsub.shape[0]}")
        p_treat = dsub[t_col].mean()
        p_y = dsub["Y"].mean()
        wt = dsub["W_NORM"]
        w_mean_treat = (dsub[t_col] * wt).sum() / wt.sum()
        w_mean_y = (dsub["Y"] * wt).sum() / wt.sum()
        print(f"  Exposure rate (unweighted): {p_treat:.3f}")
        print(f"  MDE rate (unweighted):      {p_y:.3f}")
        print(f"  Exposure rate (weighted):   {w_mean_treat:.3f}")
        print(f"  MDE rate (weighted):        {w_mean_y:.3f}")

    print("\nAll exposure-specific modeling subsets built (for QC).")

    # Save analysis-ready dataset

    # You can choose which columns to keep; here we keep everything in df_analysis
    df_out = df_analysis.copy()
    print("\n[FINAL] Analysis dataset shape:", df_out.shape)
    print("Columns:", list(df_out.columns))

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved analysis-ready data to:\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
