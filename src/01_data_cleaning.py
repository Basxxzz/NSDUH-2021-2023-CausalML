#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_data_cleaning.py
------------------------------------
NSDUH 2021–2023 pooled dataset cleaning & analysis-ready dataset builder.

Outputs:
    - nsduh_analysis.csv         (analysis-level cleaned dataset)
    - analysis_dfs.pkl           (dict of exposure-specific modeling subsets)

Steps:
    1) Column subset & type harmonization
    2) Sample selection: women aged 18–49
    3) Derived variables (Y / exposures / covariates)
    4) Survey weight construction (RAW_W / W_NORM)
    5) Build analysis_dfs (four exposure-specific tables)

This script corresponds to notebook Steps 1–5.
"""

import os
import pickle
import numpy as np
import pandas as pd


# Utility: safe value_counts
def vc(s, n=10):
    return s.value_counts(dropna=False).head(n).to_dict()


# 0. Constants
SPECIAL_MISSINGS = {-9, 97, 98, 99}

Y_COLS = ["IRAMDEYR"]
T_COLS = ["ILLYR", "PNRNMYR", "STMNMYR", "MJEVER", "CBDHMPEVR"]
X_COLS = [
    "AGE3", "IRPREG", "PREGNANT", "NEWRACE2", "EDUHIGHCAT", "INCOME",
    "IRINSUR4", "TOBFLAG", "NICVAPFLAG", "ALCMON", "YEAR"
]
WEIGHT_COLS = ["ANALWT2_C1", "ANALWT2_C2", "ANALWT2_C3"]
SCREEN_COLS = ["IRSEX", "SEXAGE", "CATAG7"]

KEEP_COLS = list(dict.fromkeys(Y_COLS + T_COLS + X_COLS + WEIGHT_COLS + SCREEN_COLS))


# 1. Column subset & type cleaning
def load_and_subset(path):
    print(f"[LOAD] Reading pooled NSDUH file:\n  {path}")
    df = pd.read_stata(path, convert_categoricals=False)

    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    print(f"[Step 1] Keep {len(present)} columns; missing {len(missing)} columns: {missing}")

    df_small = df[present].copy()

    # convert all to numeric
    for c in df_small.columns:
        df_small[c] = pd.to_numeric(df_small[c], errors="coerce")

    print("Shape after column selection:", df_small.shape)
    print(df_small.iloc[:2, :10])

    # QC
    print("\n[QC] Key column distributions:")
    for c in ["IRSEX", "SEXAGE", "AGE3", "YEAR", "IRAMDEYR",
              "ILLYR", "PNRNMYR", "STMNMYR"]:
        if c in df_small.columns:
            print(c, vc(df_small[c]))

    return df_small


# 2. Sample selection: women aged 18–49
def select_women_18_49(df):
    print("\n[Step 2] Build analysis sample (female 18–49)")

    female_mask = (df["IRSEX"] == 2)
    print("Count of females:", int(female_mask.sum()), "/", len(df))

    # Try SEXAGE → if not real age → fallback CATAG7 → fallback AGE3
    age_mask = pd.Series(False, index=df.index)

    if "SEXAGE" in df.columns and df["SEXAGE"].notna().any():
        sa = df["SEXAGE"]
        is_yearlike = (
            sa.dropna().median() >= 12 and
            sa.dropna().median() <= 80 and
            sa.dropna().nunique() >= 10
        )
        print("Does SEXAGE look like actual age?", bool(is_yearlike),
              "| distribution:", vc(sa))
        if is_yearlike:
            age_mask = (sa >= 18) & (sa <= 49)

    if not age_mask.any():
        print("Using CATAG7 as fallback. Distribution:", vc(df["CATAG7"]))
        age_mask = df["CATAG7"].isin([2, 3, 4])  # 18–49 in NSDUH grouping

    if not age_mask.any():
        print("Using AGE3 as fallback. Distribution:", vc(df["AGE3"]))
        age_mask = df["AGE3"].isin([1, 2, 3])  # 18–49

    sample_mask = female_mask & age_mask
    df_analysis = df.loc[sample_mask].copy()

    print("Analysis sample size (female 18–49):", df_analysis.shape)

    print("\n[QC] Analysis sample basic:")
    for c in ["IRSEX", "SEXAGE", "CATAG7", "AGE3", "YEAR"]:
        if c in df_analysis.columns:
            print(c, vc(df_analysis[c]))
    print("Sample size by YEAR:", df_analysis["YEAR"].value_counts(dropna=False).to_dict())

    print("Women of reproductive age (18–49) count:", df_analysis.shape[0])
    print("Proportion among total:",
          round(df_analysis.shape[0] / df.shape[0] * 100, 2), "%")

    return df_analysis


# 3. Derived variables
def binarize_01(series, valid_pos=[1], valid_neg=[0, 2], specials=SPECIAL_MISSINGS):
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(~s.isin(specials))
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin(valid_pos)] = 1.0
    out[s.isin(valid_neg)] = 0.0
    return out


def add_derived_variables(df):
    df = df.copy()

    # Y
    df["Y"] = binarize_01(df["IRAMDEYR"], valid_pos=[1], valid_neg=[0, 2])

    # exposures
    df["ILLYR_bin"]   = binarize_01(df["ILLYR"])
    df["PNRNMYR_bin"] = binarize_01(df["PNRNMYR"])
    df["STMNMYR_bin"] = binarize_01(df["STMNMYR"])

    mj  = binarize_01(df["MJEVER"], valid_pos=[1], valid_neg=[2])
    cbd = binarize_01(df["CBDHMPEVR"], valid_pos=[1], valid_neg=[2])
    df["ANY_CANNA_EVER"] = ((mj == 1) | (cbd == 1)).astype("float")
    df.loc[mj.isna() & cbd.isna(), "ANY_CANNA_EVER"] = np.nan

    # pregnancy
    df["IRPREG"] = binarize_01(df["PREGNANT"], valid_pos=[1], valid_neg=[2])

    # nicotine
    tb  = binarize_01(df["TOBFLAG"])
    vap = binarize_01(df["NICVAPFLAG"])
    df["ANY_NIC_EVER"] = ((tb == 1) | (vap == 1)).astype("float")
    df.loc[tb.isna() & vap.isna(), "ANY_NIC_EVER"] = np.nan

    # alcohol
    df["ALCMON_bin"] = binarize_01(df["ALCMON"], valid_pos=[1], valid_neg=[0, 2])

    print("\n[Step 3] Derived variable QC:")
    for c in ["Y", "ILLYR_bin", "PNRNMYR_bin", "STMNMYR_bin", "ANY_CANNA_EVER",
              "IRPREG", "ANY_NIC_EVER", "ALCMON_bin"]:
        print(c, vc(df[c]))

    return df


# 4. Weight construction
def choose_weight(row):
    y = int(row["YEAR"]) if not pd.isna(row["YEAR"]) else np.nan
    if y == 2021:
        return row.get("ANALWT2_C1", np.nan)
    if y == 2022:
        return row.get("ANALWT2_C2", np.nan)
    if y == 2023:
        return row.get("ANALWT2_C3", np.nan)
    return np.nan


def normalize_weights(w):
    w2 = w / 3.0
    mean_w = np.nanmean(w2)
    return w2 / mean_w if mean_w and mean_w > 0 else w2


def add_weights(df):
    df = df.copy()

    df["RAW_W"] = df.apply(choose_weight, axis=1)
    df["W_NORM"] = normalize_weights(df["RAW_W"])

    print("\n[Step 4] Weight QC:")
    print("RAW_W describe:\n", df["RAW_W"].describe())
    print("\nW_NORM describe (mean≈1):\n", df["W_NORM"].describe())
    print("\nProportion with non-missing weights by YEAR:")
    print(df.groupby("YEAR")["RAW_W"].apply(lambda x: x.notna().mean()).to_dict())

    return df


# 5. Build exposure-specific modeling datasets
def build_analysis_dfs(df):
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
        mask = (
            df["Y"].isin([0, 1])
            & df[t_col].isin([0, 1])
            & df["W_NORM"].notna()
        )
        dsub = df.loc[mask, ["Y", t_col, "W_NORM"] + covariate_cols].copy()

        dsub["Y"] = dsub["Y"].astype(int)
        dsub[t_col] = dsub[t_col].astype(int)

        analysis_dfs[label] = dsub

        print(f"\n[{label}] rows:", dsub.shape[0])
        wt = dsub["W_NORM"]
        print("  Exposure rate (unweighted):", dsub[t_col].mean())
        print("  MDE rate (unweighted):     ", dsub["Y"].mean())
        print("  Exposure rate (weighted):  ", (dsub[t_col] * wt).sum() / wt.sum())
        print("  MDE rate (weighted):       ", (dsub["Y"] * wt).sum() / wt.sum())

    print("\nAll exposure-specific modeling subsets built (QC).")
    return analysis_dfs


# Main pipeline
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "NSDUH_2021_2023.dta")

    df0 = load_and_subset(data_path)
    df1 = select_women_18_49(df0)
    df2 = add_derived_variables(df1)
    df3 = add_weights(df2)

    # 5) build modeling datasets
    analysis_dfs = build_analysis_dfs(df3)

    # Save outputs
    out_csv = os.path.join(root, "nsduh_analysis.csv")
    df3.to_csv(out_csv, index=False)
    print("\n[FINAL] Saved analysis dataset to:\n ", out_csv)

    out_pkl = os.path.join(root, "analysis_dfs.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(analysis_dfs, f)
    print("[FINAL] Saved analysis_dfs.pkl to:\n ", out_pkl)


if __name__ == "__main__":
    main()
    
