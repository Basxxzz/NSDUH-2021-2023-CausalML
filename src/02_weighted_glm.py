"""
02_weighted_glm.py

Baseline weighted logistic regression for NSDUH 2021–2023.
Direct translation of notebook Step 6.1 (complete-case weighted GLM).

Input:
    nsduh_analysis.csv  (output from 01_data_cleaning.py)

Output:
    results/ate_table_B_weighted_logit.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Paths

PROJECT_ROOT = Path("/Users/zach/Downloads/Phd_Prerequisites/Paper/Code/Code_Cell_repo")
DATA_PATH = PROJECT_ROOT / "nsduh_analysis.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "ate_table_B_weighted_logit.csv"


# Helper: complete-case weighted logistic regression

def fit_weighted_logit_complete_case(dsub: pd.DataFrame,
                                     treat_col: str,
                                     covariate_cols: list) -> dict:
    """
    Run a complete-case survey-weighted logistic regression:
        logit P(Y=1) ~ treat_col + covariates
    using W_NORM as frequency weight.
    """
    cols = ["Y", treat_col, "W_NORM"] + covariate_cols
    d = dsub[cols].replace([np.inf, -np.inf], np.nan)

    n0 = d.shape[0]
    miss_tbl = d.isna().sum().sort_values(ascending=False)

    # drop any row with missing in these columns (complete-case)
    d = d.dropna()
    n1 = d.shape[0]
    dropped = n0 - n1

    print(f"  - Original rows: {n0} | Dropped (missing): {dropped} ({(dropped / max(n0, 1)) * 100:.1f}%)")
    print("  - Top 5 variables with most missing values:")
    print(miss_tbl.head(5).to_string())

    if n1 == 0:
        raise ValueError("  No complete cases left after dropping missing values.")

    y = d["Y"].astype(int).values
    X = d[[treat_col] + covariate_cols].copy()
    X = sm.add_constant(X, has_constant="add")
    w = d["W_NORM"].values

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w)
    res = model.fit()

    OR = np.exp(res.params[treat_col])
    ci_low, ci_high = np.exp(res.conf_int().loc[treat_col])
    pval = res.pvalues[treat_col]

    return {
        "N_used": int(n1),
        "Dropped": int(dropped),
        "Exposure": treat_col,
        "OR": float(OR),
        "95%CI_low": float(ci_low),
        "95%CI_high": float(ci_high),
        "p_value": float(pval),
    }


# Main script

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find analysis file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[LOAD] Analysis dataset loaded: {df.shape}")

    # Covariates (same as notebook Step 5)
    covariate_cols = [
        "AGE3", "IRPREG", "NEWRACE2", "EDUHIGHCAT", "INCOME",
        "IRINSUR4", "ANY_NIC_EVER", "ALCMON_bin", "YEAR"
    ]

    for c in covariate_cols + ["Y", "W_NORM"]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in nsduh_analysis.csv")

    # Mapping from binary exposure column to human-readable label
    exposure_map = {
        "ILLYR_bin": "ILLYR",
        "PNRNMYR_bin": "PNRNMYR",
        "STMNMYR_bin": "STMNMYR",
        "ANY_CANNA_EVER": "ANY_CANNA_EVER",
    }

    results = []

    for t_col, label in exposure_map.items():
        if t_col not in df.columns:
            print(f"\n[Skip] {label}: column '{t_col}' not found in dataset.")
            continue

        # build modeling subset: valid Y, valid treatment, non-missing weights
        mask = (
            df["Y"].isin([0, 1]) &
            df[t_col].isin([0, 1]) &
            df["W_NORM"].notna()
        )

        cols = ["Y", t_col, "W_NORM"] + covariate_cols
        dsub = df.loc[mask, cols].copy()

        if dsub.empty:
            print(f"\n[Skip] {label}: no rows satisfy modeling conditions.")
            continue

        # basic QC (unweighted + weighted prevalence)
        dsub["Y"] = dsub["Y"].astype(int)
        dsub[t_col] = dsub[t_col].astype(int)

        print(f"\n[Baseline-B] {label}")
        print(f"  Rows before complete-case filtering: {dsub.shape[0]}")

        p_treat = dsub[t_col].mean()
        p_y = dsub["Y"].mean()
        wt = dsub["W_NORM"]
        w_mean_treat = (dsub[t_col] * wt).sum() / wt.sum()
        w_mean_y = (dsub["Y"] * wt).sum() / wt.sum()

        print(f"  Exposure prevalence (unweighted): {p_treat:.3f}")
        print(f"  MDE prevalence (unweighted):      {p_y:.3f}")
        print(f"  Exposure prevalence (weighted):   {w_mean_treat:.3f}")
        print(f"  MDE prevalence (weighted):        {w_mean_y:.3f}")

        # run weighted GLM (complete-case)
        res_dict = fit_weighted_logit_complete_case(dsub, t_col, covariate_cols)
        results.append(res_dict)

    if not results:
        raise RuntimeError("No GLM result was produced. Check data and filters.")

    results_df = pd.DataFrame(results)
    print("\n[STEP 6] Weighted GLM Results (complete-case):")
    print(results_df)

    # save table
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved GLM results to:\n  {RESULTS_CSV}")


if __name__ == "__main__":
    main()
