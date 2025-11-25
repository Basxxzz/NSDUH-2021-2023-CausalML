"""
03_dml_ate.py

Double Machine Learning (ATE) for NSDUH 2021–2023.

This script mirrors the notebook Step 7:
- Uses GradientBoosting models for nuisance functions
- Linear regression for the final stage
- BootstrapInference for confidence intervals
- One script per exposure, with survey weights (W_NORM) as sample_weight

Input:
    nsduh_analysis.csv   (output of 01_data_cleaning.py)

Output:
    results/ate_table_DML.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression

from econml.dml import DML
from econml.inference import BootstrapInference


# Paths

PROJECT_ROOT = Path("/Users/zach/Downloads/Phd_Prerequisites/Paper/Code/Code_Cell_repo")

DATA_PATH = PROJECT_ROOT / "nsduh_analysis.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "ate_table_DML.csv"


# DML helper

def build_preprocessor(categorical_cols, numeric_cols):
    """
    Build column-wise preprocessing:
    - categorical: impute most frequent + one-hot
    - numeric: median imputation
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def run_dml_for_exposure(dsub: pd.DataFrame,
                         treat_col: str,
                         covariate_cols: list,
                         preprocessor: ColumnTransformer,
                         n_bootstrap: int = 500,
                         random_state: int = 42) -> dict:
    """
    Run DML ATE for a single binary exposure using econml.DML.

    - Complete-case on Y, T, W_NORM, covariates
    - Preprocessing on X via given preprocessor
    - GradientBoosting models for Y and T
    - Linear regression for the final stage
    - BootstrapInference for CIs
    """
    cols = ["Y", treat_col, "W_NORM"] + covariate_cols

    d = (
        dsub[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    n_used = d.shape[0]
    if n_used == 0:
        raise ValueError(f"No complete-case observations for {treat_col}.")

    y = d["Y"].astype(int).values
    t = d[treat_col].astype(int).values
    w = d["W_NORM"].values

    X_raw = d[covariate_cols].copy()
    X = preprocessor.fit_transform(X_raw)

    model_y = GradientBoostingRegressor(random_state=random_state)
    model_t = GradientBoostingClassifier(random_state=random_state)

    dml = DML(
        model_y=model_y,
        model_t=model_t,
        model_final=LinearRegression(),
        discrete_treatment=True,
        cv=3,
        random_state=random_state,
    )

    inf = BootstrapInference(n_bootstrap_samples=n_bootstrap)

    dml.fit(Y=y, T=t, X=X, sample_weight=w, inference=inf)

    ate = dml.ate(X=X)
    ci_low, ci_high = dml.ate_interval(X=X, alpha=0.05)

    return {
        "N_used": int(n_used),
        "Exposure": treat_col,
        "ATE": float(ate),
        "95%CI_low": float(ci_low),
        "95%CI_high": float(ci_high),
    }


# Main script

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find analysis file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[LOAD] Analysis dataset loaded: {df.shape}")

    # Covariates: same as in the notebook
    covariate_cols = [
        "AGE3",
        "IRPREG",
        "NEWRACE2",
        "EDUHIGHCAT",
        "INCOME",
        "IRINSUR4",
        "ANY_NIC_EVER",
        "ALCMON_bin",
        "YEAR",
    ]

    # Check required columns
    needed = covariate_cols + ["Y", "W_NORM"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in nsduh_analysis.csv")

    # Define exposures mapping (binary columns already exist in nsduh_analysis.csv)
    exposure_map = {
        "ILLYR_bin": "ILLYR",
        "PNRNMYR_bin": "PNRNMYR",
        "STMNMYR_bin": "STMNMYR",
        "ANY_CANNA_EVER": "ANY_CANNA_EVER",
    }

    # Preprocessing specification (as in notebook)
    categorical_cols = ["AGE3", "NEWRACE2", "EDUHIGHCAT", "IRINSUR4", "YEAR"]
    numeric_cols = ["IRPREG", "ANY_NIC_EVER", "ALCMON_bin", "INCOME"]

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    results = []

    for t_col, label in exposure_map.items():
        if t_col not in df.columns:
            print(f"\n[Skip] {label}: column '{t_col}' not found in dataset.")
            continue

        # Modeling subset: same filters as in GLM (valid Y, T, weights)
        mask = (
            df["Y"].isin([0, 1])
            & df[t_col].isin([0, 1])
            & df["W_NORM"].notna()
        )
        dsub = df.loc[mask, :].copy()

        if dsub.empty:
            print(f"\n[Skip] {label}: no observations satisfy modeling conditions.")
            continue

        print(f"\n[DML-ATE] {label}")
        print(f"  Rows before complete-case filtering: {dsub.shape[0]}")

        # run DML
        out = run_dml_for_exposure(
            dsub=dsub,
            treat_col=t_col,
            covariate_cols=covariate_cols,
            preprocessor=preprocessor,
            n_bootstrap=500,
            random_state=42,
        )
        print(
            f"  -> N_used={out['N_used']}, "
            f"ATE={out['ATE']:.3f}, "
            f"95% CI=({out['95%CI_low']:.3f}, {out['95%CI_high']:.3f})"
        )

        results.append(out)

    if not results:
        raise RuntimeError("No DML results produced. Check data and filters.")

    results_df = pd.DataFrame(results)
    print("\n[STEP 7] DML ATE Results:")
    print(results_df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved DML ATE results to:\n  {RESULTS_CSV}")


if __name__ == "__main__":
    main()
