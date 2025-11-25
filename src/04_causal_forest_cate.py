#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_causal_forest_cate.py

Estimate subgroup CATEs using CausalForestDML for multiple binary exposures.

Expected input:
    - A pickle file containing a dict[str, pd.DataFrame], e.g.:
        {
            "ILLYR": df_illyr,
            "PNRNMYR": df_pnrnmyr,
            "STMNMYR": df_stmnmyr,
            "ANY_CANNA_EVER": df_canna
        }
      Each DataFrame must contain:
        - "Y": binary outcome (past-year MDE, 0/1)
        - "W_NORM": normalized survey weight
        - Covariates: AGE3, IRPREG, NEWRACE2, EDUHIGHCAT, INCOME,
                      IRINSUR4, ANY_NIC_EVER, ALCMON_bin, YEAR
        - One treatment column that is either:
              * endswith("_bin")   (e.g. "ILLYR_bin", "PNRNMYR_bin", etc.)
              * or exactly "ANY_CANNA_EVER"

Output:
    - A single CSV file with subgroup mean CATEs:
        results/subgroup_ate_table.csv  (can be overridden by --output)
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Configuration

CATEGORICAL_COLS: List[str] = ["AGE3", "NEWRACE2", "EDUHIGHCAT", "IRINSUR4", "YEAR"]
NUMERIC_COLS: List[str] = ["IRPREG", "ANY_NIC_EVER", "ALCMON_bin", "INCOME"]

# Columns to be used as covariates X in the causal forest
COVARIATE_COLS: List[str] = [
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


# Preprocessing

def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used to preprocess covariates X.
    - Categorical: impute most frequent, then one-hot encode
    - Numeric:     median imputation
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("num", numeric_pipeline, NUMERIC_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


# CATE estimation for a single exposure

def detect_treatment_column(df: pd.DataFrame) -> str:
    """
    Detect the treatment column in a given analysis DataFrame.

    Logic:
        - Prefer first column that ends with '_bin'
        - If none, fall back to 'ANY_CANNA_EVER'
    """
    candidate_cols = [c for c in df.columns if c.endswith("_bin")]
    if candidate_cols:
        return candidate_cols[0]
    if "ANY_CANNA_EVER" in df.columns:
        return "ANY_CANNA_EVER"
    raise ValueError("No valid treatment column found in DataFrame.")


def run_causal_forest_cate(
    df: pd.DataFrame,
    treatment_col: str,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """
    Run CausalForestDML on a single exposure and return subgroup mean CATEs.

    Subgroups:
        - pregnancy status: IRPREG (0/1)
        - income terciles:  qcut(INCOME, 3)
        - race groups:      NEWRACE2 codes

    Returns:
        DataFrame with columns:
            ["Exposure", "Group", "Level", "CATE"]
    """
    required_cols = ["Y", treatment_col, "W_NORM"] + COVARIATE_COLS

    # 1) basic cleaning04_causal_forest_cate.py
    data = (
        df[required_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    # 2) Column name deduplication (if there are duplicates, only keep the column that appears for the first time)
    # This step does not change any statistical meaning; it only avoids technical conflicts.
    data = data.loc[:, ~data.columns.duplicated()]

    if data.empty:
        raise ValueError(f"Data for treatment {treatment_col} is empty after cleaning.")

    y = data["Y"].astype(int).values
    t = data[treatment_col].astype(int).values
    w = data["W_NORM"].values

    # 3) covariates
    x_raw = data[COVARIATE_COLS].copy()
    x_raw = x_raw.loc[:, ~x_raw.columns.duplicated()]

    # print(f"[DEBUG] {treatment_col} | x_raw columns:", list(x_raw.columns))

    model_y = GradientBoostingRegressor(random_state=42)
    model_t = GradientBoostingClassifier(random_state=42)

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=50,
        random_state=42,
    )

    # 4) fitting causal forest
    cf.fit(Y=y, T=t, X=x_raw, sample_weight=w)
    cate_pred = cf.effect(x_raw)

    data = data.reset_index(drop=True)
    data["CATE"] = cate_pred

    # 5) Subgroup CATE summaries
    subgroup_results = []

    # (1) Pregnancy status
    pregnancy_means = data.groupby("IRPREG")["CATE"].mean()
    for level, cate in pregnancy_means.items():
        subgroup_results.append(
            {
                "Exposure": treatment_col,
                "Group": "pregnancy",
                "Level": str(level),
                "CATE": float(cate),
            }
        )

    # (2) Income terciles
    income_bins = pd.qcut(data["INCOME"], 3, duplicates="drop")
    income_means = data.groupby(income_bins)["CATE"].mean()
    for bin_label, cate in income_means.items():
        subgroup_results.append(
            {
                "Exposure": treatment_col,
                "Group": "income_tercile",
                "Level": str(bin_label),
                "CATE": float(cate),
            }
        )

    # (3) Race groups
    race_means = data.groupby("NEWRACE2")["CATE"].mean()
    for level, cate in race_means.items():
        subgroup_results.append(
            {
                "Exposure": treatment_col,
                "Group": "race",
                "Level": str(level),
                "CATE": float(cate),
            }
        )

    return pd.DataFrame(subgroup_results)


# Main pipeline

def load_analysis_datasets(path: str) -> Dict[str, pd.DataFrame]:
    """
    Load the analysis datasets dictionary from a pickle file.

    The pickle is expected to contain a mapping:
        exposure_label (str) -> pd.DataFrame
    """
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(obj)}.")
    return obj  # type: ignore[return-value]


def run_pipeline(input_path: str, output_path: str) -> None:
    """
    Run CATE estimation via CausalForestDML for all exposures in the
    analysis_dfs dict and save subgroup mean CATEs to a CSV file.
    """
    print(f"[INFO] Loading analysis datasets from: {input_path}")
    analysis_dfs = load_analysis_datasets(input_path)

    preprocessor = build_preprocessor()

    all_results = []

    for exposure_label, subset_df in analysis_dfs.items():
        print(f"[CATE] Running CausalForestDML for exposure: {exposure_label}")

        treatment_col = detect_treatment_column(subset_df)
        cate_df = run_causal_forest_cate(
            df=subset_df,
            treatment_col=treatment_col,
            preprocessor=preprocessor,
        )
        all_results.append(cate_df)

    results_df = pd.concat(all_results, ignore_index=True)

    print("\n[STEP 8] CATE subgroup means (first 15 rows):")
    print(results_df.head(15))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved subgroup CATE table to:\n  {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate subgroup CATEs via CausalForestDML."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/analysis_dfs.pkl",
        help="Path to pickle file containing analysis_dfs dict.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/subgroup_ate_table.csv",
        help="Path to output CSV file for subgroup CATEs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(input_path=args.input, output_path=args.output)
