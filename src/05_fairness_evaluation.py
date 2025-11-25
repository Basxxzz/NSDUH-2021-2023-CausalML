#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_fairness_evaluation.py
------------------------------------
Fairness diagnostics for NSDUH MDE risk models.

Inputs:
    - analysis_dfs.pkl
        dict[str, pd.DataFrame] with keys like:
            "ILLYR", "PNRNMYR", "STMNMYR", "ANY_CANNA_EVER"
        Each DataFrame must contain:
            - Y (0/1 outcome)
            - W_NORM (normalized survey weight)
            - A single treatment column:
                  * name endswith "_bin"  (e.g. "ILLYR_bin")
                  * or exactly "ANY_CANNA_EVER"
            - Covariates:
                  AGE3, IRPREG, NEWRACE2, EDUHIGHCAT, INCOME,
                  IRINSUR4, ANY_NIC_EVER, ALCMON_bin, YEAR

Outputs:
    - fairness_table.csv        (weighted TPR / FPR / PPV by group)
    - fairness_table_boot.csv   (same metrics with bootstrap CIs)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

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

CATEGORICAL_COLS: List[str] = ["AGE3", "NEWRACE2", "EDUHIGHCAT", "IRINSUR4", "YEAR"]
NUMERIC_COLS: List[str] = ["IRPREG", "ANY_NIC_EVER", "ALCMON_bin", "INCOME"]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def detect_treatment_column(df: pd.DataFrame) -> str:
    """Return the treatment column name in df."""
    bin_cols = [c for c in df.columns if c.endswith("_bin")]
    if bin_cols:
        return bin_cols[0]
    if "ANY_CANNA_EVER" in df.columns:
        return "ANY_CANNA_EVER"
    raise ValueError("No treatment column found (expected *_bin or ANY_CANNA_EVER).")


def build_risk_pipeline(treat_col: str) -> Pipeline:
    """Return a sklearn Pipeline used to model risk p(Y=1 | X)."""
    # We explicitly construct the column lists used in the transformer.
    # group columns are not included here.
    feature_cat = CATEGORICAL_COLS
    feature_num = [treat_col] + NUMERIC_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]), feature_cat),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
            ]), feature_num),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])
    return pipe


def weighted_confusion(
    df: pd.DataFrame,
    pred_col: str = "pred",
    y_col: str = "Y",
    w_col: str = "W_NORM",
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute weighted TP, FP, TN, FN and derived TPR, FPR, PPV.
    """
    y = df[y_col].astype(int)
    p = df[pred_col].astype(int)
    w = df[w_col]

    tp = ((p == 1) & (y == 1)) * w
    fp = ((p == 1) & (y == 0)) * w
    tn = ((p == 0) & (y == 0)) * w
    fn = ((p == 0) & (y == 1)) * w

    tp, fp, tn, fn = tp.sum(), fp.sum(), tn.sum(), fn.sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    return tp, fp, tn, fn, tpr, fpr, ppv


# ----------------------------------------------------------------------
# Main fairness metrics (single pass, no bootstrap)
# ----------------------------------------------------------------------

def train_risk_model(
    dsub: pd.DataFrame,
    treat_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit a weighted logistic risk model on dsub and return a copy
    with predicted risk p_hat and (optionally) group_col carried along.
    """
    cols = ["Y", treat_col, "W_NORM"] + COVARIATE_COLS
    if group_col is not None and group_col not in cols and group_col in dsub.columns:
        cols = cols + [group_col]

    data = dsub[cols].copy()

    y = data["Y"].astype(int).values
    w = data["W_NORM"].values

    feature_cols = [treat_col] + COVARIATE_COLS
    X = data[feature_cols].copy()

    # FIX: remove duplicate columns safely
    X = X.loc[:, ~X.columns.duplicated()]

    pipe = build_risk_pipeline(treat_col)
    pipe.fit(X, y, clf__sample_weight=w)

    data["p_hat"] = pipe.predict_proba(X)[:, 1]
    return data


def fairness_metrics(
    dsub: pd.DataFrame,
    treat_col: str,
    group_col: str,
    threshold: str | float = "prevalence",
) -> pd.DataFrame:
    """
    Weighted TPR / FPR / PPV by group using a single fitted risk model.
    """
    scored = train_risk_model(dsub, treat_col, group_col=group_col)

    if threshold == "prevalence":
        thr = (scored["Y"] * scored["W_NORM"]).sum() / scored["W_NORM"].sum()
    else:
        thr = float(threshold)

    scored["pred"] = (scored["p_hat"] >= thr).astype(int)

    rows = []
    for level, gdf in scored.groupby(group_col):
        tp, fp, tn, fn, tpr, fpr, ppv = weighted_confusion(gdf)
        rows.append(
            {
                "Exposure": treat_col,
                "Group": group_col,
                "Level": level,
                "Threshold": round(thr, 4),
                "TPR": tpr,
                "FPR": fpr,
                "PPV": ppv,
            }
        )

    res = pd.DataFrame(rows)
    # Centered gaps relative to group mean
    res["TPR_gap"] = res["TPR"] - res["TPR"].mean()
    res["FPR_gap"] = res["FPR"] - res["FPR"].mean()
    res["PPV_gap"] = res["PPV"] - res["PPV"].mean()
    return res


# ----------------------------------------------------------------------
# Bootstrap CIs for group metrics
# ----------------------------------------------------------------------

def collapse_newrace2(
    df: pd.DataFrame,
    max_code: int = 5,
    new_col: str = "NEWRACE2_collapsed",
) -> pd.DataFrame:
    """
    Optionally collapse NEWRACE2 > max_code into max_code.
    """
    out = df.copy()
    if "NEWRACE2" in out.columns:
        out[new_col] = out["NEWRACE2"].where(out["NEWRACE2"] <= max_code, max_code)
    return out


def _bootstrap_group_ci(
    scored: pd.DataFrame,
    group_col: str,
    B: int = 300,
    rng: int = 42,
) -> pd.DataFrame:
    """
    For each group level, compute (TPR, FPR, PPV) and bootstrap CIs.
    """
    rows = []
    rng_state = np.random.RandomState(rng)

    for level, gdf in scored.groupby(group_col):
        gdf = gdf.copy()
        n_group = len(gdf)
        tp, fp, tn, fn, tpr, fpr, ppv = weighted_confusion(gdf)

        # Default: point estimate only
        tpr_lo = tpr_hi = tpr
        fpr_lo = fpr_hi = fpr
        ppv_lo = ppv_hi = ppv
        flag_sparse = 0

        # Treat very small groups as sparse
        if n_group < 50:
            flag_sparse = 1
        else:
            tpr_samples = []
            fpr_samples = []
            ppv_samples = []

            idx = np.arange(n_group)
            for _ in range(B):
                bs_idx = rng_state.choice(idx, size=n_group, replace=True)
                bs = gdf.iloc[bs_idx]
                _, _, _, _, tpr_b, fpr_b, ppv_b = weighted_confusion(bs)
                tpr_samples.append(tpr_b)
                fpr_samples.append(fpr_b)
                ppv_samples.append(ppv_b)

            def _ci(values: List[float]) -> Tuple[float, float]:
                arr = np.array(values, dtype=float)
                return (
                    float(np.nanpercentile(arr, 2.5)),
                    float(np.nanpercentile(arr, 97.5)),
                )

            tpr_lo, tpr_hi = _ci(tpr_samples)
            fpr_lo, fpr_hi = _ci(fpr_samples)
            ppv_lo, ppv_hi = _ci(ppv_samples)

        rows.append(
            {
                "Level": level,
                "N_group": n_group,
                "TPR": tpr,
                "TPR_lo": tpr_lo,
                "TPR_hi": tpr_hi,
                "FPR": fpr,
                "FPR_lo": fpr_lo,
                "FPR_hi": fpr_hi,
                "PPV": ppv,
                "PPV_lo": ppv_lo,
                "PPV_hi": ppv_hi,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "flag_sparse": flag_sparse,
            }
        )

    return pd.DataFrame(rows)


def fairness_bootstrap(
    dsub: pd.DataFrame,
    treat_col: str,
    group_col: str,
    threshold: str | float = "prevalence",
    B: int = 300,
) -> pd.DataFrame:
    """
    Compute fairness metrics with bootstrap CIs for one exposure × group.
    """
    scored = train_risk_model(dsub, treat_col, group_col=group_col)

    if threshold == "prevalence":
        thr = (scored["Y"] * scored["W_NORM"]).sum() / scored["W_NORM"].sum()
    else:
        thr = float(threshold)

    scored["pred"] = (scored["p_hat"] >= thr).astype(int)

    if group_col not in scored.columns:
        raise KeyError(f"group_col '{group_col}' not found in scored dataframe")

    ci_table = _bootstrap_group_ci(scored, group_col=group_col, B=B, rng=42)
    ci_table.insert(0, "Group", group_col)
    ci_table.insert(0, "Exposure", treat_col)
    ci_table.insert(2, "Threshold", round(thr, 4))
    return ci_table


# ----------------------------------------------------------------------
# Top-level pipeline
# ----------------------------------------------------------------------

def load_analysis_dfs(path: str) -> Dict[str, pd.DataFrame]:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj  # type: ignore[return-value]


def run_fairness_pipeline(
    input_path: str,
    out_main: str,
    out_boot: str,
    bootstrap_B: int = 300,
) -> None:
    print(f"[INFO] Loading analysis datasets from: {input_path}")
    analysis_dfs = load_analysis_dfs(input_path)

    # 1) Single-pass fairness metrics
    fairness_tables: List[pd.DataFrame] = []
    basic_groups = ["IRPREG", "NEWRACE2", "YEAR"]

    for label, dsub in analysis_dfs.items():
        treat_col = detect_treatment_column(dsub)
        print(f"[Fairness] {label}")

        for gcol in basic_groups:
            if gcol in dsub.columns:
                tab = fairness_metrics(dsub, treat_col, group_col=gcol, threshold="prevalence")
                fairness_tables.append(tab)

    fairness_df = pd.concat(fairness_tables, ignore_index=True)
    print("\n[STEP 9] Fairness diagnostics (weighted):")
    print(fairness_df.head(15))

    os.makedirs(os.path.dirname(out_main), exist_ok=True)
    fairness_df.to_csv(out_main, index=False)
    print(f"[INFO] Saved fairness table to:\n  {out_main}")

    # 2) Bootstrap CIs
    boot_tables: List[pd.DataFrame] = []
    for label, dsub in analysis_dfs.items():
        treat_col = detect_treatment_column(dsub)
        dsub2 = collapse_newrace2(dsub)

        candidate_groups = ["IRPREG", "NEWRACE2", "NEWRACE2_collapsed", "YEAR"]
        groups_existing = [g for g in candidate_groups if g in dsub2.columns]

        for gcol in groups_existing:
            print(f"[Fairness-Boot] {label} × {gcol}")
            tab = fairness_bootstrap(
                dsub2,
                treat_col,
                group_col=gcol,
                threshold="prevalence",
                B=bootstrap_B,
            )
            boot_tables.append(tab)

    fairness_boot_df = pd.concat(boot_tables, ignore_index=True)
    print("\n[STEP 9b] Fairness with bootstrap CIs:")
    print(fairness_boot_df.head(15))

    os.makedirs(os.path.dirname(out_boot), exist_ok=True)
    fairness_boot_df.to_csv(out_boot, index=False)
    print(f"[INFO] Saved bootstrap fairness table to:\n  {out_boot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fairness evaluation for NSDUH MDE risk models."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="analysis_dfs.pkl",
        help="Path to analysis_dfs.pkl.",
    )
    parser.add_argument(
        "--out-main",
        type=str,
        default="results/fairness_table.csv",
        help="Output CSV for main fairness metrics.",
    )
    parser.add_argument(
        "--out-boot",
        type=str,
        default="results/fairness_table_boot.csv",
        help="Output CSV for bootstrap fairness metrics.",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=300,
        help="Number of bootstrap replicates per group.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fairness_pipeline(
        input_path=args.input,
        out_main=args.out_main,
        out_boot=args.out_boot,
        bootstrap_B=args.B,
    )
