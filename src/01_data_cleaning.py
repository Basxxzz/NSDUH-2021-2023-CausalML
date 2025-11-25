"""
01_data_cleaning.py

Clean NSDUH_2021_2023.dta and build the analysis dataset for the CausalML pipeline.

This script is a direct script version of the logic used in Yifan.ipynb:
- load pooled 2021–2023 data
- keep women 18–49 using SEXAGE / CATAG7 / AGE3
- keep relevant variables
- convert to numeric
- construct pooled weight W_NORM from ANALWT2_C1/C2/C3
- derive ANY_CANNA_EVER
- save to nsduh_cleaned.csv
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# paths

FILE = "/Users/zach/Downloads/Phd_Prerequisites/Paper/Code/Code_Cell_repo/NSDUH_2021_2023.dta"
OUTPUT = "/Users/zach/Downloads/Phd_Prerequisites/Paper/Code/Code_Cell_repo/nsduh_cleaned.csv"


# helper 

def vc(s, n=8):
    return s.value_counts(dropna=False).head(n).to_dict()


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


# main cleaning

def main():
    print("Loading:", FILE)
    df = pd.read_stata(FILE, convert_categoricals=False)

    SPECIAL_MISSINGS = {-9, 97, 98, 99}

    # 1) columns consistent with notebook
    Y_cols = ["IRAMDEYR"]  # outcome
    T_cols = ["ILLYR", "PNRNMYR", "STMNMYR", "MJEVER", "CBDHMPEVR"]  # exposures
    X_cols = [
        "AGE3", "IRPREG", "PREGNANT", "NEWRACE2", "EDUHIGHCAT", "INCOME",
        "IRINSUR4", "TOBFLAG", "NICVAPFLAG", "ALCMON", "YEAR"
    ]
    weight_cols = ["ANALWT2_C1", "ANALWT2_C2", "ANALWT2_C3"]
    screen_cols = ["IRSEX", "SEXAGE", "CATAG7"]

    keep_cols = list(dict.fromkeys(Y_cols + T_cols + X_cols + weight_cols + screen_cols))

    present = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    print(f"keep {len(present)} column；missing {len(missing)} column：{missing[:10]}{' ...' if len(missing) > 10 else ''}")

    df_small = df[present].copy()

    # Unified conversion to numerical values
    for c in df_small.columns:
        df_small[c] = to_num(df_small[c])

    print("shape:", df_small.shape)

    print("\n[QC] key column distribution：")
    for c in ["IRSEX", "SEXAGE", "AGE3", "YEAR", "IRAMDEYR", "ILLYR", "PNRNMYR", "STMNMYR"]:
        if c in df_small.columns:
            print(c, vc(df_small[c]))

    # === Step 2: analyze sample） ===
    print("\n[STEP 2] Construction of analysis samples（female 18–49）")

    # sex: female
    female_mask = (df_small["IRSEX"] == 2)
    print("count of female:", int(female_mask.sum()), "/", len(df_small))

    # age filter：SEXAGE → CATAG7 → AGE3
    age_mask = pd.Series(False, index=df_small.index)

    if "SEXAGE" in df_small.columns and df_small["SEXAGE"].notna().any():
        sa = df_small["SEXAGE"]
        is_yearlike = (
            sa.dropna().median() >= 12
            and sa.dropna().median() <= 80
            and sa.dropna().nunique() >= 10
        )
        print("SEXAGE", bool(is_yearlike), " | distribution:", vc(sa))
        if is_yearlike:
            age_mask = (sa >= 18) & (sa <= 49)

    if not age_mask.any() and "CATAG7" in df_small.columns:
        print("use CATAG7 back。distribution:", vc(df_small["CATAG7"]))
        age_mask = df_small["CATAG7"].isin([2, 3, 4])

    if not age_mask.any() and "AGE3" in df_small.columns:
        print("use AGE3 , distribution:", vc(df_small["AGE3"]))
        age_mask = df_small["AGE3"].isin([1, 2, 3])

    if not age_mask.any():
        print("[warning for debug")
        age_mask = pd.Series(True, index=df_small.index)

    sample_mask = female_mask & age_mask
    df_analysis = df_small.loc[sample_mask].copy()
    print("analyze sample size（female 18–49）:", df_analysis.shape)

    print("\n[QC] analyze sample basic：")
    for c in ["IRSEX", "SEXAGE", "CATAG7", "AGE3", "YEAR"]:
        if c in df_analysis.columns:
            print(c, vc(df_analysis[c]))
    if "YEAR" in df_analysis.columns:
        print("each year sample size：", df_analysis["YEAR"].value_counts(dropna=False).to_dict())

    # weight

    weight_cols = ["ANALWT2_C1", "ANALWT2_C2", "ANALWT2_C3"]

    def choose_weight(row):
        y = int(row["YEAR"]) if not pd.isna(row["YEAR"]) else np.nan
        if y == 2021:
            return row.get("ANALWT2_C1", np.nan)
        elif y == 2022:
            return row.get("ANALWT2_C2", np.nan)
        elif y == 2023:
            return row.get("ANALWT2_C3", np.nan)
        else:
            return np.nan

    def normalize_weights(w):
        w_div3 = w / 3.0
        mean_w = np.nanmean(w_div3)
        return w_div3 / mean_w

    # use join to avoid keyerror
    for wcol in weight_cols:
        if wcol not in df_small.columns:
            print(f" {wcol}，check origninal data")

    df_analysis["RAW_W"] = df_analysis.apply(choose_weight, axis=1)
    df_analysis["W_NORM"] = normalize_weights(df_analysis["RAW_W"])

    # build ANY_CANNA_EVER
    has_cannabis = (df_analysis.get("MJEVER", 0) == 1)
    has_hemp = (df_analysis.get("CBDHMPEVR", 0) == 1)
    df_analysis["ANY_CANNA_EVER"] = (has_cannabis | has_hemp).astype(int)

    # keep final datasets
    final_keep = [
        "YEAR",
        "W_NORM",
        "IRAMDEYR",
        "ILLYR", "PNRNMYR", "STMNMYR", "ANY_CANNA_EVER",
        "AGE3", "IRPREG", "NEWRACE2", "EDUHIGHCAT", "INCOME",
        "IRINSUR4", "ANY_NIC_EVER", "ALCMON",
    ]
    final_keep = [c for c in final_keep if c in df_analysis.columns]
    df_final = df_analysis[final_keep].copy()

    print("\n[FINAL]:", df_final.shape)
    print("column name:", list(df_final.columns))

    df_final.to_csv(OUTPUT, index=False)
    print(f"Saved cleaned data to: {OUTPUT}")


if __name__ == "__main__":
    main()
