"""Feature engineering and augmentation utilities for time series models."""
from typing import List, Tuple
import pandas as pd

DIFF_SPECS: List[Tuple[str, int]] = [
    ("FI_Sp500", 1),
    ("FI_Sp500", 2),
    ("FI_Sp500", 3),
    ("FI_Nasdaq", 1),
    ("FI_Nasdaq", 2),
    ("FI_Nasdaq", 3),
    ("FI_Dow", 1),
    ("FI_Dow", 2),
    ("FI_Dow", 3),
    ("FI_FX_USDKRW", 1),
    ("FI_FX_JPYKRW", 1),
    ("FI_FX_CNYKRW", 1),
    ("FI_Gold", 1),
    ("FI_WTI", 1),
    ("KE_마스크", 3),
    ("KE_코로나", 3),
]

MOMENTUM_FEATURES_TARGET_1 = [
    "FI_Sp500_d1",
    "FI_Sp500_d3",
    "FI_Nasdaq_d1",
    "FI_Nasdaq_d3",
    "FI_Dow_d1",
    "FI_Dow_d3",
    "FI_FX_USDKRW_d1",
    "FI_Gold_d1",
    "FI_WTI_d1",
    "Weekday",
]

OPTIMIZED_FEATURES_TARGET_1 = [
    "FI_Sp500_d1",
    "FI_Sp500_d2",
    "FI_Sp500_d3",
    "FI_Nasdaq_d1",
    "FI_Nasdaq_d2",
    "FI_Nasdaq_d3",
    "FI_Dow_d1",
    "FI_Dow_d2",
    "FI_Dow_d3",
    "FI_FX_USDKRW_d1",
    "FI_FX_JPYKRW_d1",
    "FI_FX_CNYKRW_d1",
    "FI_Gold_d1",
    "FI_WTI_d1",
    "Spread_Corp_CD_d1",
    "Ratio_Copper_Gold_d1",
    "Spread_Nas_Dow_d1",
    "KE_마스크_d3",
    "KE_코로나_d3",
    "Weekday",
]


def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes causal difference/momentum and spread features along chronological Time_step.

    Expects df to be sorted by Time_step ascending.
    """
    out = df.copy()
    new_cols = {}

    # 1. Base differences
    for base_col, step in DIFF_SPECS:
        col_name = f"{base_col}_d{step}"
        if base_col in out.columns:
            new_cols[col_name] = out[base_col].diff(step)

    # 2. Macro spreads (Fundamental macro-economic and market barometers)
    # Credit/term spread: Corp 3yr - CD 91
    if "FI_Corp3yr" in out.columns and "FI_CD91" in out.columns:
        corp_cd = out["FI_Corp3yr"] - out["FI_CD91"]
        new_cols["Spread_Corp_CD_d1"] = corp_cd.diff(1)

    # Copper-Gold ratio (Dr. Copper / Safe haven Gold)
    if "FI_Guri" in out.columns and "FI_Gold" in out.columns:
        copper_gold = out["FI_Guri"] - out["FI_Gold"]
        new_cols["Ratio_Copper_Gold_d1"] = copper_gold.diff(1)

    # Nasdaq-Dow tech momentum spread
    if "FI_Nasdaq" in out.columns and "FI_Dow" in out.columns:
        nas_dow = out["FI_Nasdaq"] - out["FI_Dow"]
        new_cols["Spread_Nas_Dow_d1"] = nas_dow.diff(1)

    if new_cols:
        aug_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, aug_df], axis=1)
    return out
