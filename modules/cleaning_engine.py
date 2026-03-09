"""
cleaning_engine.py
------------------
Applies data cleaning strategies to a DataFrame based on user configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict


def apply_cleaning(
    df: pd.DataFrame,
    missing_strategy: Dict[str, str],
    remove_duplicates: bool = True,
    outlier_strategy: str = "keep",
    fix_types: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Apply a set of cleaning operations to a DataFrame.

    Args:
        df: The input pandas DataFrame.
        missing_strategy: Dict mapping column names to strategies ('mean', 'median', 'mode', 'drop').
        remove_duplicates: Whether to remove duplicate rows.
        outlier_strategy: One of 'keep', 'remove', or 'cap'.
        fix_types: Whether to attempt automatic type correction.

    Returns:
        Tuple of (cleaned DataFrame, list of applied operation descriptions).
    """
    cleaned_df = df.copy()
    log = []

    # --- Fix Data Types ---
    if fix_types:
        for col in cleaned_df.select_dtypes(include="object").columns:
            try:
                converted = pd.to_numeric(cleaned_df[col], errors="coerce")
                success_rate = converted.notna().mean()
                if success_rate > 0.85:
                    cleaned_df[col] = converted
                    log.append(f"🔧 Converted column `{col}` to numeric type.")
            except Exception:
                pass

    # --- Handle Missing Values ---
    for col, strategy in missing_strategy.items():
        if col not in cleaned_df.columns:
            continue
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count == 0:
            continue

        if strategy == "mean" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            fill_val = cleaned_df[col].mean()
            cleaned_df[col].fillna(fill_val, inplace=True)
            log.append(f"✅ Filled `{col}` missing values with mean ({fill_val:.4f}).")
        elif strategy == "median" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            fill_val = cleaned_df[col].median()
            cleaned_df[col].fillna(fill_val, inplace=True)
            log.append(f"✅ Filled `{col}` missing values with median ({fill_val:.4f}).")
        elif strategy == "mode":
            fill_val = cleaned_df[col].mode()
            if not fill_val.empty:
                cleaned_df[col].fillna(fill_val[0], inplace=True)
                log.append(f"✅ Filled `{col}` missing values with mode ({fill_val[0]}).")
        elif strategy == "drop":
            before = len(cleaned_df)
            cleaned_df.dropna(subset=[col], inplace=True)
            removed = before - len(cleaned_df)
            log.append(f"🗑️ Dropped {removed} rows with missing values in `{col}`.")

    # --- Remove Duplicates ---
    if remove_duplicates:
        before = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = before - len(cleaned_df)
        if removed > 0:
            log.append(f"🗑️ Removed {removed} duplicate rows.")

    # --- Handle Outliers ---
    if outlier_strategy in ("remove", "cap"):
        numeric_cols = cleaned_df.select_dtypes(include="number").columns
        for col in numeric_cols:
            col_data = cleaned_df[col].dropna()
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            if outlier_strategy == "remove":
                before = len(cleaned_df)
                cleaned_df = cleaned_df[
                    (cleaned_df[col].isna()) | ((cleaned_df[col] >= lower) & (cleaned_df[col] <= upper))
                ]
                removed = before - len(cleaned_df)
                if removed > 0:
                    log.append(f"🗑️ Removed {removed} outlier rows from `{col}`.")
            elif outlier_strategy == "cap":
                outlier_count = int(((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).sum())
                cleaned_df[col] = cleaned_df[col].clip(lower=lower, upper=upper)
                if outlier_count > 0:
                    log.append(f"📌 Capped {outlier_count} outliers in `{col}` to IQR bounds.")

    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df, log


def get_suggested_strategy(df: pd.DataFrame, col: str) -> str:
    """
    Suggest a missing value strategy for a column.

    Returns one of: 'mean', 'median', 'mode', 'drop'
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        # Use median for skewed distributions
        skewness = abs(df[col].skew())
        return "median" if skewness > 1.0 else "mean"
    else:
        return "mode"
