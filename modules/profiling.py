"""
profiling.py
------------
Computes comprehensive profiling statistics for a dataset.
"""

import pandas as pd
import numpy as np


def get_profile(df: pd.DataFrame) -> dict:
    """
    Generate a full profile of the DataFrame.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A dictionary with profiling metadata and statistics.
    """
    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(include=["object", "category"])

    # Per-column stats
    col_stats = []
    for col in df.columns:
        col_info = {
            "Column": col,
            "Type": str(df[col].dtype),
            "Missing": int(df[col].isnull().sum()),
            "Missing %": round(df[col].isnull().mean() * 100, 2),
            "Unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["Mean"] = round(df[col].mean(), 4) if not df[col].isna().all() else None
            col_info["Std"] = round(df[col].std(), 4) if not df[col].isna().all() else None
            col_info["Min"] = round(df[col].min(), 4) if not df[col].isna().all() else None
            col_info["Max"] = round(df[col].max(), 4) if not df[col].isna().all() else None
        else:
            col_info["Mean"] = None
            col_info["Std"] = None
            col_info["Min"] = None
            col_info["Max"] = None
        col_stats.append(col_info)

    return {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": categorical_df.columns.tolist(),
        "total_missing": int(df.isnull().sum().sum()),
        "total_missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "column_stats": col_stats,
        "describe": df.describe(include="all"),
    }
