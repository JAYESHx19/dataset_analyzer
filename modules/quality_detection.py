"""
quality_detection.py
--------------------
Detects data quality issues: missing values, duplicates, outliers, type errors.
Also computes the Dataset Health Score (0-100).
"""

import pandas as pd
import numpy as np


def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing missing value counts and percentages per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"Column": df.columns, "Missing Count": missing.values, "Missing %": pct.values})
    return result[result["Missing Count"] > 0].reset_index(drop=True)


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return the count of duplicate rows."""
    return int(df.duplicated().sum())


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numeric columns using the IQR method.

    Returns:
        Dictionary: {column_name: outlier_count}
    """
    numeric_cols = df.select_dtypes(include="number").columns
    outlier_info = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = int(((col_data < lower) | (col_data > upper)).sum())
        if count > 0:
            outlier_info[col] = count
    return outlier_info


def detect_type_errors(df: pd.DataFrame) -> list:
    """
    Detect columns that appear numeric but are stored as object/string.

    Returns:
        List of column names suspected to have incorrect data types.
    """
    problematic = []
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_numeric(df[col].dropna(), errors="coerce")
            success_rate = converted.notna().mean()
            if success_rate > 0.85:  # >85% of values can be parsed as numbers
                problematic.append(col)
        except Exception:
            pass
    return problematic


def compute_health_score(df: pd.DataFrame) -> dict:
    """
    Compute a Dataset Health Score on a scale of 0–100.

    Weights:
        Missing values   -> 40%
        Duplicate rows   -> 20%
        Outliers         -> 25%
        Data type errors -> 15%
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

    total_rows = df.shape[0]
    total_cols = df.shape[1]

    duplicate_count = detect_duplicates(df)
    duplicate_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0

    outlier_info = detect_outliers_iqr(df)
    total_outliers = sum(outlier_info.values())
    outlier_pct = (total_outliers / total_rows * 100) if total_rows > 0 else 0

    type_errors = detect_type_errors(df)
    type_error_pct = (len(type_errors) / total_cols * 100) if total_cols > 0 else 0

    missing_score = 40 * (1 - min(missing_pct, 100) / 100)
    duplicate_score = 20 * (1 - min(duplicate_pct, 100) / 100)
    outlier_score = 25 * (1 - min(outlier_pct, 100) / 100)
    datatype_score = 15 * (1 - min(type_error_pct, 100) / 100)

    health_score = missing_score + duplicate_score + outlier_score + datatype_score

    if health_score >= 90:
        status = "🟢 Excellent dataset quality"
    elif health_score >= 75:
        status = "🟡 Good dataset quality"
    elif health_score >= 50:
        status = "🟠 Moderate quality — cleaning recommended"
    elif health_score >= 30:
        status = "🔴 Poor dataset quality"
    else:
        status = "⛔ Very poor dataset quality"

    return {
        "health_score": round(health_score, 1),
        "status": status,
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
        "type_error_pct": round(type_error_pct, 2),
        "missing_score": round(missing_score, 2),
        "duplicate_score": round(duplicate_score, 2),
        "outlier_score": round(outlier_score, 2),
        "datatype_score": round(datatype_score, 2),
        "outlier_info": outlier_info,
        "type_errors": type_errors,
        "duplicate_count": duplicate_count,
        "total_missing": total_missing,
        "total_outliers": total_outliers,
    }
