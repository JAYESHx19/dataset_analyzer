"""
insight_engine.py
-----------------
Automatically extracts key insights and feature importance from the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_insights(df: pd.DataFrame) -> list[dict]:
    """
    Automatically generate a list of ranked insights from a DataFrame.

    Returns:
        A list of dicts with 'type', 'title', and 'description' keys.
    """
    insights = []
    numeric_df = df.select_dtypes(include="number")

    # --- Correlation insights ---
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr_idx = corr_matrix.stack().idxmax()
        max_corr_val = corr_matrix.stack().max()
        if max_corr_val > 0:
            insights.append({
                "type": "correlation",
                "title": f"Strongest Correlation: `{max_corr_idx[0]}` ↔ `{max_corr_idx[1]}`",
                "description": f"Correlation coefficient: **{max_corr_val:.2f}** — These two columns are strongly related.",
                "score": float(max_corr_val),
            })

    # --- Dominant category insights ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        top = df[col].value_counts(normalize=True).head(1)
        if not top.empty:
            dominant_val = top.index[0]
            dominant_pct = round(top.values[0] * 100, 1)
            insights.append({
                "type": "category",
                "title": f"Dominant Category in `{col}`: {dominant_val}",
                "description": f"**{dominant_val}** accounts for **{dominant_pct}%** of all values in `{col}`.",
                "score": dominant_pct / 100,
            })

    # --- High missing value warning ---
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 20:
            insights.append({
                "type": "warning",
                "title": f"High Missing Rate in `{col}`",
                "description": f"`{col}` has **{missing_pct:.1f}%** missing values — imputation or removal recommended.",
                "score": missing_pct / 100,
            })

    # --- Skewness insights ---
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        if abs(skew) > 2:
            direction = "positively" if skew > 0 else "negatively"
            insights.append({
                "type": "distribution",
                "title": f"Skewed Distribution: `{col}`",
                "description": f"`{col}` is **{direction} skewed** (skewness={skew:.2f}). Consider log-transformation.",
                "score": min(abs(skew) / 10, 1.0),
            })

    # Sort by score descending
    insights.sort(key=lambda x: x.get("score", 0), reverse=True)
    return insights[:10]  # Return top 10 insights


def compute_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature importance using correlation-based ranking.
    Each feature's importance is the mean of its absolute correlations with all others.

    Returns:
        DataFrame with columns [Feature, Importance]
    """
    df_enc = df.copy()

    # Encode categorical columns
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        try:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        except Exception:
            df_enc.drop(columns=[col], inplace=True)

    numeric_df = df_enc.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["Feature", "Importance"])

    corr = numeric_df.corr().abs()
    np.fill_diagonal(corr.values, 0)
    importance = corr.mean(axis=1).sort_values(ascending=False)

    return pd.DataFrame({"Feature": importance.index, "Importance": importance.values.round(4)})
