"""
advanced_analysis.py
--------------------
Data Drift detection and dataset comparison between two uploaded datasets.
"""

import pandas as pd
import numpy as np
from scipy import stats


def compare_datasets(df_old: pd.DataFrame, df_new: pd.DataFrame) -> dict:
    """
    Compare two DataFrames and detect statistical drift.

    Args:
        df_old: The reference (baseline) dataset.
        df_new: The new dataset to compare against.

    Returns:
        A dict with drift summaries per shared column.
    """
    shared_cols = list(set(df_old.columns) & set(df_new.columns))
    results = []
    drift_detected = False

    for col in shared_cols:
        col_result = {"column": col, "drift": False}

        if pd.api.types.is_numeric_dtype(df_old[col]) and pd.api.types.is_numeric_dtype(df_new[col]):
            old_data = df_old[col].dropna()
            new_data = df_new[col].dropna()

            old_mean = old_data.mean()
            new_mean = new_data.mean()
            mean_change = round(new_mean - old_mean, 4)
            mean_change_pct = round((mean_change / old_mean * 100) if old_mean != 0 else 0, 2)

            # KS Test for distribution shift
            ks_stat, ks_p_value = stats.ks_2samp(old_data, new_data)
            drifted = ks_p_value < 0.05

            col_result.update({
                "type": "numeric",
                "old_mean": round(old_mean, 4),
                "new_mean": round(new_mean, 4),
                "mean_change": mean_change,
                "mean_change_pct": mean_change_pct,
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_p_value, 4),
                "drift": drifted,
                "summary": (
                    f"`{col}` mean changed: **{old_mean:.2f}** → **{new_mean:.2f}** "
                    f"({'+' if mean_change >= 0 else ''}{mean_change:.2f}, "
                    f"KS p={ks_p_value:.4f})"
                ),
            })

        else:
            # Categorical comparison
            old_counts = df_old[col].value_counts(normalize=True)
            new_counts = df_new[col].value_counts(normalize=True)
            all_cats = set(old_counts.index) | set(new_counts.index)
            shifts = {}
            for cat in all_cats:
                old_pct = old_counts.get(cat, 0) * 100
                new_pct = new_counts.get(cat, 0) * 100
                change = round(new_pct - old_pct, 2)
                if abs(change) > 5:
                    shifts[str(cat)] = {"old_pct": round(old_pct, 2), "new_pct": round(new_pct, 2), "change": change}

            drifted = len(shifts) > 0
            col_result.update({
                "type": "categorical",
                "drift": drifted,
                "category_shifts": shifts,
                "summary": (
                    f"`{col}` category distribution changed in {len(shifts)} categories."
                    if drifted else f"`{col}` category distribution is stable."
                ),
            })

        if col_result["drift"]:
            drift_detected = True
        results.append(col_result)

    # Shape comparison
    shape_changes = {
        "old_rows": len(df_old),
        "new_rows": len(df_new),
        "old_cols": len(df_old.columns),
        "new_cols": len(df_new.columns),
        "new_columns": list(set(df_new.columns) - set(df_old.columns)),
        "removed_columns": list(set(df_old.columns) - set(df_new.columns)),
    }

    return {
        "drift_detected": drift_detected,
        "shared_columns": shared_cols,
        "column_results": results,
        "shape_changes": shape_changes,
    }
