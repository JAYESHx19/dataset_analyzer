"""
report_generator.py
-------------------
Generates natural language reports and auto-documentation for the dataset.
"""

import pandas as pd
from datetime import datetime


def generate_nl_report(
    df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    profile: dict,
    health_before: dict,
    health_after: dict,
    insights: list,
    cleaning_log: list,
) -> str:
    """
    Generate a full natural language analysis report as a text string.

    Args:
        df: Original DataFrame.
        cleaned_df: Cleaned DataFrame.
        profile: Output of profiling.get_profile().
        health_before: Health score dict before cleaning.
        health_after: Health score dict after cleaning.
        insights: List of insight dicts.
        cleaning_log: List of applied cleaning step descriptions.

    Returns:
        A formatted string report.
    """
    now = datetime.now().strftime("%B %d, %Y %H:%M")
    lines = []

    lines.append("=" * 60)
    lines.append("     INTELLIGENT DATASET ANALYSIS REPORT")
    lines.append(f"     Generated: {now}")
    lines.append("=" * 60)

    # --- Dataset Overview ---
    lines.append("\n📊 DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total Rows    : {profile['total_rows']:,}")
    lines.append(f"Total Columns : {profile['total_columns']}")
    lines.append(f"Numeric Cols  : {len(profile['numeric_columns'])}")
    lines.append(f"Categorical   : {len(profile['categorical_columns'])}")
    lines.append(f"Total Missing : {profile['total_missing']:,} cells ({profile['total_missing_pct']}%)")
    lines.append(f"Duplicates    : {profile['duplicates']}")
    lines.append(f"Memory Usage  : {profile['memory_usage_kb']} KB")

    # --- Data Quality Issues ---
    lines.append("\n🔍 DATA QUALITY ISSUES (Before Cleaning)")
    lines.append("-" * 40)
    lines.append(f"Dataset Health Score : {health_before['health_score']} / 100")
    lines.append(f"Status               : {health_before['status']}")
    lines.append(f"Missing Values       : {health_before['missing_pct']}%")
    lines.append(f"Duplicate Rows       : {health_before['duplicate_count']} ({health_before['duplicate_pct']}%)")
    lines.append(f"Outliers Detected    : {health_before['total_outliers']} ({health_before['outlier_pct']}%)")
    lines.append(f"Type Errors          : {health_before['type_error_pct']}% of columns")

    # --- Cleaning Steps ---
    if cleaning_log:
        lines.append("\n🧹 CLEANING METHODS APPLIED")
        lines.append("-" * 40)
        for step in cleaning_log:
            lines.append(f"  {step}")
        lines.append(f"\nAfter Cleaning:")
        lines.append(f"  Rows Remaining    : {len(cleaned_df):,}")
        lines.append(f"  Health Score Now  : {health_after['health_score']} / 100")
    else:
        lines.append("\n🧹 CLEANING")
        lines.append("-" * 40)
        lines.append("  No cleaning has been applied yet.")

    # --- Key Insights ---
    lines.append("\n💡 KEY INSIGHTS")
    lines.append("-" * 40)
    if insights:
        for i, insight in enumerate(insights[:5], 1):
            lines.append(f"  {i}. {insight['title']}")
            lines.append(f"     {insight['description'].replace('**', '').replace('`', '')}")
    else:
        lines.append("  No significant insights detected.")

    # --- Final Summary ---
    lines.append("\n📋 FINAL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"The dataset contains {profile['total_rows']:,} records across {profile['total_columns']} columns.")
    if cleaning_log:
        lines.append(f"After cleaning, {len(cleaned_df):,} records remain.")
        improvement = health_after['health_score'] - health_before['health_score']
        lines.append(f"Dataset health improved by {improvement:+.1f} points to {health_after['health_score']}/100.")

    lines.append("\n" + "=" * 60)
    lines.append("     END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_documentation(df: pd.DataFrame, filename: str = "dataset") -> str:
    """
    Generate automatic dataset documentation similar to a data catalog entry.

    Args:
        df: The DataFrame to document.
        filename: Name of the original dataset file.

    Returns:
        A formatted documentation string.
    """
    now = datetime.now().strftime("%B %d, %Y %H:%M")
    lines = []

    lines.append("=" * 60)
    lines.append("     DATASET AUTO-DOCUMENTATION")
    lines.append(f"     Dataset: {filename}")
    lines.append(f"     Created: {now}")
    lines.append("=" * 60)

    lines.append("\n📁 DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Filename   : {filename}")
    lines.append(f"Rows       : {df.shape[0]:,}")
    lines.append(f"Columns    : {df.shape[1]}")

    lines.append("\n📋 COLUMN DESCRIPTIONS")
    lines.append("-" * 40)

    for col in df.columns:
        lines.append(f"\n  Column  : {col}")
        lines.append(f"  Type    : {df[col].dtype}")
        missing = df[col].isnull().sum()
        missing_pct = round(missing / len(df) * 100, 2)
        lines.append(f"  Missing : {missing} ({missing_pct}%)")
        lines.append(f"  Unique  : {df[col].nunique()}")

        if pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"  Min     : {df[col].min()}")
            lines.append(f"  Max     : {df[col].max()}")
            lines.append(f"  Mean    : {round(df[col].mean(), 4)}")
            lines.append(f"  Std Dev : {round(df[col].std(), 4)}")
            skew = df[col].skew()
            lines.append(f"  Skewness: {round(skew, 4)}")
            if abs(skew) > 1:
                lines.append(f"  Note    : Column is skewed — consider transformation.")
            if missing_pct > 10:
                lines.append(f"  Note    : High missing rate — imputation recommended.")
        else:
            top_vals = df[col].value_counts().head(5).index.tolist()
            lines.append(f"  Top Values: {', '.join(str(v) for v in top_vals)}")
            if missing_pct > 10:
                lines.append(f"  Note    : High missing rate — mode imputation or dropping recommended.")

    lines.append("\n\n" + "=" * 60)
    lines.append("     END OF DOCUMENTATION")
    lines.append("=" * 60)

    return "\n".join(lines)
