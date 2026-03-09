"""
query_assistant.py
------------------
Rule-based keyword Q&A assistant for answering questions about the dataset.
"""

import pandas as pd
import re
from typing import Optional


class QueryAssistant:
    """
    Intent-based Q&A assistant that answers natural language questions
    about the dataset using keyword matching patterns.
    """

    def __init__(self, df: pd.DataFrame, quality_results: dict, insights: list, cleaning_log: list):
        """
        Initialize the assistant with dataset context.

        Args:
            df: The loaded (original) DataFrame.
            quality_results: Output of quality_detection.compute_health_score().
            insights: List of dicts from insight_engine.extract_insights().
            cleaning_log: List of applied cleaning step descriptions.
        """
        self.df = df
        self.quality = quality_results
        self.insights = insights
        self.cleaning_log = cleaning_log

    def answer(self, question: str) -> str:
        """
        Answer a natural language question about the dataset.

        Args:
            question: A user question string.

        Returns:
            Answer string, or fallback message.
        """
        q = question.lower().strip()

        # --- Missing Values ---
        if any(kw in q for kw in ["missing", "null", "nan", "empty"]):
            if "most" in q or "highest" in q or "max" in q:
                missing = self.df.isnull().sum().sort_values(ascending=False)
                if missing.iloc[0] > 0:
                    col = missing.index[0]
                    cnt = missing.iloc[0]
                    pct = round(cnt / len(self.df) * 100, 2)
                    return f"📊 **`{col}`** has the most missing values: **{cnt:,}** cells ({pct}%)."
                return "✅ No missing values found in any column."
            total = self.quality.get("total_missing", 0)
            pct = self.quality.get("missing_pct", 0)
            return f"📊 The dataset has **{total:,}** missing cells (**{pct}%** of all data)."

        # --- Duplicates ---
        if any(kw in q for kw in ["duplicate", "duplicated", "repeated", "copies"]):
            dup = self.quality.get("duplicate_count", 0)
            dup_pct = self.quality.get("duplicate_pct", 0)
            if "remov" in q:
                removed = sum(1 for s in self.cleaning_log if "duplicate" in s.lower())
                if removed:
                    return f"🗑️ Duplicates were removed. Check the cleaning log for details."
                return "⚠️ Duplicates have not been removed yet. Apply cleaning first."
            return f"📊 The dataset contains **{dup:,}** duplicate rows (**{dup_pct}%** of total rows)."

        # --- Outliers ---
        if "outlier" in q:
            total = self.quality.get("total_outliers", 0)
            pct = self.quality.get("outlier_pct", 0)
            if "which" in q or "column" in q or "most" in q:
                oinfo = self.quality.get("outlier_info", {})
                if oinfo:
                    worst = max(oinfo, key=oinfo.get)
                    return f"📊 **`{worst}`** has the most outliers: **{oinfo[worst]:,}** rows."
                return "✅ No outliers detected."
            return f"📊 **{total:,}** outliers detected (**{pct}%** of rows)."

        # --- Correlation ---
        if any(kw in q for kw in ["correlat", "related", "relationship"]):
            corr_insights = [i for i in self.insights if i["type"] == "correlation"]
            if corr_insights:
                return f"🔗 {corr_insights[0]['title']}\n{corr_insights[0]['description']}"
            return "ℹ️ No strong correlations were found in numeric columns."

        # --- Rows/columns ---
        if any(kw in q for kw in ["row", "record", "sample", "size"]):
            return f"📏 The dataset has **{len(self.df):,} rows**."
        if any(kw in q for kw in ["column", "feature", "field", "variable"]):
            return f"📏 The dataset has **{self.df.shape[1]} columns**: `{'`, `'.join(self.df.columns.tolist())}`."

        # --- Health Score ---
        if any(kw in q for kw in ["health", "quality", "score", "clean"]):
            score = self.quality.get("health_score", 0)
            status = self.quality.get("status", "")
            return f"💊 Dataset Health Score: **{score} / 100**\nStatus: {status}"

        # --- Insights ---
        if any(kw in q for kw in ["insight", "finding", "trend", "pattern"]):
            if self.insights:
                top = self.insights[0]
                return f"💡 Top Insight:\n**{top['title']}**\n{top['description']}"
            return "ℹ️ No insights generated yet. Upload and analyze a dataset first."

        # --- Cleaning log ---
        if any(kw in q for kw in ["clean", "applied", "changed", "modified"]):
            if self.cleaning_log:
                return "🧹 Applied cleaning steps:\n" + "\n".join(f"- {s}" for s in self.cleaning_log)
            return "⚠️ No cleaning has been applied yet."

        # --- Column-specific queries ---
        for col in self.df.columns:
            if col.lower() in q:
                dtype = str(self.df[col].dtype)
                missing = int(self.df[col].isnull().sum())
                unique = int(self.df[col].nunique())
                response = (
                    f"📋 **Column: `{col}`**\n"
                    f"- Type: `{dtype}`\n"
                    f"- Missing: {missing}\n"
                    f"- Unique values: {unique}\n"
                )
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    response += (
                        f"- Min: {self.df[col].min():.4f}\n"
                        f"- Max: {self.df[col].max():.4f}\n"
                        f"- Mean: {self.df[col].mean():.4f}"
                    )
                else:
                    top = self.df[col].value_counts().head(3).index.tolist()
                    response += f"- Top values: {', '.join(str(v) for v in top)}"
                return response

        # --- Fallback ---
        return (
            "🤔 I couldn't understand that question. Try asking about:\n"
            "- Missing values\n"
            "- Duplicate rows\n"
            "- Outliers\n"
            "- Correlations\n"
            "- Dataset size\n"
            "- Health score\n"
            "- A specific column name"
        )
