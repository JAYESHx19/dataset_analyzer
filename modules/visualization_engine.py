"""
visualization_engine.py
-----------------------
Generates Plotly visualizations for the dataset analysis dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOTLY_TEMPLATE = "plotly_white"


def _rgb_to_rgba(color: str, alpha: float = 0.3) -> str:
    """
    Convert a Plotly rgb(...) color string to an rgba(...) string with alpha.
    Falls back to the original color string if parsing fails.
    """
    import re
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color.strip())
    if m:
        r, g, b = m.group(1), m.group(2), m.group(3)
        return f"rgba({r},{g},{b},{alpha})"
    # Already rgba or a hex — return as-is
    return color
PRIMARY_COLOR = "#4F46E5"
SECONDARY_COLOR = "#22C55E"


def missing_value_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values found!", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="green"))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
        return fig

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Column", "y": "Missing Count"},
        color=missing.values,
        color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis=dict(title=dict(text="Column", font=dict(color="#000000"))),
        yaxis=dict(title=dict(text="Missing Count", font=dict(color="#000000"))),
        title=dict(text="Missing Values per Column", font=dict(color="#000000")),
        height=400,
        coloraxis_showscale=False,
    )
    return fig


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap showing null positions across a sample of rows."""
    sample = df.head(200)
    z = sample.isnull().astype(int).values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=df.columns.tolist(),
        # 0 = present (light), 1 = missing (very dark red/crimson)
        colorscale=[[0, "#F0F9FF"], [1, "#7F1D1D"]],
        showscale=True,
        colorbar=dict(
            tickvals=[0, 1],
            ticktext=["Present", "Missing"],
            thickness=14,
            len=0.6,
        ),
    ))
    fig.update_layout(
        title="Missing Value Heatmap (first 200 rows)",
        xaxis_title="Columns",
        yaxis_title="Rows",
        template=PLOTLY_TEMPLATE,
        height=400,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
        return fig

    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text="Correlation Heatmap", font=dict(color="#000000")),
        template=PLOTLY_TEMPLATE,
        height=480,
    )
    return fig


def histogram_grid(df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    """Grid of histograms for numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    if not numeric_cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns found.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        return fig

    n = len(numeric_cols)
    cols_per_row = min(3, n)
    rows = (n + cols_per_row - 1) // cols_per_row
    fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=numeric_cols)

    colors = px.colors.qualitative.Vivid
    for i, col in enumerate(numeric_cols):
        row = i // cols_per_row + 1
        col_idx = i % cols_per_row + 1
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, marker_color=colors[i % len(colors)], showlegend=False),
            row=row, col=col_idx
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE, 
        title=dict(text="Numeric Column Distributions", font=dict(color="#000000")),
        height=250 * rows,
    )
    return fig


def boxplot_outliers(df: pd.DataFrame, max_cols: int = 8) -> go.Figure:
    """
    Boxplots for outlier visualization.
    Each numeric column gets its own subplot so differing scales
    don't compress each other's outlier visibility.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    if not numeric_cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns found.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    n = len(numeric_cols)
    cols_per_row = min(3, n)
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows,
        cols=cols_per_row,
        subplot_titles=numeric_cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    colors = px.colors.qualitative.Vivid
    for i, col in enumerate(numeric_cols):
        row = i // cols_per_row + 1
        col_idx = i % cols_per_row + 1
        col_data = df[col].dropna()

        # Compute IQR to highlight outliers in a different color
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower) | (col_data > upper)]

        fig.add_trace(
            go.Box(
                y=col_data,
                name=col,
                boxpoints="outliers",
                marker=dict(
                    color=colors[i % len(colors)],
                    outliercolor="#EF4444",
                    size=5,
                    line=dict(color="#EF4444", width=1),
                ),
                line=dict(color=colors[i % len(colors)]),
                fillcolor=_rgb_to_rgba(colors[i % len(colors)], alpha=0.25),  # semi-transparent fill
                showlegend=False,
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "Value: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col_idx,
        )

        # Annotate outlier count
        if len(outliers) > 0:
            fig.add_annotation(
                text=f"⚠ {len(outliers)} outliers",
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                x=0.5, y=1.12,
                xanchor="center",
                showarrow=False,
                font=dict(size=10, color="#EF4444"),
                row=row, col=col_idx,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Outlier Boxplots (red dots = outliers)", font=dict(size=15, color="#000000")),
        height=280 * rows,
        paper_bgcolor="#FAFAFA",
        xaxis=dict(title=dict(text="Columns", font=dict(color="#000000"))),
        yaxis=dict(title=dict(text="Values", font=dict(color="#000000"))),
    )
    return fig


def category_distribution(df: pd.DataFrame, max_cols: int = 3) -> list:
    """List of Plotly pie/bar charts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:max_cols]
    figures = []
    for col in cat_cols:
        counts = df[col].value_counts().head(15)
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={"x": col, "y": "Count"},
            title=f"Distribution of `{col}`",
            color=counts.values,
            color_continuous_scale=["#4F46E5", "#22C55E"],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, height=350)
        figures.append((col, fig))
    return figures


def before_after_comparison(before_missing: int, after_missing: int,
                             before_outliers: int, after_outliers: int) -> go.Figure:
    """Grouped bar chart comparing before vs after cleaning metrics."""
    metrics = ["Missing Values", "Outliers"]
    before = [before_missing, before_outliers]
    after = [after_missing, after_outliers]

    fig = go.Figure(data=[
        go.Bar(name="Before Cleaning", x=metrics, y=before, marker_color="#EF4444"),
        go.Bar(name="After Cleaning", x=metrics, y=after, marker_color="#22C55E"),
    ])
    fig.update_layout(
        barmode="group",
        title="Before vs After Cleaning",
        template=PLOTLY_TEMPLATE,
        height=380,
        yaxis_title="Count",
    )
    return fig


def drift_comparison_chart(df_old: pd.DataFrame, df_new: pd.DataFrame, col: str) -> go.Figure:
    """Overlay histogram for drift comparison between two datasets."""
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(df_old[col]):
        fig.add_trace(go.Histogram(x=df_old[col].dropna(), name="Dataset A (Old)",
                                   opacity=0.6, marker_color="#4F46E5"))
        fig.add_trace(go.Histogram(x=df_new[col].dropna(), name="Dataset B (New)",
                                   opacity=0.6, marker_color="#F59E0B"))
        fig.update_layout(barmode="overlay", title=f"Distribution Drift: {col}",
                          template=PLOTLY_TEMPLATE, height=380)
    else:
        counts_old = df_old[col].value_counts().rename("Dataset A")
        counts_new = df_new[col].value_counts().rename("Dataset B")
        combined = pd.concat([counts_old, counts_new], axis=1).fillna(0)
        fig = go.Figure(data=[
            go.Bar(name="Dataset A (Old)", x=combined.index, y=combined["Dataset A"], marker_color="#4F46E5"),
            go.Bar(name="Dataset B (New)", x=combined.index, y=combined["Dataset B"], marker_color="#F59E0B"),
        ])
        fig.update_layout(barmode="group", title=f"Category Drift: {col}",
                          template=PLOTLY_TEMPLATE, height=380)
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for feature importance scores."""
    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Correlation-based)",
        color="Importance",
        color_continuous_scale=["#C7D2FE", "#4F46E5"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, height=400)
    return fig
