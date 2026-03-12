"""
app.py
------
Main Streamlit application for the Intelligent Dataset Analyzer & Auto-Cleaning System.
Multi-page layout with sidebar navigation.
"""

import streamlit as st
import pandas as pd
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.data_loader import load_dataset, get_dataset_metadata
from modules.profiling import get_profile
from modules.quality_detection import (
    detect_missing, detect_duplicates, detect_outliers_iqr,
    detect_type_errors, compute_health_score,
)
from modules.cleaning_engine import apply_cleaning, get_suggested_strategy
from modules.visualization_engine import (
    missing_value_chart, missing_heatmap, correlation_heatmap,
    histogram_grid, boxplot_outliers, category_distribution,
    before_after_comparison, drift_comparison_chart, feature_importance_chart,
)
from modules.insight_engine import extract_insights, compute_feature_importance
from modules.advanced_analysis import compare_datasets
from modules.report_generator import generate_nl_report, generate_documentation
from modules.query_assistant import QueryAssistant
from utils.helpers import (
    init_session_state, has_dataset, has_cleaned_dataset,
    no_data_warning, dataframe_to_csv_bytes, text_to_bytes, pdf_to_bytes,
    text_to_pdf_bytes, format_number, render_health_bar, card_css,
    save_csv_to_disk, save_text_to_disk, save_pdf_to_disk,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSense — Intelligent Dataset Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(card_css(), unsafe_allow_html=True)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #F8FAFC; }
    section[data-testid="stSidebar"] { background: #1E293B !important; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 14px; border-radius: 8px; cursor: pointer;
        font-size: 14px; font-weight: 600; transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover { background: #334155 !important; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; font-weight: 700; }
    h1 { color: #000000 !important; }
    h2 { color: #000000 !important; }
    h3 { color: #000000 !important; }
    h4 { color: #000000 !important; }
    h5, h6 { color: #000000 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 12px; padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #E2E8F0;
    }
    .stAlert { border-radius: 10px; }
    .stButton > button {
        border-radius: 8px; font-weight: 600; font-size: 14px;
        padding: 8px 20px; transition: all 0.2s; background: #4F46E5; color: white;
    }
    .stButton > button:hover { background: #4338CA; }
    .stDownloadButton > button {
        background: #4F46E5 !important; color: white !important;
        border-radius: 8px; font-weight: 600; border: none;
    }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .section-title {
        font-size: 16px; font-weight: 700; color: #000000 !important; margin-bottom: 12px;
    }
    /* Force all text to black with maximum specificity */
    .stMarkdown, .stMarkdown * { color: #000000 !important; font-weight: 600 !important; }
    .stSelectbox, .stSelectbox * { color: #000000 !important; font-weight: 600 !important; }
    .stSlider, .stSlider * { color: #000000 !important; font-weight: 600 !important; }
    .stTextInput, .stTextInput * { color: #000000 !important; font-weight: 600 !important; }
    .stTextArea, .stTextArea * { color: #000000 !important; font-weight: 600 !important; }
    .stCheckbox, .stCheckbox * { color: #000000 !important; font-weight: 600 !important; }
    .stRadio, .stRadio * { color: #000000 !important; font-weight: 600 !important; }
    .stSelectbox > label, .stSlider > label, .stTextInput > label, .stTextArea > label { 
        color: #000000 !important; font-weight: 700 !important; 
    }
    .stExpander, .stExpander * { color: #000000 !important; font-weight: 600 !important; }
    .stCaption { color: #000000 !important; font-weight: 600 !important; }
    .stInfo { background: #EFF6FF; border: 1px solid #BFDBFE; }
    .stWarning { background: #FFFBEB; border: 1px solid #FDE68A; }
    .stError { background: #FEF2F2; border: 1px solid #FECACA; }
    .stSuccess { background: #F0FDF4; border: 1px solid #BBF7D0; }
    div[data-testid="stExpander"] {
        background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important;
    }
    div[data-testid="stExpander"] * {
        color: #000000 !important; font-weight: 600 !important;
    }
    /* White expander header */
    div[data-testid="stExpander"] > div > div:first-child {
        background: #FFFFFF !important; border-bottom: 1px solid #E2E8F0 !important;
    }
    /* Expander arrow and title */
    div[data-testid="stExpander"] > div > div:first-child > div {
        color: #000000 !important; font-weight: 700 !important;
    }
    /* Force all expander variations to white */
    .streamlit-expanderHeader, .streamlit-expanderHeader * {
        background: #FFFFFF !important;
        color: #000000 !important;
    }
    div[style*="background: rgb"], div[style*="background-color"] {
        background: #FFFFFF !important;
    }
    /* Override any hover states */
    div[data-testid="stExpander"]:hover,
    div[data-testid="stExpander"]:hover *,
    div[data-testid="stExpander"]:hover > div,
    div[data-testid="stExpander"]:hover > div > div,
    div[data-testid="stExpander"]:hover > div > div > div {
        background: #FFFFFF !important;
        color: #000000 !important;
        box-shadow: none !important;
    }
    .insight-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 8px;
        padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .insight-card h4 {
        color: #000000 !important; margin: 0 0 8px 0; font-size: 14px; font-weight: 700;
    }
    .insight-card p {
        color: #000000 !important; margin: 0; font-size: 13px; line-height: 1.5; font-weight: 600;
    }
    .chat-bubble-user {
        background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px;
        padding: 12px 16px; margin: 8px 0; max-width: 80%; margin-left: auto;
        color: #000000 !important; font-weight: 600;
    }
    .chat-bubble-bot {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 12px 16px; margin: 8px 0; max-width: 80%; color: #000000 !important; font-weight: 600;
    }
    /* Force black text for all metrics and data displays with maximum specificity */
    div[data-testid="metric-container"] * { color: #000000 !important; font-weight: 600 !important; }
    div[data-testid="element-container"] * { color: #000000 !important; font-weight: 600 !important; }
    .stMetric * { color: #000000 !important; font-weight: 600 !important; }
    /* Tabs styling */
    .stTabs * { color: #000000 !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] * { color: #000000 !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab"] * { color: #000000 !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #000000 !important; font-weight: 700 !important; }
    /* Override all possible text elements */
    p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th { color: #000000 !important; }
    /* Streamlit specific overrides */
    .css-1d391kg { color: #000000 !important; }
    .css-1v0mbdj { color: #000000 !important; }
    .css-1lcbmhc { color: #000000 !important; }
    .css-1r6slb0 { color: #000000 !important; }
    .css-1vq4p4l { color: #000000 !important; }
    .css-1kyxreq { color: #000000 !important; }
    /* File uploader styling - maximum specificity and override */
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] > div,
    div[data-testid="stFileUploader"] > div > div,
    div[data-testid="stFileUploader"] > div > div > div,
    div[data-testid="stFileUploader"] > div > div > div > div {
        background: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }
    div[data-testid="stFileUploader"] *,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] div,
    div[data-testid="stFileUploader"] section {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Override any inline styles */
    div[data-testid="stFileUploader"] [style*="background"],
    div[data-testid="stFileUploader"] [style*="color"] {
        background: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Target Streamlit specific uploader classes */
    .css-1cpxqw2,
    .css-1cpxqw2 *,
    .css-1cpxqw2 > div,
    .css-1cpxqw2 > div > div {
        background: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Force override for all uploader elements */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stFileUploader"] * {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Text area styling for reports */
    div[data-testid="stTextArea"] > div > div > textarea,
    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #D1D5DB !important;
    }
    div[data-testid="stTextArea"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stTextArea"] * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Override any dark text area styles */
    .stTextArea,
    .stTextArea *,
    textarea,
    textarea * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    /* Cleaning results section styling */
    .stSuccess,
    .stSuccess * {
        background-color: #22C55E !important;
        color: #FFFFFF !important;
    }
    .stSuccess p,
    .stSuccess span,
    .stSuccess div,
    .stSuccess li,
    .stSuccess ul,
    .stSuccess strong,
    .stSuccess b {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }
    /* Success message override */
    [data-testid="stSuccess"] {
        background-color: #22C55E !important;
    }
    [data-testid="stSuccess"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSuccess"] div[data-testid="stMarkdownContainer"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSuccess"] ul,
    [data-testid="stSuccess"] li {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="font-size: 22px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.3px;">
            🔬 DataSense
        </div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">
            Intelligent Dataset Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    PAGES = {
        "🏠  Welcome": "welcome",
        "📂  Upload Dataset": "upload",
        "📊  Data Profiling": "profiling",
        "🔍  Quality Report": "quality",
        "🧹  Cleaning Strategies": "cleaning",
        "📈  Visualization Dashboard": "visualization",
        "💡  Insights & Feature Importance": "insights",
        "🔄  Data Drift Analyzer": "drift",
        "📝  Natural Language Report": "report",
        "🤖  AI Data Assistant": "assistant",
        "⬇️  Download Results": "download",
    }

    page_label = st.radio("Navigation", list(PAGES.keys()), label_visibility="hidden")
    page = PAGES[page_label]

    if has_dataset():
        st.markdown("---")
        df = st.session_state.df
        st.markdown(f"""
        <div style="font-size: 12px; color: #94A3B8; padding: 4px 0;">
            <b style="color: #CBD5E1;">📁 {st.session_state.filename}</b><br>
            {format_number(df.shape[0])} rows &nbsp;·&nbsp; {df.shape[1]} cols
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.quality_before:
            score = st.session_state.quality_before["health_score"]
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <div style="font-size: 11px; color: #000000; margin-bottom: 4px; font-weight: 600;">Dataset Health</div>
                <div style="background: #E5E7EB; border-radius: 8px; height: 10px; border: 1px solid #D1D5DB;">
                    <div style="width: {score}%; background: {'#22C55E' if score >= 75 else '#F59E0B' if score >= 50 else '#EF4444'};
                                height: 10px; border-radius: 8px; border: 1px solid {'#22C55E' if score >= 75 else '#F59E0B' if score >= 50 else '#EF4444'};"></div>
                </div>
                <div style="font-size: 13px; color: {'#22C55E' if score >= 75 else '#F59E0B' if score >= 50 else '#EF4444'};
                            font-weight: 700; margin-top: 4px;">{score}/100</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "welcome":
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <div style="font-size: 52px; margin-bottom: 8px;">🔬</div>
        <h1 style="font-size: 38px; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -1px;">
            Intelligent Dataset Analyzer
        </h1>
        <p style="font-size: 18px; color: #475569; margin-top: 12px; max-width: 600px; margin-left: auto; margin-right: auto; font-weight: 500;">
            Automatically analyze, clean, visualize, and understand your datasets in minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("📊", "Data Profiling", "Automatic structure analysis with column stats"),
        ("🧹", "Auto Cleaning", "Intelligent preprocessing and imputation"),
        ("💡", "Smart Insights", "Correlation and pattern detection"),
        ("🤖", "AI Assistant", "Ask questions in natural language"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 15px; font-weight: 700; color: #0F172A;">{title}</div>
                <div style="font-size: 12px; color: #475569; margin-top: 6px; font-weight: 500;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="metric-card">
            <div class="section-title">✨ What This System Does</div>
            <ul style="color: #374151; font-size: 14px; line-height: 2; font-weight: 500;">
                <li>Profiles your dataset structure automatically</li>
                <li>Detects missing values, duplicates & outliers</li>
                <li>Computes a <b style="color: #1E293B;">Dataset Health Score</b> (0–100)</li>
                <li>Applies intelligent cleaning strategies</li>
                <li>Generates interactive Plotly visualizations</li>
                <li>Extracts key insights & feature importance</li>
                <li>Detects data drift between two datasets</li>
                <li>Produces natural language analysis reports</li>
                <li>Auto-documents your dataset columns</li>
                <li>Answers questions about your data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-card">
            <div class="section-title">🚀 Getting Started</div>
            <ol style="color: #374151; font-size: 14px; line-height: 2.2; font-weight: 500;">
                <li>Go to <b style="color: #1E293B;">Upload Dataset</b> and upload a CSV or Excel file</li>
                <li>View the <b style="color: #1E293B;">Data Profiling</b> dashboard</li>
                <li>Review the <b style="color: #1E293B;">Quality Report</b> and health score</li>
                <li>Configure and run <b>Cleaning Strategies</b></li>
                <li>Explore the <b>Visualization Dashboard</b></li>
                <li>Read auto-generated <b>Insights</b></li>
                <li>Ask questions via the <b>AI Assistant</b></li>
                <li>Download cleaned data and reports</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Use the sidebar to navigate between pages. Start with **Upload Dataset**.", icon="💡")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "upload":
    st.markdown('<div class="section-title">📂 Upload Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop your dataset here, or click to browse",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Loading dataset..."):
            df, error = load_dataset(uploaded)

        if error:
            st.error(f"❌ {error}")
        else:
            st.session_state.df = df
            st.session_state.filename = uploaded.name
            # Reset derived state on new upload
            for key in ["cleaned_df", "quality_before", "quality_after", "profile",
                        "insights", "cleaning_log", "report_text", "doc_text", "chat_history"]:
                st.session_state[key] = [] if key in ("cleaning_log", "chat_history") else None

            # Compute profile and health score immediately
            st.session_state.profile = get_profile(df)
            st.session_state.quality_before = compute_health_score(df)

            meta = get_dataset_metadata(df, uploaded)
            st.success(f"✅ Dataset loaded successfully: **{meta['filename']}**")

            # Overview cards
            c1, c2, c3, c4, c5 = st.columns(5)
            q = st.session_state.quality_before
            c1.metric("📏 Total Rows", format_number(meta["rows"]))
            c2.metric("📋 Total Columns", meta["columns"])
            c3.metric("⚠️ Missing Values", f"{q['missing_pct']}%")
            c4.metric("🔁 Duplicates", format_number(q["duplicate_count"]))
            c5.metric("💾 File Size", f"{meta['file_size_kb']} KB")

            st.markdown("---")
            render_health_bar(q["health_score"])
            st.caption(q["status"])

            st.markdown("---")
            st.markdown("#### 🗂️ Dataset Preview")
            st.dataframe(df.head(50), use_container_width=True)

            st.markdown("#### 📋 Column Types")
            type_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
            })
            st.dataframe(type_df, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="border: 2px dashed #CBD5E1; border-radius: 14px; padding: 60px;
                    text-align: center; background: #FFFFFF; margin-top: 10px;">
            <div style="font-size: 48px; margin-bottom: 14px;">📂</div>
            <div style="font-size: 18px; font-weight: 600; color: #000000;">
                Drag and drop your dataset here
            </div>
            <div style="font-size: 13px; color: #000000; margin-top: 8px; font-weight: 500;">
                Limit 200MB per file • CSV, XLSX, XLS
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Need a sample dataset? Run **`python generate_sample.py`** from the project folder to create one, then upload `assets/sample_dataset.csv`.", icon="🧪")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA PROFILING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "profiling":
    if not has_dataset():
        no_data_warning("Data Profiling")

    df = st.session_state.df
    profile = st.session_state.profile or get_profile(df)
    st.session_state.profile = profile

    st.markdown('<div class="section-title">📊 Data Profiling</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", format_number(profile["total_rows"]))
    c2.metric("Total Columns", profile["total_columns"])
    c3.metric("Numeric Columns", len(profile["numeric_columns"]))
    c4.metric("Categorical Columns", len(profile["categorical_columns"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Missing", format_number(profile["total_missing"]))
    c6.metric("Missing %", f"{profile['total_missing_pct']}%")
    c7.metric("Duplicate Rows", format_number(profile["duplicates"]))
    c8.metric("Memory Usage", f"{profile['memory_usage_kb']} KB")

    st.markdown("---")
    st.markdown("#### 📋 Per-Column Statistics")
    col_stats_df = pd.DataFrame(profile["column_stats"])
    st.dataframe(col_stats_df, use_container_width=True, hide_index=True)

    with st.expander("📈 Full Descriptive Statistics (df.describe)"):
        st.dataframe(profile["describe"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "quality":
    if not has_dataset():
        no_data_warning("Quality Report")

    df = st.session_state.df
    q = st.session_state.quality_before or compute_health_score(df)
    st.session_state.quality_before = q

    st.markdown('<div class="section-title">🔍 Data Quality Report</div>', unsafe_allow_html=True)

    # Health score card
    col_h, col_r = st.columns([2, 1])
    with col_h:
        st.markdown("#### 💊 Dataset Health Score")
        render_health_bar(q["health_score"])
        st.caption(q["status"])

        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Missing Score", f"{q['missing_score']}/40")
        sc2.metric("Duplicate Score", f"{q['duplicate_score']}/20")
        sc3.metric("Outlier Score", f"{q['outlier_score']}/25")
        sc4.metric("Type Score", f"{q['datatype_score']}/15")

    with col_r:
        # Score interpretation table
        st.markdown("""
        <div class="metric-card">
            <div style="font-weight: 600; font-size: 13px; color: #374151; margin-bottom: 10px;">Score Legend</div>
            <div style="font-size: 12px; color: #4B5563; line-height: 2.2;">
                🟢 90–100 &nbsp; Excellent<br>
                🟡 75–89  &nbsp; Good<br>
                🟠 50–74  &nbsp; Moderate<br>
                🔴 30–49  &nbsp; Poor<br>
                ⛔ 0–29   &nbsp; Very Poor
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Duplicates", "Outliers", "Type Errors"])

    with tab1:
        missing_df = detect_missing(df)
        if missing_df.empty:
            st.success("✅ No missing values detected!")
        else:
            st.error(f"⚠️ {len(missing_df)} columns contain missing values.")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            with col2:
                st.plotly_chart(missing_value_chart(df), use_container_width=True)
        st.plotly_chart(missing_heatmap(df), use_container_width=True)

    with tab2:
        dup_count = q["duplicate_count"]
        if dup_count == 0:
            st.success("✅ No duplicate rows detected.")
        else:
            st.warning(f"🔁 **{dup_count:,}** duplicate rows found ({q['duplicate_pct']}% of dataset).")
            st.dataframe(df[df.duplicated(keep=False)].head(20),
                         use_container_width=True)
            st.caption("Sample of duplicate rows (up to 20 shown)")

    with tab3:
        outlier_info = q["outlier_info"]
        if not outlier_info:
            st.success("✅ No outliers detected.")
        else:
            st.warning(f"📦 **{q['total_outliers']:,}** outliers detected across {len(outlier_info)} columns.")
            out_df = pd.DataFrame({"Column": list(outlier_info.keys()),
                                   "Outlier Count": list(outlier_info.values())})
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(out_df, use_container_width=True, hide_index=True)
            with c2:
                st.plotly_chart(boxplot_outliers(df), use_container_width=True)

    with tab4:
        type_errors = q["type_errors"]
        if not type_errors:
            st.success("✅ No data type errors detected.")
        else:
            st.warning(f"🔧 **{len(type_errors)}** columns appear to have incorrect data types:")
            for col in type_errors:
                st.markdown(f"- **{col}** — stored as **text** but contains numeric values")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLEANING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "cleaning":
    if not has_dataset():
        no_data_warning("Cleaning Strategies")

    df = st.session_state.df
    st.markdown('<div class="section-title">🧹 Cleaning Strategies</div>', unsafe_allow_html=True)

    # Missing value strategies
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    missing_strategy = {}

    st.markdown("### 1️⃣ Missing Value Handling")
    if not missing_cols:
        st.success("✅ No missing values — nothing to configure here.")
    else:
        st.markdown(f"Configure imputation for **{len(missing_cols)}** columns with missing data.")
        for col in missing_cols:
            suggested = get_suggested_strategy(df, col)
            dtype_label = "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical"
            options = (["mean", "median", "mode", "drop"]
                       if pd.api.types.is_numeric_dtype(df[col])
                       else ["mode", "drop"])
            
            st.markdown(f"**{col}** ({dtype_label}, {int(df[col].isnull().sum())} missing)")
            chosen = st.radio(
                "Select strategy:",
                options,
                index=options.index(suggested) if suggested in options else 0,
                format_func=lambda x: {
                    "mean": "Mean (average)",
                    "median": "Median (middle value)",
                    "mode": "Mode (most frequent)",
                    "drop": "Drop rows with missing values"
                }.get(x, x),
                horizontal=True,
                key=f"missing_{col}",
                label_visibility="collapsed"
            )
            missing_strategy[col] = chosen
            st.markdown("---")

    st.markdown("---")
    
    # Duplicate handling
    st.markdown("### 2️⃣ Duplicate Rows")
    remove_dups = st.checkbox("Remove duplicate rows", value=True, key="remove_dups_cb")
    dup_count = st.session_state.quality_before["duplicate_count"] if st.session_state.quality_before else detect_duplicates(df)
    st.caption(f"Currently: **{dup_count:,}** duplicate rows detected.")

    st.markdown("---")
    
    # Outlier handling
    st.markdown("### 3️⃣ Outlier Handling")
    outlier_opt = st.radio(
        "Select outlier strategy:",
        ["keep", "cap", "remove"],
        format_func=lambda x: {
            "keep": "Keep outliers (no change)",
            "cap": "Cap using IQR bounds",
            "remove": "Remove outlier rows",
        }[x],
        horizontal=True,
        key="outlier_radio",
    )

    st.markdown("---")
    
    # Data type correction
    st.markdown("### 4️⃣ Data Type Correction")
    fix_dtypes = st.checkbox("Auto-convert numeric-looking text columns", value=True, key="fix_dtypes_cb")
    if st.session_state.quality_before:
        errs = st.session_state.quality_before.get("type_errors", [])
        if errs:
            st.caption(f"Columns to fix: **{', '.join(errs)}**")

    st.markdown("---")
    st.markdown("---")
    if st.button("🚀 Run Data Cleaning", type="primary", use_container_width=False):
        with st.spinner("Cleaning dataset..."):
            cleaned_df, log = apply_cleaning(
                df,
                missing_strategy=missing_strategy,
                remove_duplicates=remove_dups,
                outlier_strategy=outlier_opt,
                fix_types=fix_dtypes,
            )
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_log = log
            st.session_state.quality_after = compute_health_score(cleaned_df)

        st.success(f"✅ Cleaning complete! {len(cleaned_df):,} rows remaining." + 
                   ("\n\n**Applied cleaning steps:**\n" + "\n".join([f"- {step}" for step in log]) if log else ""))

        # Before vs After health
        q_before = st.session_state.quality_before
        q_after = st.session_state.quality_after
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Cleaning:**")
            render_health_bar(q_before["health_score"])
        with col2:
            st.markdown("**After Cleaning:**")
            render_health_bar(q_after["health_score"])

        fig = before_after_comparison(
            q_before["total_missing"], q_after["total_missing"],
            q_before["total_outliers"], q_after["total_outliers"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Cleaned Dataset (CSV)",
            data=dataframe_to_csv_bytes(cleaned_df),
            file_name="clean_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "visualization":
    if not has_dataset():
        no_data_warning("Visualization Dashboard")

    df = st.session_state.cleaned_df if has_cleaned_dataset() else st.session_state.df
    label = "cleaned" if has_cleaned_dataset() else "original"
    st.markdown(f'<div class="section-title">📈 Visualization Dashboard <span style="font-size:13px;color:#000000;font-weight:500;">({label} dataset)</span></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Missing Values", "🔗 Correlations", "📊 Distributions",
        "📦 Outliers", "🗂️ Categories",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(missing_value_chart(df), use_container_width=True)
        with c2:
            st.plotly_chart(missing_heatmap(df), use_container_width=True)

    with tab2:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    with tab3:
        st.plotly_chart(histogram_grid(df), use_container_width=True)

    with tab4:
        st.plotly_chart(boxplot_outliers(df), use_container_width=True)

    with tab5:
        figs = category_distribution(df)
        if not figs:
            st.info("No categorical columns found.")
        else:
            for col_name, fig in figs:
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "insights":
    if not has_dataset():
        no_data_warning("Insights")

    df = st.session_state.cleaned_df if has_cleaned_dataset() else st.session_state.df

    st.markdown('<div class="section-title">💡 Insights & Feature Importance</div>', unsafe_allow_html=True)

    if st.session_state.insights is None:
        with st.spinner("Generating insights..."):
            st.session_state.insights = extract_insights(df)

    insights = st.session_state.insights

    if insights:
        st.markdown("#### 🏆 Top Dataset Insights")
        for ins in insights:
            icon = {"correlation": "🔗", "category": "🏷️", "warning": "⚠️", "distribution": "📊"}.get(ins["type"], "💡")
            st.markdown(f"""
            <div class="insight-card">
                <h4>{icon} {ins['title']}</h4>
                <p>{ins['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant insights could be automatically detected.")

    st.markdown("---")
    st.markdown("#### 📊 Feature Importance Analysis")
    with st.spinner("Computing feature importance..."):
        importance_df = compute_feature_importance(df)

    if importance_df.empty:
        st.info("Not enough numeric columns for feature importance analysis (need ≥ 2).")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
        with col2:
            st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA DRIFT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "drift":
    st.markdown('<div class="section-title">🔄 Data Drift Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Upload two datasets to compare their distributions and detect data drift.")

    col1, col2 = st.columns(2)
    with col1:
        file_old = st.file_uploader("Dataset A (Baseline / Old)", type=["csv", "xlsx"], key="drift_old")
    with col2:
        file_new = st.file_uploader("Dataset B (New / Current)", type=["csv", "xlsx"], key="drift_new")

    if file_old and file_new:
        df_old, err1 = load_dataset(file_old)
        df_new, err2 = load_dataset(file_new)

        if err1 or err2:
            st.error(f"Error loading files: {err1 or err2}")
        else:
            st.session_state.df_old = df_old
            st.session_state.df_new = df_new

            if st.button("🔍 Analyze Drift", type="primary"):
                with st.spinner("Comparing datasets..."):
                    results = compare_datasets(df_old, df_new)
                    st.session_state.drift_results = results

    if st.session_state.drift_results:
        results = st.session_state.drift_results
        df_old = st.session_state.df_old
        df_new = st.session_state.df_new

        if results["drift_detected"]:
            st.error("🚨 **Data Drift Detected!** Significant distribution changes found.")
        else:
            st.success("✅ No significant data drift detected.")

        # Shape changes
        sc = results["shape_changes"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset A Rows", format_number(sc["old_rows"]))
        c2.metric("Dataset B Rows", format_number(sc["new_rows"]),
                  delta=sc["new_rows"] - sc["old_rows"])
        c3.metric("New Columns", ", ".join(sc["new_columns"]) or "None")
        c4.metric("Removed Columns", ", ".join(sc["removed_columns"]) or "None")

        st.markdown("---")
        st.markdown("#### 📊 Column-Level Drift Analysis")

        drifted = [r for r in results["column_results"] if r["drift"]]
        stable = [r for r in results["column_results"] if not r["drift"]]

        st.markdown(f"🔴 **{len(drifted)} drifted** columns &nbsp;|&nbsp; 🟢 **{len(stable)} stable** columns")

        for col_res in results["column_results"]:
            drift_badge = "🔴 Drifted" if col_res["drift"] else "🟢 Stable"
            with st.expander(f"{drift_badge} — {col_res['column']}"):
                st.markdown(col_res["summary"])
                if col_res["type"] == "numeric":
                    st.markdown(f"- Old mean: **{col_res['old_mean']}** → New mean: **{col_res['new_mean']}**")
                    st.markdown(f"- KS Statistic: **{col_res['ks_statistic']}**, p-value: **{col_res['ks_p_value']}**")
                elif col_res["type"] == "categorical" and col_res.get("category_shifts"):
                    shifts_df = pd.DataFrame(col_res["category_shifts"]).T
                    st.dataframe(shifts_df, use_container_width=True)

                if col_res["column"] in df_old.columns and col_res["column"] in df_new.columns:
                    fig = drift_comparison_chart(df_old, df_new, col_res["column"])
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NATURAL LANGUAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "report":
    if not has_dataset():
        no_data_warning("Natural Language Report")

    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df if has_cleaned_dataset() else df
    profile = st.session_state.profile or get_profile(df)
    health_before = st.session_state.quality_before or compute_health_score(df)
    health_after = st.session_state.quality_after or health_before
    insights = st.session_state.insights or extract_insights(df)
    cleaning_log = st.session_state.cleaning_log or []

    st.markdown('<div class="section-title">📝 Natural Language Report</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Analysis Report", "📁 Dataset Documentation"])

    with tab1:
        if st.button("🔄 Generate Analysis Report", type="primary"):
            with st.spinner("Generating report..."):
                report = generate_nl_report(
                    df, cleaned_df, profile, health_before, health_after, insights, cleaning_log
                )
                st.session_state.report_text = report
                # Pre-generate PDF bytes for reliability
                st.session_state.report_pdf_bytes = text_to_pdf_bytes(report)

        if st.session_state.report_text:
            st.text_area("Report Preview", st.session_state.report_text, height=480, label_visibility="collapsed")
            st.download_button(
                "⬇️ Download Analysis Report (PDF)",
                data=st.session_state.get("report_pdf_bytes", b""),
                file_name="analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with tab2:
        if st.button("🔄 Generate Dataset Documentation", type="primary"):
            with st.spinner("Generating documentation..."):
                doc = generate_documentation(df, st.session_state.filename or "dataset")
                st.session_state.doc_text = doc
                st.session_state.doc_pdf_bytes = text_to_pdf_bytes(doc)

        if st.session_state.doc_text:
            st.text_area("Documentation Preview", st.session_state.doc_text, height=480, label_visibility="collapsed")
            st.download_button(
                "⬇️ Download Documentation (PDF)",
                data=st.session_state.get("doc_pdf_bytes", b""),
                file_name="dataset_documentation.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI DATA ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "assistant":
    if not has_dataset():
        no_data_warning("AI Data Assistant")

    df = st.session_state.df
    quality = st.session_state.quality_before or compute_health_score(df)
    insights = st.session_state.insights or extract_insights(df)
    cleaning_log = st.session_state.cleaning_log or []

    st.markdown('<div class="section-title">🤖 AI Data Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your dataset in plain English.")

    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Which column has the most missing values?
        - How many duplicate rows were found?
        - What is the strongest correlation?
        - How many rows does the dataset have?
        - What is the dataset health score?
        - Tell me about the Age column
        - What outliers were detected?
        """)

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">👤 {msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["text"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Your question:", placeholder="e.g. Which column has the most missing values?",
                                 label_visibility="collapsed")
        submitted = st.form_submit_button("Ask ➤", type="primary")

    if submitted and question.strip():
        assistant = QueryAssistant(df, quality, insights, cleaning_log)
        answer = assistant.answer(question)
        st.session_state.chat_history.append({"role": "user", "text": question})
        st.session_state.chat_history.append({"role": "bot", "text": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOAD RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "download":
    if not has_dataset():
        no_data_warning("Download Results")

    st.markdown('<div class="section-title">⬇️ Download Results</div>', unsafe_allow_html=True)

    # ── 1. Clean Dataset ──────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Clean Dataset")
    if has_cleaned_dataset():
        cleaned_df = st.session_state.cleaned_df
        st.success(f"✅ Cleaned dataset ready — {len(cleaned_df):,} rows, {cleaned_df.shape[1]} columns")
        
        st.download_button(
            "⬇️ Download Clean Dataset (CSV)",
            data=dataframe_to_csv_bytes(cleaned_df),
            file_name="clean_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    else:
        st.warning("⚠️ Original dataset only (Uncleaned)")
        st.download_button(
            "⬇️ Download Original Dataset (CSV)",
            data=dataframe_to_csv_bytes(st.session_state.df),
            file_name="original_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ── 2. Analysis Report & Documentation ────────────────────────────────────
    st.markdown("#### � PDF Reports")
    
    # Auto-generate if not yet done
    df = st.session_state.df
    if not st.session_state.report_text:
        cleaned_df2 = st.session_state.cleaned_df if has_cleaned_dataset() else df
        profile = st.session_state.profile or get_profile(df)
        h_before = st.session_state.quality_before or compute_health_score(df)
        h_after = st.session_state.quality_after or h_before
        insights = st.session_state.insights or extract_insights(df)
        log = st.session_state.cleaning_log or []
        st.session_state.report_text = generate_nl_report(df, cleaned_df2, profile, h_before, h_after, insights, log)
        st.session_state.report_pdf_bytes = text_to_pdf_bytes(st.session_state.report_text)
    
    if "report_pdf_bytes" not in st.session_state:
        st.session_state.report_pdf_bytes = text_to_pdf_bytes(st.session_state.report_text)

    if not st.session_state.doc_text:
        st.session_state.doc_text = generate_documentation(df, st.session_state.filename or "dataset")
        st.session_state.doc_pdf_bytes = text_to_pdf_bytes(st.session_state.doc_text)

    if "doc_pdf_bytes" not in st.session_state:
        st.session_state.doc_pdf_bytes = text_to_pdf_bytes(st.session_state.doc_text)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "📄 Download Analysis Report (PDF)",
            data=st.session_state.report_pdf_bytes,
            file_name="analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    with col_b:
        st.download_button(
            "📋 Download Documentation (PDF)",
            data=st.session_state.doc_pdf_bytes,
            file_name="dataset_documentation.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )


    # ── 4. Before vs After Summary ────────────────────────────────────────────
    if has_cleaned_dataset() and st.session_state.quality_before and st.session_state.quality_after:
        st.markdown("---")
        st.markdown("#### 📊 Before vs After Cleaning Summary")
        q_b = st.session_state.quality_before
        q_a = st.session_state.quality_after
        summary_df = pd.DataFrame({
            "Metric": ["Health Score", "Missing Values %", "Duplicate Rows", "Outliers"],
            "Before": [
                str(f"{q_b['health_score']}/100"), str(f"{q_b['missing_pct']}%"),
                str(q_b["duplicate_count"]), str(q_b["total_outliers"])
            ],
            "After": [
                str(f"{q_a['health_score']}/100"), str(f"{q_a['missing_pct']}%"),
                str(q_a["duplicate_count"]), str(q_a["total_outliers"])
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        fig = before_after_comparison(
            q_b["total_missing"], q_a["total_missing"],
            q_b["total_outliers"], q_a["total_outliers"],
        )
        st.plotly_chart(fig, use_container_width=True)


