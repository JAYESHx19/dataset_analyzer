"""
helpers.py
----------
Streamlit session state utilities and general helper functions.
"""

import streamlit as st
import pandas as pd
import os

# Resolve base directory of the project (parent of utils/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_csv_to_disk(df: pd.DataFrame, filename: str = "clean_dataset.csv") -> str:
    """
    Save a DataFrame as CSV to the cleaned_data/ directory.
    Returns the absolute path of the saved file.
    """
    out_dir = os.path.join(BASE_DIR, "cleaned_data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_text_to_disk(text: str, filename: str) -> str:
    """
    Save a text string to the reports/ directory.
    Returns the absolute path of the saved file.
    """
    out_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path



def _strip_emojis(text: str) -> str:
    """
    Remove emoji and other high-unicode characters that standard PDF fonts
    cannot handle (e.g. Courier/Helvetica only support Latin-1).
    """
    import re
    # Remove characters outside the Basic Multilingual Plane (BMP) or just keep printable ASCII/Latin-1
    return "".join(c for c in text if ord(c) < 256)


def save_pdf_to_disk(text: str, filename: str) -> str:
    """
    Save a text string to the reports/ directory as a PDF file.
    Returns the absolute path of the saved file.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import textwrap

    out_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    
    # Ensure filename ends with .pdf
    if not filename.endswith('.pdf'):
        filename = filename.rsplit('.', 1)[0] + '.pdf'
        
    path = os.path.join(out_dir, filename)
    
    # Create PDF
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    
    # Clean text of emojis before processing
    safe_text = _strip_emojis(text)
    
    # Text settings
    c.setFont("Courier", 10)
    text_object = c.beginText(40, height - 40)
    
    # Process text line by line, handling wrapping
    lines = safe_text.split('\n')
    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=88) if line.strip() else [""]
        for wl in wrapped_lines:
            if text_object.getY() < 40:
                c.drawText(text_object)
                c.showPage()
                c.setFont("Courier", 10)
                text_object = c.beginText(40, height - 40)
            text_object.textLine(wl)
            
    c.drawText(text_object)
    c.save()
    return path


def text_to_pdf_bytes(text: str) -> bytes:
    """
    Convert a text string to PDF bytes in memory.
    """
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import textwrap

    # Clean text first
    safe_text = _strip_emojis(text)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Courier", 10)
    text_object = c.beginText(40, height - 40)
    
    lines = safe_text.split('\n')
    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=88) if line.strip() else [""]
        for wl in wrapped_lines:
            if text_object.getY() < 40:
                c.drawText(text_object)
                c.showPage()
                c.setFont("Courier", 10)
                text_object = c.beginText(40, height - 40)
            text_object.textLine(wl)
            
    c.drawText(text_object)
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()


def init_session_state():
    """Initialize Streamlit session state variables if not already set."""
    defaults = {
        "df": None,
        "cleaned_df": None,
        "filename": None,
        "profile": None,
        "quality_before": None,
        "quality_after": None,
        "insights": None,
        "cleaning_log": [],
        "missing_strategy": {},
        "remove_duplicates": True,
        "outlier_strategy": "keep",
        "fix_types": True,
        "report_text": None,
        "doc_text": None,
        "chat_history": [],
        "df_old": None,
        "df_new": None,
        "drift_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_dataset() -> bool:
    """Check if a dataset has been uploaded."""
    return st.session_state.get("df") is not None


def has_cleaned_dataset() -> bool:
    """Check if a cleaned dataset exists."""
    return st.session_state.get("cleaned_df") is not None


def no_data_warning(page: str = "this page"):
    """Display a styled warning when no dataset is uploaded."""
    st.warning(f"⚠️ Please upload a dataset first to use **{page}**.", icon="📂")
    st.stop()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 encoded CSV bytes for downloading."""
    return df.to_csv(index=False).encode("utf-8")


def text_to_bytes(text: str) -> bytes:
    """Encode a text string to UTF-8 bytes for downloading (fallback)."""
    return text.encode("utf-8")


def pdf_to_bytes(filepath: str) -> bytes:
    """Read a PDF file and return its bytes for downloading via Streamlit."""
    with open(filepath, "rb") as f:
        return f.read()


def format_number(n) -> str:
    """Format large numbers with commas."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def health_score_color(score: float) -> str:
    """Return a hex color for a health score."""
    if score >= 90:
        return "#22C55E"
    elif score >= 75:
        return "#84CC16"
    elif score >= 50:
        return "#F59E0B"
    elif score >= 30:
        return "#EF4444"
    return "#991B1B"


def render_health_bar(score: float, label: str = "Dataset Health Score"):
    """Render a visual health score bar using Streamlit's progress."""
    color = health_score_color(score)
    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 14px; color: #000000; font-weight: 600;">{label}</span>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="flex: 1; background: #E5E7EB; border-radius: 999px; height: 16px; border: 1px solid #D1D5DB;">
            <div style="width: {score}%; background: {color}; height: 16px; border-radius: 999px;
                        transition: width 0.5s ease; border: 1px solid {color};"></div>
        </div>
        <span style="font-size: 22px; font-weight: 700; color: {color};">{score:.1f}<span
              style="font-size: 14px; color: #000000; font-weight: 500;">/100</span></span>
    </div>
    """, unsafe_allow_html=True)


def card_css() -> str:
    """Return shared CSS card styles."""
    return """
    <style>
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border: 1px solid #F1F5F9;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 12px;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.2;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #9CA3AF;
        margin-top: 4px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 14px;
        padding-bottom: 6px;
        border-bottom: 2px solid #E0E7FF;
    }
    .insight-card {
        background: linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 100%);
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .insight-card h4 {
        font-size: 15px;
        font-weight: 600;
        color: #312E81;
        margin: 0 0 4px 0;
    }
    .insight-card p {
        font-size: 13px;
        color: #374151;
        margin: 0;
    }
    .chat-bubble-user {
        background: #4F46E5;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 14px;
    }
    .chat-bubble-bot {
        background: #F1F5F9;
        color: #1E293B;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        font-size: 14px;
    }
    </style>
    """
