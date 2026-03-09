"""
data_loader.py
--------------
Handles dataset uploading, validation, and loading into pandas DataFrames.
"""

import pandas as pd
import io
from typing import Optional, Tuple


def load_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        Tuple of (DataFrame or None, error message string).
    """
    if uploaded_file is None:
        return None, "No file uploaded."

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel files."

        if df.empty:
            return None, "The uploaded file is empty."

        return df, ""
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def get_dataset_metadata(df: pd.DataFrame, uploaded_file) -> dict:
    """
    Extract basic metadata from a DataFrame and the uploaded file.

    Args:
        df: The loaded pandas DataFrame.
        uploaded_file: The original Streamlit UploadedFile object.

    Returns:
        A dictionary containing metadata about the dataset.
    """
    file_size_kb = round(uploaded_file.size / 1024, 2) if hasattr(uploaded_file, "size") else "N/A"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "filename": uploaded_file.name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "file_size_kb": file_size_kb,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_types": df.dtypes.astype(str).to_dict(),
    }
