"""
generate_sample.py
------------------
Creates a realistic sample dataset with intentional data quality issues
for testing the Dataset Analyzer application.

Run: python generate_sample.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

categories = ["Electronics", "Clothing", "Food", "Books", "Sports"]
cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
genders = ["Male", "Female", "Other"]

# Core data
df = pd.DataFrame({
    "CustomerID": range(1001, 1001 + n),
    "Age": np.random.randint(18, 70, n).astype(float),
    "Gender": np.random.choice(genders, n, p=[0.5, 0.45, 0.05]),
    "City": np.random.choice(cities, n),
    "Category": np.random.choice(categories, n, p=[0.35, 0.25, 0.20, 0.10, 0.10]),
    "Advertising_Spend": np.random.normal(5000, 1500, n).round(2),
    "Sales": None,  # Will correlate with Advertising
    "Income": np.random.lognormal(10.5, 0.5, n).round(2),
    "Purchase_Frequency": np.random.randint(1, 30, n).astype(float),
    "Satisfaction_Score": np.random.randint(1, 6, n).astype(float),
    "Discount_Applied": np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    "Return_Rate": np.random.uniform(0, 0.3, n).round(4),
    "Date_Joined": pd.date_range("2020-01-01", periods=n, freq="8h").astype(str),
})

# Correlate Sales with Advertising
df["Sales"] = (df["Advertising_Spend"] * 1.8 + np.random.normal(0, 1000, n)).round(2)

# --- Inject data quality issues ---

# 1. Missing Values (~8-12%)
missing_idx_age = np.random.choice(n, int(n * 0.08), replace=False)
df.loc[missing_idx_age, "Age"] = np.nan

missing_idx_income = np.random.choice(n, int(n * 0.12), replace=False)
df.loc[missing_idx_income, "Income"] = np.nan

missing_idx_city = np.random.choice(n, int(n * 0.05), replace=False)
df.loc[missing_idx_city, "City"] = np.nan

missing_idx_sat = np.random.choice(n, int(n * 0.06), replace=False)
df.loc[missing_idx_sat, "Satisfaction_Score"] = np.nan

# 2. Duplicate Rows (~2%)
dup_indices = np.random.choice(n, int(n * 0.02), replace=False)
df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

# 3. Outliers in a few numeric columns
df.loc[10, "Advertising_Spend"] = 90000  # extreme outlier
df.loc[25, "Sales"] = 200000
df.loc[50, "Income"] = 5000000
df.loc[75, "Age"] = 150  # invalid age

# 4. A column that looks numeric but stored as string (type error simulation)
df["Satisfaction_Score"] = df["Satisfaction_Score"].apply(
    lambda x: str(int(x)) if not pd.isna(x) else np.nan
)

# Save
output_path = os.path.join(os.path.dirname(__file__), "assets", "sample_dataset.csv")
df.to_csv(output_path, index=False)
print(f"[OK] Sample dataset saved: {output_path}")
print(f"     Rows: {len(df)}, Columns: {df.shape[1]}")
print(f"     Missing values: {df.isnull().sum().sum()}")
print(f"     Duplicates: {df.duplicated().sum()}")
