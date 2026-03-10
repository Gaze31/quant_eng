import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os
print(os.getcwd())
file_path = "data.csv"

def check_broken_lines(file_path):
    """Check for lines with unmatched quotes in CSV file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                if line.count('"') % 2 != 0:
                    print(f"Broken line {line_no}: {line.strip()[:100]}...")
        print("Quote check completed.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")

check_broken_lines(file_path)

try:
    df = pd.read_csv(
        file_path,
        engine='python',
        quotechar='"',
        skipinitialspace=True,
        on_bad_lines='skip'
    )
except pd.errors.ParserError:
    df = pd.read_csv(
        file_path,
        engine='python',
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip'
    )

print(df.head())


def plot_missing_data(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    plt.show()

def plot_distribution(df, column):
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns found.")
        return
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_time_series(df, date_col, value_col):
    if date_col not in df.columns or value_col not in df.columns:
        print("Columns not found.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df[date_col], errors='coerce'), df[value_col])
    plt.title(f"Time Series of {value_col}")
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, column):
    if column not in df.columns:
        print("Column not found.")
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column].dropna())
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.show()

