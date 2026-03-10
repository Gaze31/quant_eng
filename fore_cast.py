# ...existing code...
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

file_path = "data.csv"

def load_csv_safe(path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    size = os.path.getsize(path)
    if size == 0:
        raise pd.errors.EmptyDataError(f"{path} is empty ({size} bytes)")
    try:
        # try normal read (tolerant to bad lines)
        return pd.read_csv(path, engine="python", sep=",", on_bad_lines='skip', **kwargs)
    except pd.errors.EmptyDataError:
        raise
    except Exception:
        # last resort: header=None to recover files without headers
        return pd.read_csv(path, engine="python", sep=",", header=None, on_bad_lines='skip', **kwargs)

df = load_csv_safe(file_path)
if df.empty or df.shape[1] == 0:
    raise pd.errors.EmptyDataError(f"No columns to parse from file: {file_path} (size={os.path.getsize(file_path)} bytes)")

print("Columns:", df.columns)
# ...existing code...
series = df['value']  # ensure 'value' exists or choose correct column after print(df.columns)
# ...existing code...

