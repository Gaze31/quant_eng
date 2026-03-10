import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

def load_csv_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        raise pd.errors.EmptyDataError(f"{path} is empty")
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty or "value" not in df.columns:
        raise pd.errors.EmptyDataError(f"No usable data in {path}")
    df = df.sort_values("date").set_index("date")
    return df["value"].astype(float)

def train_test_split_series(s, test_size=12):
    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]
    return train, test

def holt_winters_forecast(train, steps=12):
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit(optimized=True)
    return fit.forecast(steps)

def arima_forecast(train, steps=12, seasonal=False):
    model = auto_arima(train, seasonal=seasonal, error_action="ignore", suppress_warnings=True)
    preds = model.predict(n_periods=steps)
    return pd.Series(preds, index=pd.date_range(start=train.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="M"))

def evaluate(pred, test):
    mape = mean_absolute_percentage_error(test, pred)
    rmse = mean_squared_error(test, pred, squared=False)
    return {"mape": float(mape), "rmse": float(rmse)}

def plot_results(train, test, preds_hw, preds_arima):
    plt.figure(figsize=(10,4))
    plt.plot(train.index, train, label="train")
    plt.plot(test.index, test, label="test")
    plt.plot(preds_hw.index, preds_hw, label="Holt-Winters")
    plt.plot(preds_arima.index, preds_arima, label="Auto-ARIMA")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(data_path, horizon=12, no_plot=False):
    s = load_csv_safe(data_path)
    train, test = train_test_split_series(s, test_size=horizon)

    preds_hw = holt_winters_forecast(train, steps=horizon)
    if not isinstance(preds_hw, pd.Series):
        preds_hw = pd.Series(preds_hw, index=test.index)

    preds_arima = arima_forecast(train, steps=horizon, seasonal=True)

    eval_hw = evaluate(preds_hw, test)
    eval_arima = evaluate(preds_arima, test)

    out = pd.DataFrame({
        "test": test,
        "hw": preds_hw,
        "arima": preds_arima
    })
    os.makedirs("output", exist_ok=True)
    out.to_csv("output/forecasts.csv")
    print("Saved output/forecasts.csv")
    print("Holt-Winters:", eval_hw)
    print("Auto-ARIMA:", eval_arima)

    if not no_plot:
        plot_results(train, test, preds_hw, preds_arima)

# ...existing code...
import os
import numpy as np
import pandas as pd
df = pd.read_csv("data/data.csv")
# print(df.head())
# print(df.columns)
# FutureWarning: 'M' is deprecated ... use 'ME' instead.
# rng = pd.date_range(start=start, periods=n_months, freq="ME")
print(df.head())
print(df.columns)

# ...existing code...
def make_monthly_series(n_months=120, start="2010-01-01", seed=0):
    """Generate a synthetic monthly time series with trend + seasonality + noise."""
    np.random.seed(seed)
    rng = pd.date_range(start=start, periods=n_months, freq="M")
    trend = np.linspace(0, 10, n_months)
    season = 5 * np.sin(2 * np.pi * (np.arange(n_months) % 12) / 12)
    noise = np.random.normal(scale=1.0, size=n_months)
    values = 50 + trend + season + noise
    return pd.DataFrame({"date": rng, "value": values})

def main(output="data/data.csv", n_months=120, start="2010-01-01"):
    df = make_monthly_series(n_months=n_months, start=start)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {output} rows={len(df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/data.csv", help="output CSV path")
    parser.add_argument("--months", type=int, default=120, dest="months", help="number of months")
    parser.add_argument("--start", default="2010-01-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    main(output=args.output, n_months=args.months, start=args.start)
# ...existing code...