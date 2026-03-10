# ------------------- IMPORTS -------------------
import pandas as pd
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

try:
    from arch import arch_model
except:
    raise ImportError("Install GARCH first → pip install arch")


# ---------------- GARCH MODEL ------------------
def garch_forecast(train, steps=12, p=1, q=1):
    series = train.dropna().astype(float)
    am = arch_model(series, vol='Garch', p=p, q=q, mean='Constant', dist='normal')
    res = am.fit(disp='off')
    fc = res.forecast(horizon=steps, reindex=False).mean.iloc[-1].values
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq='M')
    return pd.Series(fc, index=idx, name='garch')


# ---------------- ARIMA -------------------------
def arima_forecast(train, steps=12):
    model = auto_arima(train, seasonal=True, stepwise=True)
    fc = model.predict(n_periods=steps)
    idx = pd.date_range(train.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq='M')
    return pd.Series(fc, index=idx, name='arima')


# ---------------- HOLT-WINTERS -------------------
def holt_winters_forecast(train, steps=12):
    fit = ExponentialSmoothing(train, trend="add").fit()
    fc = fit.forecast(steps)
    fc.index = pd.date_range(train.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq='M')
    return fc.rename("holt_winters")


# ---------------- TRAIN/TEST SPLIT ---------------
def split_series(s, horizon):
    return s.iloc[:-horizon], s.iloc[-horizon:]


# ---------------- PLOT ---------------------------
def plot_results(train, test, *predictions):
    plt.figure(figsize=(10,6))
    plt.plot(train, label="TRAIN")
    plt.plot(test, label="TEST")
    for p in predictions: plt.plot(p, label=p.name)
    plt.legend(); plt.title("Forecast Comparison"); plt.show()


# ---------------- MAIN EXECUTION -----------------
def main(file, horizon=12, no_plot=False):
    s = pd.read_csv(file, index_col=0, parse_dates=True).iloc[:,0]

    train, test = split_series(s, horizon)

    preds_hw = holt_winters_forecast(train, horizon)
    preds_arima = arima_forecast(train, horizon)

    try:
        preds_garch = garch_forecast(train, horizon)
    except:
        preds_garch = pd.Series([np.nan]*horizon, index=test.index, name="garch")

    out = pd.DataFrame({"actual":test, "holt_winters":preds_hw,
                        "arima":preds_arima, "garch":preds_garch})
    os.makedirs("output", exist_ok=True)
    out.to_csv("output/forecasts.csv")
    print("\n✔ Saved → output/forecasts.csv\n")

    if not no_plot:
        plot_results(train, test, preds_hw, preds_arima, preds_garch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()
    main(args.data_path, args.horizon, args.no_plot)
