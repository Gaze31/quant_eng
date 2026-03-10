import pandas as pd

# Option A: Load your own time-series data
# data = pd.read_csv("your_timeseries_file.csv", parse_dates=True, index_col="Date")

# Option B: Example data - Monthly sales for 5 years
date_rng = pd.date_range(start="2018-01-01", end="2022-12-01", freq="MS")
import numpy as np
data = pd.DataFrame({"Sales": np.random.randint(200, 700, len(date_rng))}, index=date_rng)

print(data.head())
train = data.iloc[:-12]   # First N-1 years
test  = data.iloc[-12:]   # Last 1 year for forecasting
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train["Sales"],
    order=(1,1,1),         # ARIMA(p,d,q)
    seasonal_order=(1,1,1,12)  # (P,D,Q,m)  — monthly seasonality = 12
)

result = model.fit()
print(result.summary())
forecast = result.predict(start=len(train), end=len(data)+12, dynamic=False)

# Split values
y_pred = forecast[:len(test)]
future_pred = forecast[len(test):]  # Forecast future 12 months

print(y_pred)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(train.index, train["Sales"], label="Training Data")
plt.plot(test.index, test["Sales"], label="Actual Test")
plt.plot(y_pred.index, y_pred, label="SARIMA Predictions", linewidth=2)
plt.plot(future_pred.index, future_pred, label="Future Forecast", linestyle="--", color="black")

plt.title("SARIMA Forecasting")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test, y_pred)
rmse = np.sqrt(mean_squared_error(test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
