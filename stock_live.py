import yfinance as yf
import matplotlib.pyplot as plt
# plt.style.use('seaborn')
import seaborn as sns
sns.set_style("darkgrid")  # sets Seaborn style

import pandas as pd
from matplotlib.animation import FuncAnimation  
import datetime as dt
import numpy as np

plt.style.use('ggplot')   # built-in
# or
# plt.style.use('bmh')
# or check available styles
print(plt.style.available)


symbol = "AAPL"
start_date = dt.datetime.now() - dt.timedelta(days=365)

# initial download
data = yf.download(symbol, start=start_date)

# Calculate moving averages
data["50_MA"] = data["Close"].rolling(window=50).mean()
data["200_MA"] = data["Close"].rolling(window=200).mean()

fig, ax = plt.subplots(figsize=(12, 6))

def animate(i):
    # Fetch latest data point
    latest = yf.download(symbol, period="5d", interval="1m")

    # Merge latest data into main dataset
    if not latest.empty:
        data.loc[latest.index, "Close"] = latest["Close"]

    data["50_MA"] = data["Close"].rolling(window=50).mean()
    data["200_MA"] = data["Close"].rolling(window=200).mean()

    ax.clear()
    ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
    ax.plot(data.index, data["50_MA"], label="50-Day MA", color="orange")
    ax.plot(data.index, data["200_MA"], label="200-Day MA", color="red")

    ax.set_title(f"Live Stock Price of {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

ani = FuncAnimation(fig, animate, interval=60000)  # update every 1 min
plt.show()
