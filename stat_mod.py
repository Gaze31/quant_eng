import statsmodels.api as sm
import pandas as pd

data = {
    "experience": [1,2,3,4,5,6],
    "salary": [40,45,50,60,65,70]
}

df = pd.DataFrame(data)

X = df["experience"]
y = df["salary"]

X = sm.add_constant(X)  # adds intercept
model = sm.OLS(y, X).fit()

print(model.summary())

import statsmodels.api as sm

# Sample time series
data = sm.datasets.co2.load_pandas().data["co2"].ffill()

model = sm.tsa.ARIMA(data, order=(1,1,1)).fit()
print(model.summary())
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plot_acf(data)
plot_pacf(data)
plt.show()
from statsmodels.tsa.stattools import adfuller

result = adfuller(data)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
