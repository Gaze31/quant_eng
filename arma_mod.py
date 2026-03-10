import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# python
import pandas as pd
df = pd.read_csv('data.csv', sep=',', engine='python', on_bad_lines='warn')  # or on_bad_lines='skip'
# python
df = pd.read_csv('data.csv', sep=',', quotechar='"', engine='python')
# Load sample time series
data = sm.datasets.co2.load_pandas().data['co2'].ffill()

# Convert to monthly average (optional)
data = data.resample('M').mean()

plt.figure(figsize=(10,4))
plt.plot(data)
plt.title("CO2 Levels")
plt.show()
from statsmodels.tsa.stattools import adfuller

# python
with open('data.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i == 13:
            print(repr(line))
            break

result = adfuller(data)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
diff_data = data.diff().dropna()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(diff_data)
plot_pacf(diff_data)
plt.show()
