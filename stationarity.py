import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from ta import momentum, volume, volatility, trend
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess

print('Get data and calculate returns.')
data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')
data = data['2024-08-01':]
data['return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)


# p-value < 0.05 -> stationary
print('Check for stationarity...')
ADF_result = adfuller(data['return'])
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# Check for auto-correlation
plot_acf(data['return'], lags=10)
plt.show()

diff_random_walk = np.diff(data['return'], n=1)

# p-value < 0.05 -> stationary
print('Check for stationarity...')
ADF_result = adfuller(diff_random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

# Check for auto-correlation
plot_acf(diff_random_walk, lags=10)
plt.show()


