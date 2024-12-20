import ta.volume
import yfinance as yf
import datetime as dt
from datetime import timezone, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

start_date = '2024-08-01'
df = yf.download(tickers=['BTC-USD'], start=start_date, interval='1h')

random_walk = df['Close']
random_walk.columns = ['Close']
random_walk.plot.line()
plt.show()

print(df.describe())

adf_result = adfuller(random_walk)
print(adf_result[1])


random_walk_diff = np.diff(random_walk, n=1, axis=0)
adf_result2 = adfuller(random_walk_diff)
print(adf_result2[1])

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(random_walk_diff, lags=30);

plt.tight_layout()
plt.show()

threshold = int(random_walk.shape[0] * 0.8)

train = random_walk[:threshold]
test = random_walk[threshold:]
print(test.head())

print('SHAPES')
print(f'threshold: {threshold}')
print(f'shape[0]: {random_walk.shape[0]}')

mean = np.mean(train.Close)
test.loc[:, 'pred_mean'] = mean
last_value = train.iloc[-1].Close
test.loc[:, 'pred_last'] = last_value

deltaX = threshold - 1

deltaY = last_value - train.iloc[0].Close
drift = deltaY / deltaX
print(drift)

x_vals = np.arange(threshold, random_walk.shape[0], 1)
pred_drift = drift * x_vals + random_walk.Close[threshold]
test.loc[:, 'pred_drift'] = pred_drift


fig, ax = plt.subplots()

ax.plot(train.Close, 'b-')
ax.plot(test.Close, 'b-')
ax.plot(test['pred_mean'], 'r-.', label='Mean')
ax.plot(test['pred_last'], 'g--', label='Last value')
ax.plot(test['pred_drift'], 'k:', label='Drift')

ax.axvspan(random_walk.index[threshold], random_walk.index[random_walk.shape[0] - 1], color='#808080', alpha=0.2)

ax.legend(loc=2)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Close')
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error
mse_mean = mean_squared_error(test['Close'], test['pred_mean'])
mse_last = mean_squared_error(test['Close'], test['pred_last'])
mse_drift = mean_squared_error(test['Close'], test['pred_drift'])
print(mse_mean, mse_last, mse_drift)

df_shift = df['Close'].shift(periods=1)
df_shift.head()

fig, ax = plt.subplots()
ax.plot(df['Close'], 'b-', label='actual')
ax.plot(df_shift, 'r-.', label='forecast')
ax.legend(loc=2)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()
plt.show()


