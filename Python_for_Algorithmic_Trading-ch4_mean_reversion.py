# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data = data[['Close']][-int(252):]
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))

SMA = 25
data['SMA'] = data['price'].rolling(SMA).mean()
threshold = 120
data['distance'] = data['price'] - data['SMA']
data['distance'].dropna().plot(figsize=(10, 6), legend=True)

plt.axhline(threshold, color='r')
plt.axhline(-threshold, color='r')
plt.axhline(0, color='r')
plt.show()


data['position'] = np.where(data['distance'] > threshold, -1, np.nan)
data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])
data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0, 0, data['position'])
data['position'] = data['position'].ffill().fillna(0)
data['position'].iloc[SMA:].plot(ylim=[-1.1, 1.1], figsize=(10, 6))
plt.show()


data['strategy'] = data['position'].shift(1) * data['returns']
data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
