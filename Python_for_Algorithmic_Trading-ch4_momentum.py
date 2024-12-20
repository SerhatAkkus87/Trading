# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data['SMA1'] = data['Close'].rolling(42).mean()
data['SMA2'] = data['Close'].rolling(252).mean()

data = data[['Close', 'SMA1', 'SMA2']]

data = pd.DataFrame(data['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))

# data['position'] = np.sign(data['returns'])
# data['strategy'] = data['position'].shift(1) * data['returns']
# data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()
#
# data['position'] = np.sign(data['returns'].rolling(3).mean())
# data['strategy'] = data['position'].shift(1) * data['returns']
# data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
# plt.show()

data['returns'] = np.log(data['price'] / data['price'].shift(1))

to_plot = ['returns']

for m in [1, 3, 5, 7, 9]:
    data['position_%d' % m] = np.sign(data['returns'].rolling(m).mean())
    data['strategy_%d' % m] = (data['position_%d' % m].shift(1) * data['returns'])
    to_plot.append('strategy_%d' % m)

data[to_plot].dropna().cumsum().apply(np.exp).plot(
    title='BTC intraday',
    style=['-', '--', '--', '--', '--', '--'])

plt.show()

