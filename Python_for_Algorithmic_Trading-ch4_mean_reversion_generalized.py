# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data = data[['Close']][-int(252*5):]
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))

import MRVectorBacktester as MR

mrbt = MR.MRVectorBacktester('price', data.index[0], data.index[-1], 10000, 0.001, data_raw=data)

mrbt.run_strategy(SMA=15, threshold=200)
mrbt.plot_results()
plt.show()
