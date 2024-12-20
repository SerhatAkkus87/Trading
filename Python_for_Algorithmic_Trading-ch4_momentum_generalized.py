# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data = data[['Close']][-int(60):]

data = pd.DataFrame(data['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))

import MomVectorBacktester as Mom

mombt = Mom.MomVectorBacktester('price', data.index[0], data.index[-1], 10000, 0.0, data)

mombt.run_strategy(momentum=3)

mombt.plot_results()
plt.show()

mombt = Mom.MomVectorBacktester('price', data.index[0], data.index[-1], 10000, 0.001, data)

mombt.run_strategy(momentum=3)

mombt.plot_results()
plt.show()
