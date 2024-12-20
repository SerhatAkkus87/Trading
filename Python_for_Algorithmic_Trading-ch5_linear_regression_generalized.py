# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

start_in = '2020-12-15'
end_in = '2023-01-15'
start_out = '2023-01-16'
end_out = '2024-12-15'
lags = 200

data = yf.download(tickers=['BTC-USD'], start=start_in, end=end_out, interval='1d')
data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# data = pd.read_csv('daily_data1.csv')
# data = data.set_index('Datetime')

data = data[['Close']]

data = pd.DataFrame(data['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))


import LRVectorBacktester as LR

# Instantiates an object of the LRVectorBacktester class.
lrbt = LR.LRVectorBacktester('price', start_in, end_out, 10000, 0.001, raw_data=data)

# Trains and evaluates the strategy on the same data set.
lrbt.run_strategy(start_in=start_in, end_in=end_in, start_out=start_out, end_out=end_out, lags=30)

# Uses two different data sets for the training and evaluation steps.
#lrbt.run_strategy(data.index[0], data.index[-1], data.index[0], data.index[-1], lags=5)

# Plots the out of sample strategy performance compared to the market.
lrbt.plot_results()
plt.show()
