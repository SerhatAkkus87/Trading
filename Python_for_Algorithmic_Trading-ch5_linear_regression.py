# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt
import yfinance as yf

# data = pd.read_csv('daily_data1.csv')
# data = data.set_index('Datetime')
#
# data = data[['Close']]
#
# data = pd.DataFrame(data['Close'])
# data.rename(columns={'Close': 'price'}, inplace=True)
# data['returns'] = np.log(data['price'] / data['price'].shift(1))
# data = data['2024-05-01':'2024-06-01']

start_date = '2023-01-17'
end_date = '2024-12-17'
data = yf.download(tickers=['BTC-USD'], start=start_date, end=end_date, interval='1d')
data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data = pd.DataFrame(data['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)

lags = 200

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    # Takes the price column and shifts it by lag.
    data[col] = data['price'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['price'], rcond=None)[0]

# Calculates the prediction values as the dot product.
data['prediction'] = np.dot(data[cols], reg)
# Plots the price and prediction columns.
data[['price', 'prediction']].plot(figsize=(10, 6))
plt.show()


# Predicting Future Returns
data['return'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    # Takes the returns column for the lagged data.
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['return'], rcond=None)[0]
print(reg)

# From a trading point of view, one might argue that it is not the magnitude of
# the forecasted return that is relevant, but rather whether the direction is
# forecasted correctly or not. To this end, a simple calculation yields an
# overview. Whenever the linear regression gets the direction right, meaning
# that the sign of the forecasted return is correct, the product of the market
# return and the predicted return is positive and otherwise negative.
data['prediction'] = np.dot(data[cols], reg)
data[['return', 'prediction']].iloc[lags:].plot(figsize=(10, 6))
plt.show()

# Calculates the product of the market and predicted return, takes the sign of the results and counts the values.
hits = np.sign(data['return'] * data['prediction']).value_counts()

# Prints out the counts for the two possible values.
print(hits)

# Calculates the hit ratio defined as the number of correct predictions given all predictions.
print(hits.values[0] / sum(hits))


# Predicting Future Market Direction

# This directly uses the sign of the return to be predicted for the regression.
reg = np.linalg.lstsq(data[cols], np.sign(data['return']), rcond=None)[0]

# Also, for the prediction step, only the sign is relevant.
data['prediction'] = np.sign(np.dot(data[cols], reg))
print(data['prediction'].value_counts())

hits = np.sign(data['return'] * data['prediction']).value_counts()
print(hits)
print(hits.values[0] / sum(hits))


# Vectorized Backtesting of Regression-Based Strategy
print(data.head())
print(data.columns)

# Multiplies the prediction values (positionings) by the market returns.
data['strategy'] = data['prediction'] * data['return']

# Calculates the gross performance of the base instrument and the strategy.
data[['return', 'strategy']].sum().apply(np.exp)

# Plots the gross performance of the base instrument and the strategy over time (in-sample, no transaction costs).
data[['return', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()


