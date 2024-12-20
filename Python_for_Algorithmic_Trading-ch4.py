# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')


# Three strategies in this chapter:
#   Simple moving averages (SMA) based strategies
#   Momentum strategies
#   Mean-reversion strategies

# Vectorized backtesting should be considered in the following cases:
#   Simple trading strategies
#   Interactive strategy exploration
#   Visualization as major goal
#   Comprehensive backtesting programs


# Strategies Based on Simple Moving Averages
print(data.info())

# Creates a column with 42 days of SMA values. The first 41 values will be NaN.
data['SMA1'] = data['Close'].rolling(42).mean()

# Creates a column with 252 days of SMA values. The first 251 values will be NaN.
data['SMA2'] = data['Close'].rolling(252).mean()

# Prints the final five rows of the data set.
print(data.tail())

data = data[['Close', 'SMA1', 'SMA2']][-int(252*5):]

# Visualize
from pylab import mpl, plt

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
plt.style.use("seaborn-v0_8")
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

data.plot(title='BTC/USD | 42 & 252 days SMAs', figsize=(10, 6), ax=ax1)


# Implements the trading rule in vectorized fashion. np.where() produces +1 for rows where the expression is
# True and -1 for rows where the expression is False.
data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data.dropna(inplace=True)
data['position'].plot(ylim=[-1.1, 1.1],
                      title='Market Positioning',
                      figsize=(10, 6),
                      ax=ax2)
plt.show()


# Calculate returns
# Calculates the log returns in vectorized fashion over the price column.
data['returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Plots the log returns as a histogram (frequency distribution).
data['returns'].hist(bins=35, figsize=(10, 6))
plt.show()

# Comparing the returns shows that the strategy books a win over the passive benchmark investment:

# Derives the log returns of the strategy given the positionings and market returns.
data['strategy'] = data['position'].shift(1) * data['returns']

# Sums up the single log return values for both the stock and the strategy (for illustration only).
data[['returns', 'strategy']].sum()

# Applies the exponential function to the sum of the log returns to calculate the gross performance.
data[['returns', 'strategy']].sum().apply(np.exp)

data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

# Calculates the annualized mean return in both log and regular space.
data[['returns', 'strategy']].mean() * 252
np.exp(data[['returns', 'strategy']].mean() * 252) - 1

# Calculates the annualized standard deviation in both log and regular space.
data[['returns', 'strategy']].std() * 252 ** 0.5
(data[['returns', 'strategy']].apply(np.exp) - 1).std() * 252 ** 0.5


# Defines a new column, cumret, with the gross performance over time.
data['cumret'] = data['strategy'].cumsum().apply(np.exp)

# Defines yet another column with the running maximum value of the gross performance.
data['cummax'] = data['cumret'].cummax()

# Plots the two new columns of the DataFrame object.
data[['cumret', 'cummax']].dropna().plot(figsize=(10, 6))
plt.show()

# Calculates the element-wise difference between the two columns.
drawdown = data['cummax'] - data['cumret']

# Picks out the maximum value from all differences.
drawdown.max()


# Vectorized backtesting with pandas is generally a rather efficient endeavor due to the capabilities
# of the package and the main DataFrame class.
# However, the interactive approach illustrated so far does not work well when one wishes to implement a
# larger backtesting program that, for example, optimizes the parameters of an SMA-based strategy.
# To this end, a more general approach is advisable.


