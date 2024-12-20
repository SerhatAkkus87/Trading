# Chapter 4. Mastering Vectorized Backtesting

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data = data[['Close']][-int(252*500):]

print(data.head())

# Generalizing the Approach

# SMAVectorBacktester parameter:
#   symbol: RIC (instrument data) to be used
#   SMA1: for the time window in days for the shorter SMA
#   SMA2: for the time window in days for the longer SMA
#   start: for the start date of the data selection
#   end: for the end date of the data selection


# This imports the module as SMA.
smabt = SMA.SMAVectorBacktester('Close', 42, 252, data.index[0], data.index[-1], data)

# An instance of the main class is instantiated.
smabt.run_strategy()

# Backtests the SMA-based strategy, given the parameters during instantiation.
smabt.optimize_parameters((30, 50, 2), (200, 300, 2))

# The optimize_parameters() method takes as input parameter ranges with step sizes and determines the optimal
# combination by a brute force approach.
smabt.plot_results()
plt.show()

