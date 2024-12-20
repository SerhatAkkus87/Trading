import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import yfinance as yf
import numpy as np
from fredapi import Fred

start_date = '2022-01-10'
date_today = dt.datetime.now()
end_date = date_today

symbols = ['SP500', 'M2SL']
fred = Fred(api_key='307e76f43ff899089ea8b4591912d3e8')

data_m2sl = fred.get_series(series_id='M2SL', observation_start='02/10/2022')
data_m2sl.index.name = 'Date'
data_m2sl.columns = ['M2SL']
print(data_m2sl.head())

data_m1sl = fred.get_series(series_id='M1SL', observation_start='02/10/2022')
data_m1sl.index.name = 'Date'
data_m1sl.columns = ['M1SL']
print(data_m1sl.head())

data_btc = yf.download(tickers=['BTC-USD'], start=start_date, interval='1mo')

data_btc_norm = (data_btc-data_btc.min()) / (data_btc.max() - data_btc.min())
data_m1sl_norm = (data_m1sl-data_m1sl.min()) / (data_m1sl.max() - data_m1sl.min())
data_m2sl_norm = (data_m2sl-data_m2sl.min()) / (data_m2sl.max() - data_m2sl.min())

data_m1sl_shift = data_m1sl_norm.shift(2)
data_m2sl_shift = data_m2sl_norm.shift(2)

fig, ax = plt.subplots()
ax.plot(data_m1sl_norm, 'b-', label='M1SL')
ax.plot(data_m2sl_norm, 'r-', label='M2SL')
ax.plot(data_btc_norm['Close'], 'o-', label='BTC')
ax.legend(loc=2)
plt.show()


# Testing Pairwise Cointegration
threshold = 0.1

# perform test for the current pair of stocks
score, pvalue, _ = coint(data_m1sl, data_m2sl)

# check if the current pair of stocks is cointegrated
if pvalue < threshold:
    print('M1SL and M2SL', 'are cointegrated')
else:
    print('M1SL and M2SL', 'are not cointegrated')

print(data_m1sl.corr(data_m2sl))
