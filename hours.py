import ta.volume
import yfinance as yf
import datetime as dt
from datetime import timezone, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

start_date = '2024-08-01'
df_hours = yf.download(tickers=['BTC-USD'], start=start_date, interval='1h')
df_daily = yf.download(tickers=['BTC-USD'], start=start_date, interval='1d')
df_hours['return'] = df_hours['Close'].pct_change()
df_daily['return'] = df_daily['Close'].pct_change()

print(df_hours['return'])
print(df_daily['return'])

initial_wealth = 10000


def drawdown(return_series: pd.Series):
    """
    Input: a time series of asset returns

    Output: a DataFrame that contains:
    - the wealth index
    - the prior peaks
    - percentage drawdowns
    """

    wealth_index_series = initial_wealth * (1 + return_series).cumprod()
    prior_peaks_series = wealth_index_series.cummax()
    drawdown_series = (wealth_index_series - prior_peaks_series) / prior_peaks_series

    return pd.DataFrame({
        "Wealth index": wealth_index_series,
        "Prior peaks": prior_peaks_series,
        "Drawdown": drawdown_series
    })


# identify buy signal
df_daily['signal'] = np.where(df_daily['High'] > df_goog[long_ma], 1, 0)

# identify sell signal
df_daily['signal'] = np.where(df_goog[short_ma] < df_goog[long_ma], -1, df_goog['signal'])
df_daily.dropna(inplace=True)