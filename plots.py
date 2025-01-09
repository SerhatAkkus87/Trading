from PatternPy.tradingpatterns import tradingpatterns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Have price data (OCHLV dataframe)
data = pd.read_csv('data/daily_data_2.csv')
data = data.set_index('Datetime')
symbol = 'BTC'

# # Line plot of closing prices
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, data['Close'], label=f'{symbol} Closing Price', linewidth=2)
# plt.title(f'{symbol} Closing Prices Over Time')
# plt.xlabel('Date')
#
# # Format the x-axis labels
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
# #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#
# # Rotate and align the x labels
# plt.gcf().autofmt_xdate()
#
# plt.ylabel('Closing Price (USD)')
# plt.legend()
# plt.show()
#
#
# import seaborn as sns
#
# # Seaborn style set
# sns.set(style="whitegrid")
#
# # Distribution of Daily Returns
# plt.figure(figsize=(12, 6))
# sns.histplot(data['Close'].pct_change().dropna(), bins=30, kde=True, color='blue')
# plt.title(f'Distribution of {symbol} Daily Returns')
# plt.xlabel('Daily Return')
# plt.ylabel('Frequency')
# plt.show()
#
import plotly.graph_objects as go

# Candlestick chart
candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                              open=data['Open'],
                                              high=data['High'],
                                              low=data['Low'],
                                              close=data['Close'])])

candlestick.update_layout(title=f'{symbol} Candlestick Chart',
                          xaxis_title='Date',
                          yaxis_title='Stock Price (USD)',
                          xaxis_rangeslider_visible=False)

candlestick.show()


# # Moving Average Plot
# plt.figure(figsize=(12, 6))
# data['Close'].plot(label=f'{symbol} Closing Price', linewidth=2)
# data['Close'].rolling(window=30).mean().plot(label=f'{symbol} 30-Day Avg', linestyle='--', color='orange')
# plt.title(f'{symbol} Closing Prices with 30-Day Moving Average')
# plt.xlabel('Date')
# plt.ylabel('Closing Price (USD)')
# plt.legend()
# plt.show()

# # Volume Plot
# plt.figure(figsize=(12, 6))
# plt.bar(data.index, data['Volume'], color='green', alpha=0.7)
# plt.title(f'{symbol} Trading Volume Over Time')
# plt.xlabel('Date')
# plt.ylabel('Volume')
# plt.show()

# # Correlation Heatmap
# correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title(f'Correlation Heatmap for {symbol} Financial Metrics')
# plt.show()

