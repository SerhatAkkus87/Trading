# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt
import yfinance as yf

data = pd.read_csv('data/daily_data_2.csv')
data = data.set_index('Datetime')
print(data.shape)


import ScikitVectorBacktester as SCI

start_in = '2023-12-02'
end_in = '2024-05-30'
start_out = '2024-06-01'
end_out = '2024-11-30'

# data = yf.download(tickers=['BTC-USD'], start=start_in, end=end_out, interval='1d')
# data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
# data = pd.DataFrame(data['Close'])
# print(data.info())


#data[start_out:end_out].plot(title="TEST")
#plt.show()

#print(str(data.index[0][:10]))
#print(str(data.index[-1][:10]))


scibt = SCI.ScikitVectorBacktester('Close', start_in, end_out, 10000, 0.00, 'logistic', raw_data=data.copy())
print(scibt.run_strategy(start_in, end_in, start_out, end_out, lags=15))

scibt.plot_results()
plt.show()
