import datetime
import mplfinance as mpf
import pandas as pd

data = pd.read_csv('../data/daily_data.csv')
data = data.set_index(pd.DatetimeIndex(data['Datetime']))
data = data['2024-10-01 10:00':]

print(data.index.dtype)

today = '2024-10-10 10:00:00'
yesterday = '2024-10-09 10:00:00'


data['previous_close'] = data['Close'].shift(1)
print(data)

filtered = data[yesterday:today].copy()
print(filtered)

filtered['percent'] = filtered['Open'] / filtered['previous_close']
print(filtered['percent'])

gap_ups = filtered[filtered['percent'] > 1.03]
print(gap_ups)

gap_downs = filtered[filtered['percent'] < 0.98]
print(gap_downs)

# mpf.plot(data)
# mpf.plot(data, type='renko')
# mpf.plot(data, type='line', volume=True)
# mpf.plot(data, type='pnf')

mpf.plot(data, type='candle', mav=(10, 20))

pd.set_option('display.max_rows', None)


# How did stocks that gapped down perform the first 15 minutes, 30 minutes, 1 hour of trading?
filtered = data['2024-10-09 13:30':'2024-10-09 13:45'].copy()
print(filtered)

filtered = data['2024-10-09 13:30':'2024-10-09 14:00'].copy()
print(filtered)

filtered = data['2024-10-09 13:30':'2024-10-09 14:30'].copy()
print(filtered)
