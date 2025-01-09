from PatternPy.tradingpatterns import tradingpatterns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Have price data (OCHLV dataframe)
data = pd.read_csv('data/daily_data_2.csv')
data = data.set_index('Datetime')
#data = data['2023-12-13':'2023-12-15']

# Apply pattern indicator screener
window=3
df = tradingpatterns.detect_head_shoulder(data, window=window)
df = tradingpatterns.detect_double_top_bottom(df, window=window)
df = tradingpatterns.detect_multiple_tops_bottoms(df, window=window)
df = tradingpatterns.detect_triangle_pattern(df, window=window)
df = tradingpatterns.detect_trendline(df, window=50)
df = tradingpatterns.detect_wedge(df, window=window)

# New column `head_shoulder_pattern` is created with entries containing either:
# NaN, 'Head and Shoulder' or 'Inverse Head and Shoulder'
print(df)

# Generate buy/sell signals
data['Head and Shoulder'] = df['head_shoulder_pattern'] == 'Head and Shoulder'
data['Inverse Head and Shoulder'] = df['head_shoulder_pattern'] == 'Inverse Head and Shoulder'
data['Multiple Top'] = df['multiple_top_bottom_pattern'] == 'Multiple Top'
data['Ascending Triangle'] = df['triangle_pattern'] == 'Ascending Triangle'
data['Wedge Up'] = df['wedge_pattern'] == 'Wedge Up'

data['Double Top'] = df['double_pattern'] == 'Double Top'
data['Double Bottom'] = df['double_pattern'] == 'Double Bottom'
data['Multiple Bottom'] = df['multiple_top_bottom_pattern'] == 'Multiple Bottom'
data['Descending Triangle'] = df['triangle_pattern'] == 'Descending Triangle'
data['Wedge Down'] = df['wedge_pattern'] == 'Wedge Down'


# Plot the stock market graph
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(sharex=True, nrows=6)

ax1.plot(data['Close'], label='Close Price', color='blue')
ax1.plot(data[data['Double Top']].index, data['Close'][data['Double Top']], '^', markersize=10, color='g', label='Buy Signal Double Top')
ax1.plot(data[data['Double Bottom']].index, data['Close'][data['Double Bottom']], 'v', markersize=10, color='r', label='Sell Signal Double Bottom')
ax1.set_title('Double Top/Bottom')

ax2.plot(data['Close'], label='Close Price', color='blue')
ax2.plot(data[data['Head and Shoulder']].index, data['Close'][data['Head and Shoulder']], '^', markersize=10, color='g', label='Buy Signal')
ax2.plot(data[data['Inverse Head and Shoulder']].index, data['Close'][data['Inverse Head and Shoulder']], 'v', markersize=10, color='r', label='Sell Signal')
ax2.set_title('Head and Shoulder')

ax3.plot(data['Close'], label='Close Price', color='blue')
ax3.plot(data[data['Multiple Top']].index, data['Close'][data['Multiple Top']], '^', markersize=10, color='g', label='Buy Signal Multiple Top')
ax3.plot(data[data['Multiple Bottom']].index, data['Close'][data['Multiple Bottom']], 'v', markersize=10, color='r', label='Sell Signal Multiple Bottom')
ax3.set_title('Multiple Top/Bottom')

ax4.plot(data['Close'], label='Close Price', color='blue')
ax4.plot(data[data['Ascending Triangle']].index, data['Close'][data['Ascending Triangle']], '^', markersize=10, color='g', label='Buy Signal Triangle')
ax4.plot(data[data['Descending Triangle']].index, data['Close'][data['Descending Triangle']], 'v', markersize=10, color='r', label='Sell Signal Triangle')
ax4.set_title('Ascending/Descending Triangle')

ax5.plot(data['Close'], label='Close Price', color='blue')
ax5.plot(data['support'], label='Support', color='red')
ax5.plot(data['resistance'], label='Resistance', color='green')
ax5.set_title('Trendlines')

ax6.plot(data['Close'], label='Close Price', color='blue')
ax6.plot(data[data['Wedge Up']].index, data['Close'][data['Wedge Up']], '^', markersize=10, color='g', label='Buy Signal Wedge Up')
ax6.plot(data[data['Wedge Down']].index, data['Close'][data['Wedge Down']], 'v', markersize=10, color='r', label='Sell Signal Wedge Down')
ax6.set_title('Wedge Up/Down')

# Add labels and legend
#plt.title('Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.show()




# import stumpy
#
# # Compute the matrix profile
# window_size = 14  # Adjust this based on your data
# matrix_profile = stumpy.stump(data['Close'], m=window_size)
#
# print(matrix_profile)

# # Identify patterns
# patterns = matrix_profile[:, 0] < np.percentile(matrix_profile[:, 0], 5)  # Adjust threshold as needed
#
# plt.figure(figsize=(12, 6))
# plt.plot(data['Close'], label='Close Price', color='blue')
# plt.scatter(data.index[patterns], data['Close'][patterns], color='red', label='Detected Patterns')
# plt.title('Stock Price with Detected Patterns')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
