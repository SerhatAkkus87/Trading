from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from typing import Union


# Have price data (OCHLV dataframe)
df = pd.read_csv('data/daily_data.csv')
df = df.set_index('Datetime')
df = df['2024-01-01':]
fig, ax = plt.subplots()
df['Close'].plot()
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')

fig.autofmt_xdate()
plt.tight_layout()
plt.show()


ad_fuller_result = adfuller(df['Close'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

eps_diff = np.diff(df['Close'], n=1)
ad_fuller_result = adfuller(eps_diff)

print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')


fig, ax = plt.subplots()

ax.plot(df.index[1:], eps_diff)
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share - diff (USD)')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()



def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    results = []

    for order in order_list:
        try:
            model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df


ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1

order_list = list(product(ps, qs))


train = df['Close'][:-4]
result_df = optimize_ARIMA(train, order_list, d)
print(result_df)


model = SARIMAX(train, order=(2, 1, 0), simple_differencing=False)
model_fit = model.fit(disp=False)

print(model_fit.summary())

model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()


residuals = model_fit.resid
lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(pvalue)


test = df.iloc[-4:]
test['naive_seasonal'] = df['Close'].iloc[df.shape[0]-8-1:df.shape[0]-4-1].values


ARIMA_pred = model_fit.get_prediction(df.shape[0]-4, df.shape[0]-1).predicted_mean
test['ARIMA_pred'] = ARIMA_pred.values
print(test['ARIMA_pred'])
print(ARIMA_pred.values)


fig, ax = plt.subplots()

ax.plot(df[-50:]['Close'])
ax.plot(test['Close'], 'b-', label='actual')
ax.plot(test['naive_seasonal'], 'r:', label='naive seasonal')
ax.plot(test['ARIMA_pred'], 'k--', label='ARIMA(2,1,3)')

ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(df[-50:].shape[0]-8, df[-50:].shape[0]-4, color='#808080', alpha=0.2)

ax.legend(loc=2)

#ax.set_xlim(60, 83)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape_naive_seasonal = mape(test['Close'], test['naive_seasonal'])
mape_ARIMA = mape(test['Close'], test['ARIMA_pred'])

print(mape_naive_seasonal, mape_ARIMA)


fig, ax = plt.subplots()

x = ['naive seasonal', 'ARIMA(3,2,3)']
y = [mape_naive_seasonal, mape_ARIMA]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 15)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha='center')

plt.tight_layout()
plt.show()

print(test)
