# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ta import momentum, volume, volatility, trend

#tc = 0.001

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')
data = data['2024-11-30':]
print(data.shape)
print(data.index[0])
print(data.index[-1])
data['return'] = np.log(data['Close'] / data['Close'].shift(1))
data['direction'] = np.where(data['return'] > 0, 1, -1)


cols = []

print(data.round(4).tail())

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from keras import models, layers, optimizers, callbacks
tf.compat.v1.enable_eager_execution()

optimizer = optimizers.Adam(learning_rate=0.0001)


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)


set_seeds()


# Adding Different Types of Features
print('Calculate technical indicators...')
lag = 1

data_shift = data.shift(1)
data['rsi'] = momentum.rsi(data_shift['Close'])
data['tsi'] = momentum.tsi(data_shift['Close'])

data['fi'] = volume.force_index(close=data_shift['Close'], volume=data_shift['Volume'])
data['mfi'] = volume.money_flow_index(close=data_shift['Close'], low=data_shift['Low'], high=data_shift['High'],
                                      volume=data_shift['Volume'])
data['eom'] = volume.ease_of_movement(low=data_shift['Low'], high=data_shift['High'], volume=data_shift['Volume'])

data['atr'] = volatility.average_true_range(close=data_shift['Close'], low=data_shift['Low'], high=data_shift['High'])
data['bbh'] = volatility.bollinger_hband(close=data_shift['Close'])
data['bbl'] = volatility.bollinger_lband(close=data_shift['Close'])
data['bbm'] = volatility.bollinger_mavg(close=data_shift['Close'])

data['macd'] = trend.macd(close=data_shift['Close'])
data['sma'] = trend.sma_indicator(close=data_shift['Close'])
data['ema'] = trend.ema_indicator(close=data_shift['Close'])

data['adx'] = trend.adx(close=data_shift['Close'], low=data_shift['Low'], high=data_shift['High'])
data['lag1'] = data['Close'].shift(1)
data['lag2'] = data['Close'].shift(2)
data['lag3'] = data['Close'].shift(3)
data['lag4'] = data['Close'].shift(4)
data['lag5'] = data['Close'].shift(5)

data.dropna(inplace=True)

cols.extend(['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'rsi', 'tsi', 'fi', 'mfi', 'eom', 'atr', 'bbh', 'bbl', 'bbm', 'macd', 'sma', 'ema', 'adx', 'direction'])
#cols.extend(['rsi', 'tsi', 'fi', 'mfi', 'bbh', 'bbl', 'bbm', 'macd', 'ema'])

# Build the LSTM model
# see: https://www.nature.com/articles/s41599-024-02807-x
# https://github.com/AmitOmjeeSharma/Stock-Market-Predication/blob/main/Stock_Market_Predication.ipynb
def rolling_window_forecast(df, window_size):
    predictions = []
    actuals = []
    dates = []

    for i in range(len(df) - window_size):
        train_df = df[i:i + window_size]
        test_df = df[i + window_size:i + window_size + 1]

        # Prepare training data
        X_train = train_df.drop(['direction'], axis=1)
        y_train = train_df['direction']

        # Prepare test data
        X_test = test_df.drop(['direction'], axis=1)
        y_test = test_df['direction']
        dates.append(test_df.index.values[0])

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        predictions.append(y_pred[0])
        actuals.append(y_test.values[0])

        # Debugging print statements
        print(f"Window {i + 1}/{len(df) - window_size}")
        print(f"Train period: {train_df.index.values[0]} to {train_df.index.values[-1]}")
        print(f"Test date: {test_df.index.values[0]}")
        print(f"Prediction: {y_pred[0]}, Actual: {y_test.values[0]}")

    return predictions, actuals, dates


window_size = 30
predictions, actuals, dates = rolling_window_forecast(data[cols], window_size)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluate the model's performance
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"MAE: {mae}, RMSE: {rmse}, R-squared: {r2}")

pred = np.where(np.array(predictions) > 0, 1, -1)
print(f"Correct prediction rate: {np.sum(np.array(actuals) == pred) / len(pred) * 100}%")

data['prediction'] = np.concatenate((np.zeros(window_size), pred))
data['prediction'].value_counts()

data['strategy'] = (data['prediction'] * data['return'])

data[['return', 'strategy']].sum().apply(np.exp)
data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()


