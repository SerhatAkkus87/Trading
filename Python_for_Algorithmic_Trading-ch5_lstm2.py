# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from ta import momentum, volume, volatility, trend

cutoff = '2024-11-28'

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')
data = data['2024-11-20':]
print(data.shape)
print(data.index[0])
print(data.index[-1])
data['return'] = np.log(data['Close'] / data['Close'].shift(1))
data['direction'] = np.where(data['return'] > 0, 1, 0)
data['direction'] = data['direction'].shift(1)

cols = ['Open', 'High', 'Low', 'Close', 'Volume']
lags = 5
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data['Close'].shift(lag)
    cols.append(col)
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
#
# data['rsi'] = momentum.rsi(data['Close'])
# data['tsi'] = momentum.tsi(data['Close'])
#
# data['fi'] = volume.force_index(close=data['Close'], volume=data['Volume'])
# data['mfi'] = volume.money_flow_index(close=data['Close'], low=data['Low'], high=data['High'],
#                                       volume=data['Volume'])
# data['eom'] = volume.ease_of_movement(low=data['Low'], high=data['High'], volume=data['Volume'])
#
# data['atr'] = volatility.average_true_range(close=data['Close'], low=data['Low'], high=data['High'])
# data['bbh'] = volatility.bollinger_hband(close=data['Close'])
# data['bbl'] = volatility.bollinger_lband(close=data['Close'])
# data['bbm'] = volatility.bollinger_mavg(close=data['Close'])
#
# data['macd'] = trend.macd(close=data['Close'])
# data['sma'] = trend.sma_indicator(close=data['Close'])
# data['ema'] = trend.ema_indicator(close=data['Close'])
#
# data['adx'] = trend.adx(close=data['Close'], low=data['Low'], high=data['High'])
#data['lag1'] = data_shift['Close']

data.dropna(inplace=True)

#cols.extend(['rsi', 'tsi', 'fi', 'mfi', 'eom', 'atr', 'bbh', 'bbl', 'bbm', 'macd', 'sma', 'ema', 'adx'])
#cols.extend(['rsi', 'tsi', 'fi', 'mfi', 'bbh', 'bbl', 'bbm', 'macd', 'ema'])

print(data.round(4).tail())

earlyStop = callbacks.EarlyStopping(monitor="accuracy", verbose=2, mode='min', patience=5)

# Defines the training and test data sets.
training_data = data[data.index < cutoff].copy()

# Normalizes the features data by Gaussian normalization.
mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data - mu) / std
test_data = data[data.index >= cutoff].copy()
test_data_ = (test_data - mu) / std


set_seeds()

# Build the LSTM model
# see: https://www.nature.com/articles/s41599-024-02807-x
model = models.Sequential()
model.add(layers.Input(shape=(len(cols), 1)))
model.add(layers.LSTM(units=30, activation='relu', return_sequences=True))
model.add(layers.Dropout(rate=0.1))
model.add(layers.LSTM(units=40, activation='relu', return_sequences=True))
model.add(layers.Dropout(rate=0.1))
model.add(layers.LSTM(units=50, activation='relu', return_sequences=True))
model.add(layers.Dropout(rate=0.1))
model.add(layers.LSTM(units=60, activation='relu'))
model.add(layers.Dropout(rate=0.1))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.run_eagerly = True

print("Fit new model with pimped features")
model.fit(training_data_[cols], training_data['direction'], epochs=20, callbacks=[earlyStop])
model.evaluate(training_data_[cols], training_data['direction'])


# # Analyze input features
# import shap
#
# # Load your data (assuming X_train and X_test are your feature matrices)
# # X_train and X_test should be numpy arrays or pandas DataFrames
# explainer = shap.KernelExplainer(model.predict, training_data)
# shap_values = explainer.shap_values(test_data)
#
# shap.summary_plot(shap_values, test_data)


print("Show return / strategy comparision for train set.")
pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)
training_data['prediction'] = np.where(pred > 0, 1, -1)

training_data['strategy'] = (training_data['prediction'] * training_data['return'])
training_data[['return', 'strategy']].sum().apply(np.exp)

training_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

print("Show return / strategy comparision for test set.")
model.evaluate(test_data_[cols], test_data['direction'])

pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)
test_data['prediction'] = np.where(pred > 0, 1, -1)
test_data['prediction'].value_counts()

test_data['strategy'] = (test_data['prediction'] * test_data['return'])

test_data[['return', 'strategy']].sum().apply(np.exp)
test_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()


