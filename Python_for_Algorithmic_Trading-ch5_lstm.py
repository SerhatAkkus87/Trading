# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

data = pd.read_csv('daily_data1.csv')
data = data.set_index('Datetime')

data = pd.DataFrame(data['Close'])[int(len(data) * 0.5):]
print(data.shape)
print(data.index[0])
print(data.index[-1])
data.rename(columns={'Close': 'price'}, inplace=True)
data['return'] = np.log(data['price'] / data['price'].shift(1))
data['direction'] = np.where(data['return'] > 0, 1, 0)


lags = 5

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

print(data.round(4).tail())

import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
# Build the LSTM model
model = models.Sequential()
model.add(layers.Input(shape=(lags, 1)))
model.add(layers.LSTM(units=50, return_sequences=True))
model.add(layers.LSTM(units=50))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.run_eagerly = True


cutoff = '2024-09-31'

# Defines the training and test data sets.
training_data = data[data.index < cutoff].copy()

# Normalizes the features data by Gaussian normalization.
mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data - mu) / std
test_data = data[data.index >= cutoff].copy()
test_data_ = (test_data - mu) / std

# Fits the model to the training data set.
print("Fit model with training data...")

earlyStop = callbacks.EarlyStopping(monitor="val_loss", verbose=2, mode='min', patience=3)
model.fit(training_data[cols],
          training_data['direction'],
          epochs=10, batch_size=32, callbacks=[earlyStop])

# Plot
print("Plot accuracy")
res = pd.DataFrame(model.history.history)
res.plot(figsize=(10, 6), style='--')
plt.show()


print("Show eval for train set")
model.evaluate(training_data_[cols], training_data['direction'])

pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)

# Predicts the market direction in-sample.
pred[:30].flatten()

# Transforms the predictions into long-short positions, +1 and -1.
training_data['prediction'] = np.where(pred > 0, 1, -1)

# Calculates the strategy returns given the positions.
training_data['strategy'] = (training_data['prediction'] * training_data['return'])

training_data[['return', 'strategy']].sum().apply(np.exp)

# Plots and compares the strategy performance to the benchmark performance (in-sample).
training_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()


print("Show eval for test set")
# test data evaluation
model.evaluate(test_data_[cols], test_data['direction'])

pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)

test_data['prediction'] = np.where(pred > 0, 1, -1)
test_data['prediction'].value_counts()

test_data['strategy'] = (test_data['prediction'] * test_data['return'])

test_data[['return', 'strategy']].sum().apply(np.exp)
test_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

# Adding Different Types of Features
data['momentum'] = data['return'].rolling(5).mean().shift(1)
data['volatility'] = data['return'].rolling(20).std().shift(1)
data['distance'] = (data['price'] - data['price'].rolling(50).mean()).shift(1)
data.dropna(inplace=True)

cols.extend(['momentum', 'volatility', 'distance'])
print(data.round(4).tail())


training_data = data[data.index < cutoff].copy()

mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data - mu) / std
test_data = data[data.index >= cutoff].copy()
test_data_ = (test_data - mu) / std


set_seeds()

# Build the LSTM model
model = models.Sequential()
model.add(layers.Input(shape=(lags, 1)))
model.add(layers.LSTM(units=50, return_sequences=True))
model.add(layers.LSTM(units=50))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.run_eagerly = True

print("Fit new model with pimped features")
model.fit(training_data_[cols], training_data['direction'], epochs=10, callbacks=[earlyStop])
model.evaluate(training_data_[cols], training_data['direction'])

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


