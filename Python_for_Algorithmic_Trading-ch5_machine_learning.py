# Chapter 5. Predicting Market Movements with Machine Learning

import pandas as pd
import numpy as np
import SMAVectorBacktester as SMA
import matplotlib.pyplot as plt

data = pd.read_csv('data/daily_data.csv')
data = data.set_index('Datetime')

data = data[['Close']]

data = pd.DataFrame(data['Close'])
data.rename(columns={'Close': 'price'}, inplace=True)
data['return'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)

# Using Logistic Regression to Predict Market Direction
lags = 2

cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data['price'].shift(lag)
    cols.append(col)

data.dropna(inplace=True)
print(f"cols: {cols}")
print(f"data cols: {data.columns}")


from sklearn.metrics import accuracy_score
from sklearn import linear_model

# Instantiates the model object using a C value that gives less weight to the regularization term
lm = linear_model.LogisticRegression(C=1e7, solver='lbfgs', multi_class='auto', max_iter=1000)

# Fits the model based on the sign of the returns to be predicted.
lm.fit(data[cols], np.sign(data['return']))

# Generates a new column in the DataFrame object and writes the prediction values to it.
data['prediction'] = lm.predict(data[cols])

# Shows the number of the resulting long and short positions, respectively.
print(f"Number of the resulting long and short positions:\n{data['prediction'].value_counts()}\n")

# Calculates the number of correct and wrong predictions.
hits = np.sign(data['return'].iloc[lags:] * data['prediction'].iloc[lags:]).value_counts()
print(f"number of correct and wrong predictions:\n{hits}\n")

print(f"accuracy_score: {accuracy_score(data['prediction'], np.sign(data['return']))}\n")

# However, the gross performance of the strategy…
data['strategy'] = data['prediction'] * data['return']
print(data[['return', 'strategy']].sum().apply(np.exp))

# …is much higher when compared with the passive benchmark investment.
data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

