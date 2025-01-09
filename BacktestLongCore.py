from matplotlib import pyplot as plt

import BacktestCore as btc
import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from ta import trend, utils, volume, volatility, momentum
from PatternPy.tradingpatterns import tradingpatterns

import warnings
warnings.filterwarnings('ignore')

class BacktestLongCore:
    def __init__(self, backtest_core: btc):
        self.backtestCore = backtest_core
        self.position = 0
        self.trades = 0

    def go_long(self, date_index, amount='all'):
        self.backtestCore.buy(date_index=date_index, amount=amount)
        self.position = 1
        self.trades += 1

    def go_short(self, date_index, units='all'):
        self.backtestCore.sell(date_index=date_index, units=units)
        self.position = 0
        self.trades += 1

    def run_strategy(self, strategy):
        for index, row in self.backtestCore.data.iterrows():
            if strategy(row['Close']) == 1 and self.position != 1:
                self.go_long(index)
            elif strategy(row['Close']) == 0 and self.position != 0:
                self.go_short(index)

        print()

    def run_rolling_window_strategy(self, window_size=30):
        row_count = self.backtestCore.data.shape[0]

        df = self.backtestCore.data

        # Predicting Future Returns
        df['return'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        cols = []
        for lag in range(1, window_size + 1):
            col = f'lag_{lag}'
            # Takes the price column and shifts it by lag.
            df[col] = df['return'].shift(lag)
            cols.append(col)
        df.dropna(inplace=True)

        #return_filtered = df['return'].apply(lambda x: 0 if -0.05 <= x <= 0.05 else x)
        return_filtered = df['return']
        df['signal'] = np.sign(return_filtered)
        df['prediction'] = 0

        df['signal'].value_counts().plot(kind='barh')
        plt.show()

        for i in range(len(df) - window_size):
            df_train = df[i:i + window_size]
            df_test = df[i + window_size:i + window_size + 1]

            reg = np.linalg.lstsq(df_train[cols], df_train['signal'], rcond=None)[0]

            # Calculates the prediction values as the dot product.
            pred = np.sign(np.dot(df_test[cols], reg))

            # print("QUAAACK!")
            # print(f"LOC: {df_test.shape}")

            df.loc[df_test.index, 'prediction'] = pred

            if pred == 1 and self.position == 0:
                self.go_long(date_index=df_test.index[0])
            elif pred == -1 and self.position == 1:
                self.go_short(date_index=df_test.index[0])

        df['prediction'].value_counts().plot(kind='barh')
        plt.show()

        # # # Plots the price and prediction columns.
        # # df[['return', 'prediction']].plot(figsize=(10, 6))
        # # plt.show()
        #
        # # Multiplies the prediction values (positionings) by the market returns.
        # df['strategy'] = df['prediction'] * df['return']
        #
        # # Calculates the gross performance of the base instrument and the strategy.
        # df[['return', 'strategy']][-window_size:].sum().apply(np.exp)
        #
        # # Plots the gross performance of the base instrument and the strategy over time
        # # (in-sample, no transaction costs).
        # df[['return', 'strategy']][-window_size:].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
        # plt.show()
        #
        # print(df['return'].describe())

    def run_rolling_window_strategy2(self, window_size=4):
        df = self.backtestCore.data

        # Predicting Future Returns
        df['return'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)

        df['prediction'] = 0

        for i in range(len(df) - window_size):
            df_train = df[i:i + window_size]
            df_train = tradingpatterns.detect_head_shoulder(df_train)
            df_train.loc[:, 'Head and Shoulder'] = df_train.loc[:, 'head_shoulder_pattern'].astype('object') == 'Head and Shoulder'
            df_train.loc[:, 'Inverse Head and Shoulder'] = df_train.loc[:, 'head_shoulder_pattern'].astype('object') == 'Inverse Head and Shoulder'

            if df_train['Head and Shoulder'].iloc[-2]:
                pred = -1
            elif df_train['Inverse Head and Shoulder'].iloc[-2]:
                pred = 1
            else:
                pred = 0

            df.loc[df_train.index[-1], 'prediction'] = pred

            if pred == 1 and self.position == 0:
                self.go_long(date_index=df_train.index[-1])
            elif pred == -1 and self.position == 1:
                self.go_short(date_index=df_train.index[-1])

        df['prediction'].value_counts().plot(kind='barh')
        plt.show()

def test_strategy(x):
    return 1 if x < 102060 else 0


def threshold_strategy(df, threshold=0):
    fig, ax = plt.subplots(sharex='col')

    ax.plot(df['Close'])

    for w in range(1, 6):
        #x = volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=w*5)
        sma = df['Close'].rolling(w*5).mean()
        ax.plot(sma, label=f'SMA{w*5}')

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('data/daily_data.csv')
    data = data.set_index('Datetime')
    data = data['2024-12-01':]
    print("Start...")
    back_tester = btc.BacktestCore(data=data, init_amount=10000, ptc=0.0, verbose=True)

    back_tester_long = BacktestLongCore(backtest_core=back_tester)
    #threshold_strategy(data)
    back_tester_long.run_rolling_window_strategy2()
    back_tester_long.backtestCore.print_asset_info()
