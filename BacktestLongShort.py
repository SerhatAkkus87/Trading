#
# Python Script with Long Short Class
# for Event-Based Backtesting
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ta import momentum, volume, volatility, trend
import yfinance as yf

from BacktestBase import *


class BacktestLongShort(BacktestBase):

    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.place_buy_order(bar, units=-self.units)
        if units:
            self.place_buy_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_buy_order(bar, amount=amount)

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.place_sell_order(bar, units=self.units)
        if units:
            self.place_sell_order(bar, units=units)
        elif amount:
            if amount == 'all':
                amount = self.amount
            self.place_sell_order(bar, amount=amount)

    def run_sma_strategy(self, SMA1, SMA2):
        msg = f'\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['SMA1'] = self.data['price'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position in [0, -1]:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.go_long(bar, amount='all')
                    self.position = 1  # long position
            if self.position in [0, 1]:
                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.go_short(bar, amount='all')
                    self.position = -1  # short position
        self.close_out(bar)

    def run_momentum_strategy(self, momentum):
        msg = f'\n\nRunning momentum strategy | {momentum} days'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data['momentum'] = self.data['return'].rolling(momentum).mean()
        for bar in range(momentum, len(self.data)):
            if self.position in [0, -1]:
                if self.data['momentum'].iloc[bar] > 0:
                    self.go_long(bar, amount='all')
                    self.position = 1  # long position
            if self.position in [0, 1]:
                if self.data['momentum'].iloc[bar] <= 0:
                    self.go_short(bar, amount='all')
                    self.position = -1  # short position
        self.close_out(bar)

    def run_mean_reversion_strategy(self, SMA, threshold):
        msg = f'\n\nRunning mean reversion strategy | '
        msg += f'SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital

        self.data['SMA'] = self.data['price'].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if (self.data['price'].iloc[bar] <
                        self.data['SMA'].iloc[bar] - threshold):
                    self.go_long(bar, amount=self.initial_amount)
                    self.position = 1
                elif (self.data['price'].iloc[bar] >
                      self.data['SMA'].iloc[bar] + threshold):
                    self.go_short(bar, amount=self.initial_amount)
                    self.position = -1
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
            elif self.position == -1:
                if self.data['price'].iloc[bar] <= self.data['SMA'].iloc[bar]:
                    self.place_buy_order(bar, units=-self.units)
                    self.position = 0
        self.close_out(bar)

    def rolling_window_forecast(self, cols, window_size, threshold=0.5):
        msg = f'\n\nRunning rolling window forecast | '
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)

        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital

        df = self.data[cols]

        self.data['signal'] = np.zeros(shape=df['direction'].shape)

        for i in range(len(df) - window_size):
            train_df = df[i:i + window_size]
            test_df = df[i + window_size:i + window_size + 1]

            # Prepare training data
            X_train = train_df.drop(['direction'], axis=1)
            y_train = train_df['direction']

            # Prepare test data
            X_test = test_df.drop(['direction'], axis=1)
            y_test = test_df['direction']

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make prediction
            y_pred = model.predict(X_test_scaled)[0]
            # pred = 1 if y_pred > 0 else -1
            pred = 1 if y_pred > threshold else -1 if y_pred < -threshold else 0

            self.data.loc[self.data.index[i + window_size], 'signal'] = pred

            if self.position in [0, -1] and pred == 1:
                self.go_long(i, amount='all')
                self.position = 1  # long position
            if self.position in [0, 1] and pred == -1:
                self.go_short(i, amount='all')
                self.position = -1  # short position

            # Debugging print statements
            # print(f"Window {i + 1}/{len(df) - window_size}")
            # print(f"Train period: {train_df.index.values[0]} to {train_df.index.values[-1]}")
            # print(f"Test date: {test_df.index.values[0]}")
            # print(f"Prediction: {pred}, Actual: {y_test.values[0]}")
        self.close_out(i)


    def preprocess_data(self, window_size):
        data_shift = self.data.shift(1)
        print(self.data.head())
        self.data['rsi'] = momentum.rsi(close=data_shift['price'], window=window_size)
        self.data['tsi'] = momentum.tsi(close=data_shift['price'])

        self.data['fi'] = volume.force_index(close=data_shift['price'], volume=data_shift['Volume'], window=window_size)
        self.data['mfi'] = volume.money_flow_index(close=data_shift['price'], low=data_shift['Low'], high=data_shift['High'],
                                              volume=data_shift['Volume'], window=window_size)
        # self.data['eom'] = volume.ease_of_movement(low=data_shift['Low'], high=data_shift['High'],
        #                                       volume=data_shift['Volume'], window=window_size)
        #
        # self.data['atr'] = volatility.average_true_range(close=data_shift['price'], low=data_shift['Low'],
        #                                             high=data_shift['High'], window=window_size)
        # self.data['bbh'] = volatility.bollinger_hband(close=data_shift['price'], window=window_size)
        # self.data['bbl'] = volatility.bollinger_lband(close=data_shift['price'], window=window_size)
        # self.data['bbm'] = volatility.bollinger_mavg(close=data_shift['price'], window=window_size)

        self.data['macd'] = trend.macd(close=data_shift['price'])
        self.data['sma'] = trend.sma_indicator(close=data_shift['price'], window=window_size)
        self.data['ema'] = trend.ema_indicator(close=data_shift['price'], window=window_size)

        # self.data['adx'] = trend.adx(close=data_shift['price'], low=data_shift['Low'], high=data_shift['High'], window=window_size)
        # self.data['lag1'] = self.data['price'].shift(1)
        # self.data['lag2'] = self.data['price'].shift(2)
        # self.data['lag3'] = self.data['price'].shift(3)
        # self.data['lag4'] = self.data['price'].shift(4)
        # self.data['lag5'] = self.data['price'].shift(5)

        self.data.dropna(inplace=True)

        # return ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'rsi', 'tsi', 'fi', 'mfi', 'eom', 'atr', 'bbh', 'bbl', 'bbm',
        #         'macd', 'sma', 'ema', 'adx', 'direction']

        return ['rsi', 'tsi', 'fi', 'mfi', 'macd', 'sma', 'ema', 'direction']


def update_zeros(df, col):
    # Replace 0s with NaN
    df[col].replace(0, pd.NA, inplace=True)

    # Forward fill the NaNs with the last non-zero value
    df[col].ffill(inplace=True)

    # Replace NaNs with the last non-zero value
    df[col].fillna(method='bfill', inplace=True)

    return df[col].tolist()


if __name__ == '__main__':
    # data = pd.read_csv('daily_data1.csv')
    # data = data.set_index('Datetime')
    # data = data['2024-10-30':]

    start_date = '2023-12-15'
    end_date = '2024-12-15'
    data = yf.download(tickers=['BTC-USD'], start=start_date, end=end_date, interval='1d')
    data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    print(data.head())
    print(data.columns)

    def run_strategies():
        # lsbt.run_sma_strategy(42, 252)
        # lsbt.run_momentum_strategy(60)
        # lsbt.run_mean_reversion_strategy(50, 5)

        window_size = 14
        threshold = 0.8
        cols = lsbt.preprocess_data(window_size)
        lsbt.rolling_window_forecast(cols, window_size, threshold=threshold)

        #data['signal'].value_counts()
        lsbt.data['signal'] = update_zeros(df=lsbt.data, col='signal')

        lsbt.data['strategy'] = (lsbt.data['signal'] * lsbt.data['return'])

        lsbt.data[['return', 'strategy']].sum().apply(np.exp)
        lsbt.data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
        plt.show()


    lsbt = BacktestLongShort(symbol='Close', start=data.index[0], end=data.index[-1], amount=1000000,
                             verbose=True, raw_data=data, ptc=0.001)
    run_strategies()

    # transaction costs: 10 USD fix, 1% variable
    # lsbt = BacktestLongShort(symbol='Close', start=data.index[0], end=data.index[-1],
    #                          amount=10000, ftc=10.0, ptc=0.01, verbose=False, raw_data=data)
    # run_strategies()

#     start_date = '2023-12-15'
#     end_date = '2024-08-30'

# 0.5, 14
# Final balance   [$] 1619798.86
# Net Performance [%] 61.98
# Trades Executed [#] 54

# 0.7, 14
# Final balance   [$] 2322678.71
# Net Performance [%] 132.27
# Trades Executed [#] 24

# 0.9, 14
# Final balance   [$] 1937914.31
# Net Performance [%] 93.79
# Trades Executed [#] 6

# 0.95, 14
# Final balance   [$] 1356096.77
# Net Performance [%] 35.61
# Trades Executed [#] 2


# 0.95, 30
# Final balance   [$] 1401341.37
# Net Performance [%] 40.13
# Trades Executed [#] 4

# 0.9, 30
# Final balance   [$] 1647969.95
# Net Performance [%] 64.80
# Trades Executed [#] 8

# 0.7, 30
# Final balance   [$] 1359190.39
# Net Performance [%] 35.92
# Trades Executed [#] 16

# 0.5, 30
# Final balance   [$] 1772419.55
# Net Performance [%] 77.24
# Trades Executed [#] 36


# 6 years
# 0.7, 14
# Final balance   [$] 135172936.13
# Net Performance [%] 13417.29
# Trades Executed [#] 296

# 0.8, 14
# Final balance   [$] 380719283.61
# Net Performance [%] 37971.93
# Trades Executed [#] 152

# 0.9, 14
# Final balance   [$] 158160316.36
# Net Performance [%] 15716.03
# Trades Executed [#] 74
