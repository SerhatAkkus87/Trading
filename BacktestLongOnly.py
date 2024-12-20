#
# Python Script with Long Only Class
# for Event-based Backtesting
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
from BacktestBase import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ta import momentum, volume, volatility, trend
import yfinance as yf


class BacktestLongOnly(BacktestBase):

    def run_sma_strategy(self, SMA1, SMA2):
        ''' Backtesting a SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
            shorter and longer term simple moving average (in days)
        '''
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
            if self.position == 0:
                if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            elif self.position == 1:
                if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)

    def run_momentum_strategy(self, momentum):
        ''' Backtesting a momentum-based strategy.

        Parameters
        ==========
        momentum: int
            number of days for mean return calculation
        '''
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
            if self.position == 0:
                if self.data['momentum'].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            elif self.position == 1:
                if self.data['momentum'].iloc[bar] < 0:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)

    def run_mean_reversion_strategy(self, SMA, threshold):
        ''' Backtesting a mean reversion-based strategy.

        Parameters
        ==========
        SMA: int
            simple moving average in days
        threshold: float
            absolute value for deviation-based signal relative to SMA
        '''
        msg = f'\n\nRunning mean reversion strategy | '
        msg += f'SMA={SMA} & thr={threshold}'
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        self.data['SMA'] = self.data['price'].rolling(SMA).mean()

        for bar in range(SMA, len(self.data)):
            if self.position == 0:
                if (self.data['price'].iloc[bar] <
                        self.data['SMA'].iloc[bar] - threshold):
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1
            elif self.position == 1:
                if self.data['price'].iloc[bar] >= self.data['SMA'].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0
        self.close_out(bar)

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        ''' Prepares the lagged data for the regression and prediction steps.
        '''
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data['return'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        ''' Implements the regression step.
        '''
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                              np.sign(self.lagged_data['return']),
                              rcond=None)[0]
        self.reg = reg

    def run_strategy_lr(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
        '''
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'] * \
                                   self.results['return']
        # determine when a trade takes place
        tradesX = self.results['prediction'].diff().fillna(0) != 0

        # subtract transaction costs from return when trade takes place
        self.results.loc[tradesX, 'strategy'] -= self.ptc

        self.results['creturns'] = self.amount * \
                                   self.results['return'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * \
                                    self.results['strategy'].cumsum().apply(np.exp)

        # gross performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]


        msg = f'\n\nRunning linear regression strategy | '
        msg += f'\nfixed costs {self.ftc} | '
        msg += f'proportional costs {self.ptc}'
        print(msg)
        print('=' * 55)
        self.position = 0
        self.trades = 0
        self.amount = self.initial_amount

        t = self.results.loc[tradesX, 'prediction']

        print(f"t: {t}")
        for bar in range(0, len(t)):
            if t.iloc[bar] > 0:
                self.place_buy_order(bar, amount=self.amount)
                self.position = 1
            elif t.iloc[bar] < 0:
                self.place_sell_order(bar, units=self.units)
                self.position = 0
        self.close_out(bar)

        return round(aperf, 2), round(operf, 2)


if __name__ == '__main__':
    def run_strategies():
        # lobt.run_sma_strategy(42, 252)
        # lobt.run_momentum_strategy(60)
        # lobt.run_mean_reversion_strategy(50, 5)
        lobt.run_strategy_lr(start_in='2018-12-15', end_in='2022-01-15', start_out='2022-01-16', end_out='2024-12-15', lags=200)


    # data = pd.read_csv('daily_data1.csv')
    # data = data.set_index('Datetime')
    # data = data['2024-10-30':]

    start_date = '2018-12-15'
    end_date = '2024-12-15'
    data = yf.download(tickers=['BTC-USD'], start=start_date, end=end_date, interval='1d')
    data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    lobt = BacktestLongOnly(symbol='Close', start=data.index[0], end=data.index[-1], amount=100000,
                            verbose=False, raw_data=data, ptc=0.001)

    run_strategies()

    # transaction costs: 10 USD fix, 1% variable
    # lobt = BacktestLongOnly('Close', data.index[0], data.index[-1],
    #                         10000, ftc=10.0, ptc=0.01, verbose=False, raw_data=data)
    # run_strategies()


