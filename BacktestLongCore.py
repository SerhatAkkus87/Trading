import BacktestCore as btc
import pandas as pd


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


def test_strategy(x):
    return 1 if x < 102060 else 0


if __name__ == '__main__':
    data = pd.read_csv('daily_data.csv')
    data = data.set_index('Datetime')
    data = data['2024-12-19 13:30:00':'2024-12-19 13:32:00']

    back_tester = btc.BacktestCore(data=data, init_amount=10000, ptc=0.0, verbose=True)

    back_tester_long = BacktestLongCore(backtest_core=back_tester)
    back_tester_long.run_strategy(strategy=test_strategy)
    back_tester_long.backtestCore.print_asset_info()
