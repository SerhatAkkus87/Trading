import pandas as pd


class BacktestCore(object):

    def __init__(self, data, init_amount, ptc, amount_symbol='$', unit_symbol='BTC', verbose=False):
        self.str_format_u = '{0:,.4f}'
        self.str_format_a = '{0:,.2f}'

        self.init_amount = init_amount
        self.amount = init_amount
        self.data = data
        self.ptc = ptc
        self.units = 0.0
        self.verbose = verbose
        self.amount_symbol = amount_symbol
        self.unit_symbol = unit_symbol

    def buy(self, date_index, amount='all'):
        current_price = self.data.loc[date_index, 'Close']
        a = self.amount if amount == 'all' else amount
        u = a / current_price

        self.units += u
        self.amount -= a

        if self.verbose:
            self.show_verbose("BUY ", u, a, current_price)

    def sell(self, date_index, units='all'):
        current_price = self.data.loc[date_index, 'Close']
        u = self.units if units == 'all' else units
        a = u * current_price

        self.units -= u
        self.amount += a

        if self.verbose:
            self.show_verbose("SELL", u, a, current_price)

    def show_verbose(self, text, u, a, current_price):
        u = self.str_format_u.format(round(u, 4))
        a = self.str_format_a.format(round(a, 2))
        print(f"{text}\t-> {u} {self.unit_symbol} for {a} {self.amount_symbol}\t\t"
              f"(1 {self.unit_symbol} = {round(current_price)} {self.amount_symbol})")

    def print_asset_info(self):
        u = self.str_format_u.format(round(self.units, 4))
        a = self.str_format_a.format(round(self.amount, 2))

        print(f"Amount:\t{a} {self.amount_symbol}")
        print(f"Units: \t{u} {self.unit_symbol}")
        print()


if __name__ == '__main__':
    data = pd.read_csv('data/daily_data.csv')
    data = data.set_index('Datetime')
    back_tester = BacktestCore(data=data, init_amount=10000, ptc=0.0, verbose=True)

    back_tester.print_asset_info()
    back_tester.buy(date_index='2024-12-19 13:30:00', amount=1000)
    back_tester.sell(date_index='2024-12-19 13:31:00', units=0.001)
    back_tester.buy(date_index='2024-12-19 13:32:00', amount=1000)
    back_tester.print_asset_info()

