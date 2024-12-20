import ccxt
import json
import csv
import pandas as pd
from datetime import datetime, timezone, timedelta
import timeit
import time

import matplotlib.pyplot as plt
import numpy as np
import asyncio

from ccxt.static_dependencies.marshmallow.utils import timestamp
from numpy.matlib import empty


class BitgetTrader:
    def __init__(self, symbol_path_exists=False, path_symbol="symbols.json"):
        self._exchange = self.create_exchange()
        self._pathSymbol = path_symbol

        print(self._exchange)

        if symbol_path_exists:
            self._symbols = self.read_symbols()
            print(f"Symbols loaded from {path_symbol}.")
        else:
            self._symbols = self.fetch_symbols()
            print(f"Symbols fetched from the market directly.")

        pass

    @staticmethod
    def create_exchange():
        with open("configuration.json", "r") as f:
            data = f.read()

        args = json.loads(data)

        return ccxt.bitget({'apiKey': args['APIKEY'],
                            'secret': args['SECRET'],
                            'password': args['PASSWORD'],
                            'options': {
                                "defaultType": "swap",
                                "adjustForTimeDifference": True
                            }})

    def fetch_symbols(self, b_save=True):
        symbols = None
        string = 'BTCUSDT'

        try:
            data = self._exchange.fetch_markets()
            symbols = [pair['id'] for pair in data if string in pair['id']]
            print(symbols)
            if b_save:
                self.save_symbols(symbols)

        except Exception as e:
            print("Exception in fetching market symbols.")

        return symbols

    def save_symbols(self, symbols=None):
        with open(self._pathSymbol, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Symbols"])
            for symbol in symbols:
                writer.writerow([symbol])
        print(f"Symbols has been saved to {self._pathSymbol}")

    def read_symbols(self):
        self._symbols = []
        with open(self._pathSymbol, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                self._symbols.append(row[0])
        print(self._symbols)
        return self._symbols

    @staticmethod
    def timeframe_to_timestamp(tf):
        if tf == '1m':
            return 60000
        elif tf == '15m':
            return 240000
        elif tf == '1h':
            return 3600000
        elif tf == '1d':
            return 86400000
        elif tf == '1w':
            return 86400000 * 7

    def fetch_ohcl(self, symbol='BTCUSDT', timeframe='1d', start=None, end=None, limit=None,
                   file="x.csv"):
        if self._exchange.has['fetchOHLCV']:
            ts = self.timeframe_to_timestamp(timeframe)
            end = end if end is not None else self._exchange.milliseconds()

            total = float(end - start) / ts
            count = 0

            if limit is None:
                l = 100
            else:
                l = limit

            header_exists = False
            from pathlib import Path
            my_file = Path(file)

            if my_file.is_file():
                with open(file, newline='', mode='r') as f:
                    print("HEADER")
                    header_exists = f.readline().__contains__("Datetime,Open,High,Low,Close,Volume")
                    print(header_exists)
                    if header_exists:
                        last_date = f.readlines()[-1].split(',')[0]

            with open(file, newline='', mode='a') as f:
                writer = csv.writer(f)

                if not header_exists:
                    writer.writerow(['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                else:
                    ts = self.timeframe_to_timestamp(tf=timeframe)
                    last_ts = int(time.mktime(datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S").timetuple()) * 1000)
                    start = last_ts + ts

                while start < end:
                    orders = self._exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=start, limit=limit)

                    if len(orders):
                        diff = orders[len(orders) - 1][0] - orders[len(orders) - 2][0]
                        start = orders[len(orders) - 1][0] + diff
                        for row in orders:
                            row[0] = datetime\
                                .fromtimestamp(row[0] / 1000, tz=timezone.utc)\
                                .strftime('%Y-%m-%d %H:%M:%S')
                            writer.writerow(row)

                        if count + l > total:
                            count = total
                        else:
                            count = count + l

                        percent = round(count / total * 100, 2)
                        print(f'{percent}% processed...')
                    else:
                        break

    @staticmethod
    def to_utc(time_stamp):
        return datetime.fromtimestamp(time_stamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def preprocess_df(df):
        if df is None:
            return df

        df = df.rename(columns={0: 'timestamp', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'})
        df['Datetime'] = df['timestamp'].apply(lambda date: date.to_utc())

        return df


bgTrader = BitgetTrader()
now = datetime.now(tz=timezone.utc)
since = now - timedelta(weeks=52*5)
# since = now - timedelta(days=1)
start_timestamp = int(since.timestamp() * 1000)

print(f'since.timestamp(): {int(since.timestamp())}')
# record start time
t_0 = timeit.default_timer()

print('Processing...')
try:
    bgTrader.fetch_ohcl(timeframe='1m',
                        start=start_timestamp,
                        file="daily_data.csv")
except Exception as e:
    print(e)

print('End of process.')

# record end time
t_1 = timeit.default_timer()

# print(ohlcv2)

# calculate elapsed time and print
elapsed_time = round((t_1 - t_0), 3)
print(f"Elapsed time: {elapsed_time} s")


df = pd.read_csv('daily_data.csv')
print(df.columns)
print(df.index)
print(df.tail())
