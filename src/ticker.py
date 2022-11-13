import pandas as pd
import utils

from dbg import log
from config import Config, Interval
cfg = Config()

class Ticker:
    def __init__(self, ticker, interval):
        self.id = ticker
        self.interval = interval

        if interval == Interval.DAY:
            self.path = f'{cfg.DATA_DIR_24_HOUR}/{ticker}.csv'
        elif interval == Interval.FIVE_MINUTE:
            self.path = f'{cfg.DATA_DIR_5_MINUTE}/{ticker}.csv'
        else:
            log(f'Interval not recognised {interval}')
            raise KeyError

        self.data = None

    def load(self):
        self.data = utils.loadRawStockData(self.id, self.interval)

    def slice(self, startTime, endTime):
        if isinstance(startTime, str):
            startTime = pd.to_datetime(startTime)
        if isinstance(endTime, str):
            endTime = pd.to_datetime(endTime)

        self.data = self.data[(self.data.index >= startTime) &
                              (self.data.index <= endTime)]

        if len(self.data) == 0:
            log(f'WARNING: No data exists in range {startTime} - {endTime} for ticker {self.id}')


def analyseTickerPerformance(ticker, interval, sTime, eTime):
    df = utils.loadRawStockData(ticker, interval)

    df = df[(df.index >= sTime) & (df.index <= eTime)]
    df = utils.addBaseIndicatorsToDf(df)

    # Calc SD
    sd = df['interval_return'].std()

    # Calc average return
    mean = df['interval_return'].mean()

    return {'Ticker' : ticker,
            'Mean Return' : mean,
            'Standard Deviation' : sd}