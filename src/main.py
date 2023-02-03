import pandas as pd
from tqdm import tqdm

from financeobjects import Ticker

import utils
from config import Config, Interval
cfg = Config()


def main():
    portfolio = ['CALX', 'NOVT', 'RGEN', 'LLY',
                 'AMD', 'NFLX', 'COST', 'BJ', 'WING',
                 'MSCI', 'CBRE']
    interval = Interval.DAY

    allTickers = utils.getValidTickers(interval)

    startTime = '2021-01-01'
    endTime   = '2022-11-05'

    # ------------------------------------

    sTime = pd.to_datetime(startTime)
    eTime = pd.to_datetime(endTime)

    for ticker in tqdm(allTickers):
        t = Ticker(ticker, Interval.DAY)
        t.load()


if __name__ == '__main__':
    main()

