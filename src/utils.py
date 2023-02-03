import os
import pandas as pd

from dbg import log
from config import Config, Interval
from preprocess import addBaseIndicatorsToDf
cfg = Config()


def loadStockData(ticker, interval):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        if interval == Interval.DAY:
            df = pd.read_csv(f'{cfg.DATA_DIR_24_HOUR}/{ticker}.csv', index_col=0, parse_dates=['Date'])
        elif interval == Interval.FIVE_MINUTE:
            df = pd.read_csv(f'{cfg.DATA_DIR_5_MINUTE}/{ticker}.csv', index_col=0, parse_dates=['Datetime'])
        else:
            log(f'Unrecognised interval {interval}.')
            raise KeyError
    except FileNotFoundError as ex:
        print(ex)
        return None
    return df


def loadMultipleDFsAndMergeByColumnName(colName, sDate, eDate, interval, tickers):
    mult_df = pd.DataFrame()

    for x in tickers:
        df = loadStockData(x, interval)

        if not df.index.is_unique:
            df = df.loc[~df.index.duplicated(), :]

        if df.index[-1] < eDate or df.index[0] > sDate:
            log(f'WARNING: Ticker {x} is missing stock data in requested range!')
            continue

        df = df[(df.index >= sDate) & (df.index <= eDate)]
        df = addBaseIndicatorsToDf(df)

        mult_df[x] = df[colName]

    return mult_df


def getValidTickers(interval):
    if interval == Interval.DAY:
        inDir = cfg.DATA_DIR_24_HOUR
    elif interval == Interval.FIVE_MINUTE:
        inDir = cfg.DATA_DIR_5_MINUTE
    else:
        log('Interval not recognised in getValidTickers')
        raise KeyError

    return [ticker.split('.')[0] for ticker in os.listdir(inDir)]


def loadAllRawStockData(interval):
    tickers = getValidTickers(interval)
    return {ticker : loadStockData(ticker, interval) for ticker in tickers}
