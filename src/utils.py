import os
import pandas as pd

from dbg import log
from config import Config
cfg = Config()


def loadRawStockData(ticker, interval):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        if interval == '24H':
            df = pd.read_csv(f'{cfg.DATA_DIR_RAW_24H}/{ticker}.csv', parse_dates=['Date']).set_index('Date')
        elif interval == '5M':
            df = pd.read_csv(f'{cfg.DATA_DIR_RAW_5M}/{ticker}.csv', parse_dates=['Datetime']).set_index('Datetime')
        else:
            log(f'Unrecognised interval {interval}. (5M or 24H)')
            raise KeyError
    except FileNotFoundError as ex:
        print(ex)
    else:
        return df


def getValidTickers(interval='24H'):
    if interval == '24H':
        inDir = cfg.DATA_DIR_RAW_24H
    elif interval == '5M':
        inDir = cfg.DATA_DIR_RAW_5M
    else:
        log('Interval not recognised in getValidTickers')
        raise KeyError

    return [ticker.split('.')[0] for ticker in os.listdir(inDir)]


def loadAllRawStockData(interval):
    tickers = getValidTickers(interval)
    return {ticker : loadRawStockData(ticker, interval) for ticker in tickers}
