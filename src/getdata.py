import os
import traceback
import pandas as pd
import yfinance as yf
import warnings
from multiprocessing.pool import ThreadPool
warnings.simplefilter("ignore")

from utils import loadRawStockData, log
from config import Config, Interval
cfg = Config()


def getColumnFromCsv(file, col_name):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df[col_name]


def saveToCsvFromYahoo(ticker, interval):
    stock = yf.Ticker(ticker)
    tickerFormat = ticker.replace(".", "_")

    if interval == Interval.FIVE_MINUTE:
        tickerPath = f'{cfg.DATA_DIR_5_MINUTE}/{tickerFormat}.csv'
    elif interval == Interval.DAY:
        tickerPath = f'{cfg.DATA_DIR_24_HOUR}/{tickerFormat}.csv'
    else:
        log(f'Requested interval {interval} not recognised!')
        raise KeyError

    period = "max" if interval == Interval.DAY else "60d"
    try:
        if os.path.exists(tickerPath):
            dfOld = loadRawStockData(tickerFormat, interval)
            df = stock.history(interval=interval.value, period=period)
            df = pd.concat([dfOld[df.columns], df]).drop_duplicates()
        else:
            df = stock.history(interval=interval.value, period=period)
            if len(df) < 100:
                log(f'{ticker.ljust(4)} does not have enough data to be useful!')
                return False

        df.to_csv(tickerPath)
        log(f'{ticker.ljust(4)} {interval.value} was successfully fetched')

    except Exception as ex:
        log(f'{ticker.ljust(4)} was unable to be fetched ({ex})')
        return False
    return True


def checkAndCreateDirectories():
    dirsToMake = [cfg.DATA_DIR, cfg.DATA_DIR_24_HOUR, cfg.DATA_DIR_5_MINUTE, cfg.LOG_DIR]

    for dirToMake in dirsToMake:
        if not os.path.exists(dirToMake):
            os.makedirs(dirToMake)


def getYahooFinanceIntervalData(ticker):
    #saveToCsvFromYahoo(ticker, interval=Interval.DAY)
    saveToCsvFromYahoo(ticker, interval=Interval.FIVE_MINUTE)
    return True


def getFinanceData():
    tickers = list(getColumnFromCsv(f"../Wilshire-5000-Stocks.csv", "Ticker"))

    checkAndCreateDirectories()

    with ThreadPool(cfg.THREADS) as tp:
        r = list(tp.imap(getYahooFinanceIntervalData, tickers))


def main():
    getFinanceData()


if __name__ == '__main__':
    main()