import os
import warnings
import pandas as pd
import yfinance as yf
from multiprocessing.pool import ThreadPool
warnings.simplefilter("ignore")

from utils import log
from config import Config, Interval
cfg = Config()


def getColumnFromCsv(file, col_name):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df[col_name]


def saveToCsvFromYahoo(ticker):
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(interval='1d', period="max")
        if len(df) < 100:
            log(f'{ticker.ljust(4)} does not have enough data to be useful!')
            return False

        # TODO: Send to Postgres container
        log(f'{ticker.ljust(4)} 1d was successfully fetched')
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
    saveToCsvFromYahoo(ticker, interval=Interval.DAY)
    return True


def getFinanceData():
    tickers = list(getColumnFromCsv(f"../Wilshire-5000-Stocks.csv", "Ticker"))

    checkAndCreateDirectories()
    with ThreadPool(cfg.THREADS) as tp:
        r = list(tp.imap(getYahooFinanceIntervalData, tickers))


def getTickerInfo(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info
    except:
        return None


def main():
    getFinanceData()


if __name__ == '__main__':
    main()