import os
import json
import time
import traceback
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import warnings
warnings.simplefilter("ignore")

from utils import loadStockData
from dbg import log
from config import Config
cfg = Config()


def getColumnFromCsv(file, col_name):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df[col_name]


def saveToCsvFromYahoo24H(ticker):
    stock = yf.Ticker(ticker)
    tickerFormat = ticker.replace(".", "_")

    if not os.path.exists(cfg.DATA_DIR_RAW_24H):
        os.makedirs(cfg.DATA_DIR_RAW_24H)

    tickerPath = f'{cfg.DATA_DIR_RAW_24H}/{tickerFormat}.csv'
    try:
        if os.path.exists(tickerPath):
            dfOld = loadStockData(tickerFormat, '24H')
            df = stock.history(period="5y", start=dfOld.index[-1])
            df = pd.concat([dfOld[df.columns], df]).drop_duplicates()
        else:
            df = stock.history(period="5y")
            if len(df) < 100:
                return False

        df.to_csv(tickerPath)


    except Exception as ex:
        return False
    return True


def saveToCsvFromYahoo5M(ticker):
    stock = yf.Ticker(ticker)
    tickerFormat = ticker.replace(".", "_")

    if not os.path.exists(cfg.DATA_DIR_RAW_5M):
        os.makedirs(cfg.DATA_DIR_RAW_5M)

    tickerPath = f'{cfg.DATA_DIR_RAW_5M}/{tickerFormat}.csv'
    try:
        if os.path.exists(tickerPath):
            dfOld = loadStockData(tickerFormat, '5M')
            df = stock.history(interval="5m", start=dfOld.index[-1])
            df = pd.concat([dfOld[df.columns], df]).drop_duplicates()
        else:
            df = stock.history(interval="5m")
            if len(df) < 100:
                return False

        df.to_csv(tickerPath)
    except Exception as ex:
        log(traceback.format_exc())
        return False
    return True


def getFinanceData():
    tickers = list(getColumnFromCsv(f"../Wilshire-5000-Stocks.csv", "Ticker"))
    #with ThreadPool(16) as p:
    #    r = list(tqdm(p.imap(saveToCsvFromYahoo24H, tickers), total=len(tickers)))

    with ThreadPool(4) as p:
        r = list(tqdm(p.imap(saveToCsvFromYahoo5M, tickers), total=len(tickers)))


def checkIfDatabaseUpdateRequired():
    metaPath = f'{cfg.DATA_DIR}/db_details.json'

    if not os.path.exists(cfg.DATA_DIR):
        os.makedirs(cfg.DATA_DIR)

    if not os.path.exists(metaPath):
        with open(metaPath, 'w+') as wf:
            json.dump({}, wf)

    with open(metaPath, 'r') as rf:
        metaDict = json.load(rf)

    update = False
    if 'last_updated' not in metaDict:
        metaDict['last_updated'] = time.time()
        update = True
    if time.time() - metaDict['last_updated'] > 3600 * 24 * 7:
        metaDict['last_updated'] = time.time()
        update = True

    if update:
        with open(metaPath, 'w+') as wf:
            json.dump(metaDict, wf)

    return update


def updateFinanceDatabase():
    log('Start of finance watcher')
    while True:
        if 1: #checkIfDatabaseUpdateRequired():
            log('Updating finance database...')
            getFinanceData()
            log('Adding basic indicators to data...')

            log('Database update completed.')
        time.sleep(60)
