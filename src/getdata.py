import os
import json
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import warnings
warnings.simplefilter("ignore")

from utils import loadStockData
from config import Config
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
    tickerFormat = ticker.replace(".", "_")

    if not os.path.exists(cfg.DATA_DIR_RAW_24H):
        os.makedirs(cfg.DATA_DIR_RAW_24H)

    tickerPath = f'{cfg.DATA_DIR_RAW_24H}/{tickerFormat}.csv'
    try:
        df = stock.history(period="5y")
        if len(df) < 100:
            return False

        if os.path.exists(tickerPath):
            dfOld = loadStockData(tickerFormat)[df.columns]
            df = pd.concat([dfOld, df]).drop_duplicates()
        df.to_csv(tickerPath)
    except Exception as ex:
        return False
    return True


def getFinanceData():
    tickers = list(getColumnFromCsv(f"../Wilshire-5000-Stocks.csv", "Ticker"))
    with ThreadPool(16) as p:
        r = list(tqdm(p.imap(saveToCsvFromYahoo, tickers), total=len(tickers)))


def checkIfDatabaseUpdateRequired():
    metaPath = f'{cfg.DATA_DIR}/db_details.json'
    if not os.path.exists(metaPath):
        with open(metaPath, 'w+') as wf:
            json.dump({}, wf)

    with open(metaPath, 'r') as rf:
        metaDict = json.load(rf)

    update = False
    if 'last_updated' not in metaDict:
        metaDict['last_updated'] = time.time()
        update = True
    if time.time() - metaDict['last_updated'] > 3600 * 24:
        metaDict['last_updated'] = time.time()
        update = True

    if update:
        with open(metaPath, 'w+') as wf:
            json.dump(metaDict, wf)

    return update


def updateFinanceDatabase():
    if checkIfDatabaseUpdateRequired():
        getFinanceData()