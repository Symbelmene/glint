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

from utils import loadRawStockData
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
    tickerPath = f'{cfg.DATA_DIR_RAW_24H}/{tickerFormat}.csv'
    try:
        if os.path.exists(tickerPath):
            dfOld = loadRawStockData(tickerFormat, '24H')
            df = stock.history(period="5y", start=dfOld.index[-1])
            df = pd.concat([dfOld[df.columns], df]).drop_duplicates()
        else:
            df = stock.history(period="5y")
            if len(df) < 100:
                log(f'{ticker.zfill(4)} does not have enough data to be useful!')
                return False

        df.to_csv(tickerPath)
        log(f'{ticker.zfill(4)} was successfully fetched')

    except Exception as ex:
        log(f'{ticker.zfill(4)} was unable to be fetched ({ex})')
        return False
    return True


def saveToCsvFromYahoo5M(ticker):
    stock = yf.Ticker(ticker)
    tickerFormat = ticker.replace(".", "_")

    tickerPath = f'{cfg.DATA_DIR_RAW_5M}/{tickerFormat}.csv'
    try:
        if os.path.exists(tickerPath):
            dfOld = loadRawStockData(tickerFormat, '5M')
            df = stock.history(interval="5m") #, start=dfOld.index[-1])
            df = pd.concat([dfOld[df.columns], df]).drop_duplicates()
        else:
            df = stock.history(interval="5m")
            if len(df) < 100:
                return False

        df.to_csv(tickerPath)
    except Exception as ex:
        log(f'ERROR while getting stock {tickerFormat}:')
        log(traceback.format_exc())
        return False
    return True


def getFinanceData():
    tickers = list(getColumnFromCsv(f"../Wilshire-5000-Stocks.csv", "Ticker"))
    numThreads = cfg.THREADS

    log(f'Updating 24H data to {cfg.DATA_DIR_RAW_24H}')
    if not os.path.exists(cfg.DATA_DIR_RAW_24H):
        os.makedirs(cfg.DATA_DIR_RAW_24H)

    if numThreads == 1:
        for ticker in tickers:
            saveToCsvFromYahoo24H(ticker)
    else:
        with ThreadPool(numThreads) as p:
            r = list(p.imap(saveToCsvFromYahoo24H, tickers))

    log(f'Updating 5M data to {cfg.DATA_DIR_RAW_5M}')
    if not os.path.exists(cfg.DATA_DIR_RAW_5M):
        os.makedirs(cfg.DATA_DIR_RAW_5M)

    if numThreads == 1:
        for ticker in tickers:
            saveToCsvFromYahoo5M(ticker)
    else:
        with ThreadPool(numThreads) as p:
            r = list(p.imap(saveToCsvFromYahoo5M, tickers))


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


def addDailyReturnToDF(df):
    df['interval_return'] = (df['Close'] / df['Close'].shift(1)) - 1
    return df


def addCumulativeReturnToDF(df):
    df['cum_return'] = (1 + df['interval_return']).cumprod()
    return df


def addBollingerBands(df, window=20):
    df['middle_band'] = df['Close'].rolling(window=window).mean()
    df['upper_band']  = df['middle_band'] + 1.96 * df['Close'].rolling(window=window).std()
    df['lower_band']  = df['middle_band'] - 1.96 * df['Close'].rolling(window=window).std()
    return df


def addIchimoku(df):
    # Conversion Line - (Highest value in period / lowest value in period) / 2 (Period = 9)
    highValue = df['High'].rolling(window=9).max()
    lowValue  = df['Low'].rolling(window=9).min()
    df['Conversion'] = (highValue + lowValue) / 2

    # Base Line - (Highest value in period / lowest value in period) / 2 (Period = 26)
    highValue2 = df['High'].rolling(window=26).max()
    lowValue2  = df['Low'].rolling(window=26).min()
    df['Baseline'] = (highValue2 + lowValue2) / 2

    # Span A - (Conversion + Base) / 2 - (Period = 26)
    df['SpanA'] = ((df['Conversion'] + df['Baseline']) / 2)

    # Span B - (Conversion + Base) / 2 - (Period = 52)
    highValue3 = df['High'].rolling(window=52).max()
    lowValue3  = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((highValue3 + lowValue3) / 2).shift(26)

    # Lagging Span
    df['Lagging'] = df['Close'].shift(-26)
    return df


def addBaseIndicatorsToDf(ticker, interval):
    df = loadRawStockData(ticker, interval)
    if df is None:
        return False
    df = addDailyReturnToDF(df)
    df = addCumulativeReturnToDF(df)
    df = addBollingerBands(df)
    df = addIchimoku(df)
    if interval == '5M':
        df.to_csv(f'{cfg.DATA_DIR_CLEAN_5M}/{ticker}.csv')
    if interval == '24H':
        df.to_csv(f'{cfg.DATA_DIR_CLEAN_24H}/{ticker}.csv')
    return True


def addBasicIndicatorsToAllCSVs():
    if not os.path.exists(cfg.DATA_DIR_CLEAN_24H):
        os.makedirs(cfg.DATA_DIR_CLEAN_24H)
    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_24H)]
    for ticker in tqdm(tickers):
        addBaseIndicatorsToDf(ticker, '24H')

    if not os.path.exists(cfg.DATA_DIR_CLEAN_5M):
        os.makedirs(cfg.DATA_DIR_CLEAN_5M)
    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_5M)]
    for ticker in tqdm(tickers):
        addBaseIndicatorsToDf(ticker, '5M')


def updateFinanceDatabase():
    log('Start of finance watcher')
    while True:
        if checkIfDatabaseUpdateRequired():
            log('Updating finance database...')
            getFinanceData()

            log('Adding basic indicators...')
            addBasicIndicatorsToAllCSVs()

            log('Database update completed.')

        time.sleep(60)