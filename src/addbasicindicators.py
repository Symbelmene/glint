import os
from tqdm import tqdm
from multiprocessing import Pool

from utils import loadStockData
from config import Config
cfg = Config()


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
    df = loadStockData(ticker, interval)
    if df is None:
        return False
    df = addDailyReturnToDF(df)
    df = addCumulativeReturnToDF(df)
    df = addBollingerBands(df)
    df = addIchimoku(df)
    if interval == '5M':
        df.to_csv(f'{cfg.DATA_DIR_RAW_5M}/{ticker}.csv')
    if interval == '24H':
        df.to_csv(f'{cfg.DATA_DIR_RAW_24H}/{ticker}.csv')
    return True


def addBasicIndicatorsToAllCSVs():
    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_24H)]
    with Pool(8) as p:
        r = list(tqdm(p.imap(addBaseIndicatorsToDf, tickers), total=len(tickers)))

    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_5M)]
    with Pool(8) as p:
        r = list(tqdm(p.imap(addBaseIndicatorsToDf, tickers), total=len(tickers)))
