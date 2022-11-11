from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

from config import Config
cfg = Config()


def normaliseDataframe(df):
    arr = df.values
    dic = {'min' : arr.min(axis=0),
           'max' : arr.max(axis=0)
           }
    arr = (arr - dic['min']) / (dic['max'] - dic['min'])

    return pd.DataFrame(arr, columns=df.columns).set_index(df.index), dic


def denormaliseDataframe(df, scaler):
    arr = df.values
    arr = arr * (scaler['max'] - scaler['min']) + scaler['min']
    return pd.DataFrame(arr, columns=df.columns).set_index(df.index)


def findReturnPoints(wkDict):
    df = wkDict['df'][77:]
    dfRet = pd.DataFrame()
    for intvl in range(1, wkDict['maxRetInterval']):
        dfRet[f'interval_return_{intvl}'] = df['daily_return'].rolling(intvl).sum().shift(-1*intvl)
    df['labels'] = dfRet.apply(lambda row: any((val > wkDict['acceptableReturn']) & (val < wkDict['maxReturn']) for val in row), axis=1)
    return wkDict['ticker'], df


def findAllReturnPoints(dfDict, maxRetInterval, acceptableReturn, maxReturn=0.20):
    '''
    Finds all points in the dataset at which investing would produce at least
    <acceptableReturnPerc +float> within <maxRetTime datetime>
    '''

    workList = [{'ticker' : ticker,
                 'df' : df,
                 'maxRetInterval' : maxRetInterval,
                 'acceptableReturn' : acceptableReturn,
                 'maxReturn' : maxReturn} for ticker, df in dfDict.items()]

    with Pool(8) as p:
        dfList = list(tqdm(p.imap(findReturnPoints, workList), total=len(dfDict)))

    dfDict = {k: v for k, v in dfList}

    totalGoodPoints = sum([df['labels'].sum() for df in dfDict.values()])
    totalPoints     = sum([len(df) for df in dfDict.values()])
    print(f'Found {totalGoodPoints} / {totalPoints} valid data points for at least a {100*acceptableReturn}% return within {maxRetInterval} intervals')
    return dfDict




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


def addBaseIndicatorsToDf(df):
    df = addDailyReturnToDF(df)
    df = addCumulativeReturnToDF(df)
    df = addBollingerBands(df)
    df = addIchimoku(df)
    return df


def addBasicIndicatorsToAllCSVs():
    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_24H)]
    for ticker in tqdm(tickers):
        addBaseIndicatorsToDf(ticker, '24H')

    tickers = [ticker.split('.')[0] for ticker in os.listdir(cfg.DATA_DIR_RAW_5M)]
    for ticker in tqdm(tickers):
        addBaseIndicatorsToDf(ticker, '5M')
