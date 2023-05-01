import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from config import Config, Interval
from finclasses import addBaseIndicatorsToDf
cfg = Config()


def log(e, display=True):
    if display:
        print(e, flush=True)

    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)

    logPath = f'{cfg.LOG_DIR}/log.txt'
    if not os.path.exists(logPath):
        with open(logPath, 'w+') as wf:
            wf.write(f'{datetime.now()}: {e}\n')
    else:
        with open(logPath, 'a') as wf:
            wf.write(f'{datetime.now()}: {e}\n')


def loadRawStockData(ticker, interval):
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

    df = df[~df.index.duplicated(keep='first')]
    return df


def loadMultipleDFsAndMergeByColumnName(colName, sDate, eDate, interval, tickers):
    mult_df = pd.DataFrame()

    for x in tickers:
        df = loadRawStockData(x, interval)

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
    return {ticker : loadRawStockData(ticker, interval) for ticker in tickers}


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


def pruneTickerList():
    # Remove tickers that do not have more than X entries
    # Remove tickers that do not have current entries
    # Remove tickers with large gaps in data
    # Remove tickes from csv that do not exist in database
    pass