from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool


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


def preprocessStockData():
    pass