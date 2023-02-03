import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from config import Config
cfg = Config()


class Portfolio:
    def __init__(self, tickers):
        self.tickers = tickers


class Ticker:
    def __init__(self, ticker, interval):
        self.id = ticker
        self.interval = interval

        if interval == Interval.DAY:
            self.path = f'{cfg.DATA_DIR_24_HOUR}/{ticker}.csv'
        elif interval == Interval.FIVE_MINUTE:
            self.path = f'{cfg.DATA_DIR_5_MINUTE}/{ticker}.csv'
        else:
            log(f'Interval not recognised {interval}')
            raise KeyError

        self.data = None

    def load(self):
        self.data = utils.loadStockData(self.id, self.interval)

    def slice(self, startTime, endTime):
        if isinstance(startTime, str):
            startTime = pd.to_datetime(startTime)
        if isinstance(endTime, str):
            endTime = pd.to_datetime(endTime)

        self.data = self.data[(self.data.index >= startTime) &
                              (self.data.index <= endTime)]

        if len(self.data) == 0:
            log(f'WARNING: No data exists in range {startTime} - {endTime} for ticker {self.id}')


def getPortfolioReturn(df, weights):
    returns = np.log(df / df.shift(1))

    pReturn = np.sum(weights * returns.mean()) * 252
    pVolatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    pSharpeRatio = (pReturn - cfg.RISK_FREE_RATE) / pVolatility

    return {'Return'      : pReturn,
            'Volatility'  : pVolatility,
            'Sharpe Ratio': pSharpeRatio,
            'Weights'     : weights}


def getRandomWeights(num):
    weights = np.random.random(num)
    weights /= np.sum(weights)
    return weights


def analysePortfolioBetweenDates(portfolio, interval, sTime, eTime, numRolls=10000):
    dfTickersClose = utils.loadMultipleDFsAndMergeByColumnName('Close', sTime, eTime, interval, portfolio)

    weightsList = [getRandomWeights(len(portfolio)) for _ in range(numRolls)]
    dfResults   = pd.DataFrame([getPortfolioReturn(dfTickersClose, weights) for weights in tqdm(weightsList)])

    maxSharpeVals = np.argmax(dfResults['Sharpe Ratio'])
    bestResults = dfResults.loc[maxSharpeVals]
    bestWeights = {portfolio[i]: round(bestResults['Weights'][i], 3) for i in range(len(portfolio))}
    return {'Optimal Weights' : bestWeights,
            'Volatility'      : round(bestResults['Volatility'], 5),
            'Return'          : round(bestResults['Return'], 5)}


def formatPrintOptimalDict(inDict):
    print('----OPTIMAL DICT----')
    print('Ticker Ratios:')
    for ticker, ratio in inDict['Optimal Weights'].items():
        print(f'\t{ticker.ljust(4)} : {round(ratio*100, 1)} %')
    print(f'Volatility : {round(inDict["Volatility"], 3)}')
    print(f'Return     : {round(inDict["Return"]*100, 1)} %')
    print('--------------------')





def analyseTickerPerformance(ticker, interval, sTime, eTime):
    df = utils.loadStockData(ticker, interval)

    df = df[(df.index >= sTime) & (df.index <= eTime)]
    df = utils.addBaseIndicatorsToDf(df)

    # Calc SD
    sd = df['interval_return'].std()

    # Calc average return
    mean = df['interval_return'].mean()

    return {'Ticker' : ticker,
            'Mean Return' : mean,
            'Standard Deviation' : sd}