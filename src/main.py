import numpy as np
import pandas as pd
from tqdm import tqdm

from getdata import getFinanceData

import utils
from preprocess import addBaseIndicatorsToDf
from config import Config, Interval
cfg = Config()



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


def analysePortfolioBetweenDates(portfolio, interval, sTime, eTime):
    sTime = pd.to_datetime(sTime)
    eTime = pd.to_datetime(eTime)

    dfTickersClose = utils.loadMultipleDFsAndMergeByColumnName('Close', sTime, eTime, interval, portfolio)
    #dfTickersCumulativeReturn = utils.loadMultipleDFsAndMergeByColumnName('cum_return', sTime, eTime, interval, portfolio)

    numRolls = 1000
    weightsList = [getRandomWeights(len(portfolio)) for _ in range(numRolls)]
    dfResults   = pd.DataFrame([getPortfolioReturn(dfTickersClose, weights) for weights in tqdm(weightsList)])



def main():
    #getFinanceData()

    testPortfolio = ['CALX', 'NOVT', 'RGEN', 'LLY',
                     'AMD', 'NFLX', 'COST', 'BJ', 'WING',
                     'MSCI', 'CBRE']
    startTime = '2017-01-01'
    endTime   = '2022-01-01'

    analysePortfolioBetweenDates(testPortfolio, Interval.DAY, startTime, endTime)


if __name__ == '__main__':
    main()

