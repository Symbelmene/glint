import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px

from getdata import getFinanceData

import utils
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


def plotStockTickersInPeriod(colName, tickers, interval, sTime, eTime):
    df = utils.loadMultipleDFsAndMergeByColumnName(colName, sTime, eTime, interval, tickers)

    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title=colName)
    fig.update_layout(height=800, width=1600,
                      showlegend=True)
    fig.show()


def analyseTickerPerformance(ticker, interval, sTime, eTime):
    df = utils.loadRawStockData(ticker, interval)

    df = df[(df.index >= sTime) & (df.index <= eTime)]
    df = utils.addBaseIndicatorsToDf(df)

    # Calc SD
    sd = df['interval_return'].std()

    # Calc average return
    mean = df['interval_return'].mean()

    return {'Ticker' : ticker,
            'Mean Return' : mean,
            'Standard Deviation' : sd}


def main():
    #getFinanceData()
    portfolio = ['CALX', 'NOVT', 'RGEN', 'LLY',
                 'AMD', 'NFLX', 'COST', 'BJ', 'WING',
                 'MSCI', 'CBRE']
    interval = Interval.DAY

    allTickers = utils.getValidTickers(interval)[:50]

    startTime = '2022-01-01'
    endTime   = '2022-11-05'

    sTime = pd.to_datetime(startTime)
    eTime = pd.to_datetime(endTime)

    dfAnalysis = pd.DataFrame([analyseTickerPerformance(ticker, interval, sTime, eTime) for ticker in allTickers]).set_index('Ticker')
    dfAnalysis['cat_ratio'] = dfAnalysis['Mean Return'] / dfAnalysis['Standard Deviation']

    dfAnalysis = dfAnalysis.sort_values(by='cat_ratio', ascending=False)

    plotStockTickersInPeriod('cum_return', list(dfAnalysis.index)[:10], Interval.DAY, sTime, eTime)

    optimalDict = analysePortfolioBetweenDates(portfolio, Interval.DAY, sTime, eTime, numRolls=5000)
    formatPrintOptimalDict(optimalDict)


if __name__ == '__main__':
    main()

