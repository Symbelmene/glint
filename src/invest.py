import pandas as pd
import matplotlib.pyplot as plt

import utils
from finclasses import Ticker, Portfolio
from config import Config, Interval
cfg = Config()


def getTickerSummaryData(ticker, bandSize=50):
    rollingVolatility = ticker.data['interval_return'].rolling(bandSize).apply(utils.calculateVolatility)
    rollingReturn = ticker.data['interval_return'].rolling(bandSize).sum()
    rollingSharpe = (rollingReturn - cfg.RISK_FREE_RATE) / rollingVolatility
    return rollingVolatility, rollingReturn, rollingSharpe


def plotTickerSummaryData(ticker, bandSize=50):
    rollingVolatility, rollingReturn, rollingSharpe = getTickerSummaryData(ticker, bandSize)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(rollingVolatility)
    axs[0].set_title('Volatility')
    axs[1].plot(rollingReturn)
    axs[1].set_title('Return')
    axs[2].plot(rollingSharpe)
    axs[2].set_title('Sharpe Ratio')
    plt.suptitle(f'Ticker \'{ticker.name}\' Summary Data')
    plt.show()


def findGoodStocks(startTime, endTime, interval=Interval.DAY):
    # Iterate through tickers and do the following:
    # 1. Load the ticker data
    # 2. Slice the ticker data to the currTime given
    # 3. Calculate a rolling return, volatility and sharpe ratio
    # 4. Return a list of tickers that satisfy some equation TBD
    tickers = utils.getValidTickers(interval)
    basePath = cfg.DATA_DIR_24_HOUR if interval == Interval.DAY else cfg.DATA_DIR_5_MINUTE
    for ticker in tickers:
        tickerPath = f'{basePath}/{ticker}.csv'
        t = Ticker(tickerPath, start=startTime, end=endTime)
        t.preprocess()
        plotTickerSummaryData(t, bandSize=60)
        break


def main():
    findGoodStocks(pd.to_datetime('2021-01-01'), pd.to_datetime('2023-06-30'))
