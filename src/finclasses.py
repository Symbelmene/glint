import os
import math
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd
import plotly.express as px

from config import Config
cfg = Config()


class Portfolio:
    def __init__(self, tickerList, name):
        self.name = name
        self.tickerDict = {ticker.name : ticker for ticker in tickerList}
        self.weights = [1 / len(tickerList)] * len(tickerList)

        self.data = self.mergeOnColumn('Close')

    def mergeOnColumn(self, column):
        # Check index compatibility of tickers
        minCommonTime = max([ticker.data.index.min() for ticker in self.tickerDict.values()])
        maxCommonTime = min([ticker.data.index.max() for ticker in self.tickerDict.values()])
        df = pd.concat([t.data[(t.data.index >= minCommonTime) & (t.data.index <= maxCommonTime)][column]
                        for t in self.tickerDict.values()], axis=1)
        df.columns = [t for t in self.tickerDict]
        return df

    def plot(self):
        fig = px.line(self.data, x=self.data.index, y=self.data.columns)
        fig.update_xaxes(title="Date", rangeslider_visible=True)
        fig.update_yaxes(title="Price")
        fig.update_layout(height=800, width=1200, showlegend=True)
        fig.show()

    def optimise(self, nSamples=2000, riskFreeRate=0.0125):
        returns = np.log(self.data / self.data.shift(1))
        p_ret = []  # Returns list
        p_vol = []  # Volatility list
        p_SR = []  # Sharpe Ratio list
        p_wt = []  # Stock weights list
        retList = []
        for _ in tqdm(range(nSamples)):
            # Generate random weights
            p_weights = np.random.random(len(self.tickerDict))
            p_weights /= np.sum(p_weights)

            # Add return using those weights to list
            ret_1 = np.sum(p_weights * returns.mean()) * 252
            p_ret.append(ret_1)

            # Add volatility or standard deviation to list
            vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
            p_vol.append(vol_1)

            # Get Sharpe ratio
            SR_1 = (ret_1 - riskFreeRate) / vol_1
            p_SR.append(SR_1)

            # Store the weights for each portfolio
            p_wt.append(p_weights)

            retDict = {'return': ret_1, 'volatility': vol_1, 'sharpe_ratio': SR_1, 'weights': p_weights}
            retList.append(retDict)

        # Convert to Numpy arrays
        p_ret = np.array(p_ret)
        p_vol = np.array(p_vol)
        p_SR = np.array(p_SR)
        p_wt = np.array(p_wt)
        SR_idx = np.argmax(p_SR)

        # Find the ideal portfolio weighting at that index
        self.weights = [p_wt[SR_idx][i] * 100 for i in range(len(self.tickerDict))]
        duration = str(self.data.index.max() - self.data.index.min()).replace("00:00:00", "")
        print(f'Portfolio {self.name} optimised:\n '
              f'\tReturn:     {round(100*p_vol[SR_idx], 2)}%\n'
              f'\tDuration:   {duration}\n'
              f'\tVolatility: {round(p_ret[SR_idx], 3)}')
        return p_vol[SR_idx], p_ret[SR_idx]

    def slice(self, start, end):
        for name, ticker in self.tickerDict.items():
            self.tickerDict[name] = ticker.slice(start, end)

    def calc_volatility(self):
        return 0.01 * np.sqrt(np.dot(np.array(self.weights).T, np.dot(
            np.log(self.data / self.data.shift(1)).cov() * 252, self.weights)))

    def calc_return(self):
        return 0.01 * np.sum(self.weights * np.log(self.data / self.data.shift(1)).mean()) * 252

    def calc_sharpe(self):
        return self.calc_return() / self.calc_volatility()


class Ticker:
    def __init__(self, path, preprocess=False):
        self.name = path.split('/')[-1].replace('.csv', '')
        self.path = path
        self.data = loadRawStockData(path)
        self.normalised = False

        if preprocess:
            self.preprocess()

    def preprocess(self):
        self.data = addBaseIndicatorsToDf(self.data)

    def resample(self, deltaT):
        # TODO Resample data by a given time delta
        pass

    def standardise(self):
        # TODO Standardise data and return the scaler and normalised data
        pass

    def slice(self, start=None, end=None):
        if not start:
            start = self.data.index[0]
        else:
            end = pd.to_datetime(end)
            if start < self.data.index[0]:
                print(f"WARNING: The time slice on the dataframe for {self.name} "
                      f"extends before the start of available data")
        if not end:
            end = self.data.index[-1]
        else:
            start = pd.to_datetime(start)
            if end > self.data.index[-1]:
                print(f"WARNING: The time slice on the dataframe for {self.name} "
                      f"extends beyond the end of available data")

        mask = (self.data.index >= start) & (self.data.index <= end)
        self.data = self.data[mask]

    def calc_volatility(self):
        # Returns volatility for current time periood
        return self.data['Close'].std() / math.sqrt(len(self.data))

    def calc_return(self):
        # Calculates return for given period as decimal (0.0 == break even)
        return np.sum(self.data['Close'] / self.data['Close'].shift(1) - 1)

    def calc_sharpe(self):
        return self.calc_return() / self.calc_volatility()


def loadRawStockData(path):
    df = pd.read_csv(path, index_col=0, parse_dates=['Date'])
    return df[~df.index.duplicated(keep='first')]


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


def loadTicker(path):
    tickerName = path.split('/')[-1].split('.')[0]
    return tickerName, Ticker(path)


def loadTickers(path, processes=cfg.PROCESSES):
    tickerPaths = [f'{path}/{f}' for f in os.listdir(path)][:500]
    with Pool(processes) as p:
        tickers = list(tqdm(p.imap(loadTicker, tickerPaths), total=len(tickerPaths)))
    return {name: ticker for name, ticker in tickers}
