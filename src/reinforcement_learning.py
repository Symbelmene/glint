import gymnasium as gym
from gymnasium import Env
import numpy as np
import pandas as pd

import utils
from config import Config
cfg = Config()


class StockMarket(Env):
    def __init__(self, numStocks, windowSize, start, end, startMoney):
        super(StockMarket, self).__init__()

        # Define observation space: This will randomly select numStocks stocks from the dataset and create an
        # obervable window of size windowSize. The observation space will be a 3D array of shape (numStocks,
        # windowSize, 1) which will contain the Close value of the loaded stocks.
        self.observation_shape = (numStocks, windowSize, 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float32)

        # Define action space. This is the number of different actions that can be taken by the agent, which is
        # buy or sell for each stock, or hold. This is a discrete space with 1+(numStocks*2) possible actions.
        self.action_space = gym.spaces.Discrete(1+(numStocks*2),)

        # Define chosen stocks in the environment to get information from
        self.tickers = []

        # Define the times to start and end the stock market simulation at
        self.start = start
        self.end = end

        # Define the starting money
        self.startMoney = startMoney

    def reset(self):
        # Reset start money and holdings
        self.money = self.startMoney
        self.holdings = np.zeros(len(self.tickers))

        # Reset the reward
        self.totalReward = 0

        # Reset the current step
        self.currStep = 0

        # Choose numStocks random stocks from the dataset
        validTickers = utils.getValidTickers(cfg.DATA_DIR_24_HOUR)
        self.tickers = np.random.choice(validTickers, size=self.observation_shape[0], replace=False)
        stocks = [Stock(ticker) for ticker in self.tickers]

        # Slice the ticker frames on the start and end dates then merge them into one dataframe
        for stock in stocks:
            stock.slice(self.start, self.end)

        self.stockData = pd.concat([stock.data for stock in stocks], axis=1, keys=self.tickers)

        self.window = self.stockData.iloc[:self.windowSize]

    def get_action_meanings(self):
        actionList = ['Hold']
        for ticker in self.tickers:
            actionList.append(f'Buy {ticker}')
            actionList.append(f'Sell {ticker}')
        return {idx: action for idx, action in enumerate(actionList)}

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass


class Stock:
    def __init__(self, name):
        self.name = name
        self.path = f'{cfg.DATA_DIR_24_HOUR}/{name}.csv'
        self.data = self.load()

    def load(self):
        try:
            df = pd.read_csv(self.path, index_col=0, parse_dates=['Date'])
            return df[~df.index.duplicated(keep='first')]
        except FileNotFoundError:
            print(f"ERROR: File {self.path} not found")
            exit(1)

    def slice(self, start=None, end=None):
        if not start:
            start = self.data.index[0]
        else:
            end = pd.to_datetime(end)
            if start < self.data.index[0]:
                raise ValueError(f"ERROR: The time slice on the dataframe for {self.name} "
                                    f"extends before the start of available data")
        if not end:
            end = self.data.index[-1]
        else:
            start = pd.to_datetime(start)
            if end > self.data.index[-1]:
                raise ValueError(f"ERROR: The time slice on the dataframe for {self.name} "
                                    f"extends beyond the end of available data")

        mask = (self.data.index >= start) & (self.data.index <= end)
        self.data = self.data[mask]


def main():
    sm = StockMarket(numStocks=5,
                     windowSize=10,
                     start='2019-01-01',
                     end='2020-01-01',
                     startMoney=10000)

    sm.reset()


if __name__ == '__main__':
    main()