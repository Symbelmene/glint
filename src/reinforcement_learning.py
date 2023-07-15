import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import Model, Input
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env

import utils
from config import Config, Interval
cfg = Config()


class StockMarket(Env):
    def __init__(self, numStocks, windowSize, start, end, startMoney, buyAmount=1000):
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

        # Store the buy amount
        self.buyAmount = buyAmount

    def reset(self):
        # Reset start money and holdings
        self.money = self.startMoney

        # Reset the reward
        self.totalReward = 0

        # Reset the current step
        self.currStep = 0

        # Choose numStocks random stocks from the dataset
        validTickers = utils.getValidTickers(Interval.DAY)
        self.tickers = np.random.choice(validTickers, size=self.observation_shape[0], replace=False)
        stocks = [Stock(ticker) for ticker in self.tickers]
        self.holdings = {ticker : 0 for ticker in self.tickers}

        # Slice the ticker frames on the start and end dates then merge them into one dataframe
        for stock in stocks:
            stock.slice(self.start, self.end)

        self.stockData = pd.concat([stock.data['Close'] for stock in stocks], axis=1, keys=self.tickers)

        self.window = self.stockData.iloc[:self.observation_shape[1]]

    def get_action_meanings(self):
        actionList = ['Hold']
        for ticker in self.tickers:
            actionList.append(f'Buy {ticker}')
            actionList.append(f'Sell {ticker}')
        return {idx: action for idx, action in enumerate(actionList)}

    def step(self, action, verbose=False):
        # Carry out the action
        if action == 0:
            # Hold
            report = "Holding"
        elif action % 2 == 1:
            # Buy
            ticker = self.tickers[(action-1)//2]
            price = self.window.iloc[-1][ticker]
            if self.money >= self.buyAmount:
                self.money -= self.buyAmount
                numShares = self.buyAmount // price
                self.holdings[ticker] += numShares
                report = f"Bought {numShares} shares of {ticker} at {round(price, 3)}"
            else:
                report = f"ERROR: Not enough money to buy {ticker}"
        elif action % 2 == 0:
            # Sell
            ticker = self.tickers[(action-2)//2]
            price = self.window.iloc[-1][ticker]
            if self.holdings[ticker] > 0:
                self.money += price * self.holdings[ticker]
                self.holdings[ticker] = 0
                report = f"Sold {ticker} at {round(price, 3)}"
            else:
                report = f"ERROR: No {ticker} to sell"

        # Increment the step counter
        self.currStep += 1

        # Get the new state window
        self.window = self.stockData.iloc[self.currStep:self.currStep+self.observation_shape[1]]

        reward = self.money

        # Determine if the episode is over
        done = False
        if self.currStep == len(self.stockData) - self.observation_shape[1]:
            done = True

        if verbose:
            self.summariseState(report)

        return self.window, reward, done, []

    def summariseState(self, report):
        print('\n')
        print('#' * 30)
        print(f"STEP {self.currStep}")
        print(report)
        totalShareValue = 0
        for idx, (ticker, shares) in enumerate(self.holdings.items()):
            if shares > 0:
                shareValue = round(shares*self.window.iloc[-1, idx], 2)
                shareStr = f'(${shareValue})'
                print(f"{ticker.ljust(5)}: {shares} {shareStr.rjust(8)} shares")
                totalShareValue += shareValue
        print('-'*20)
        print(f'Cash : ${round(self.money, 2)}')
        print(f'Net  : ${round(totalShareValue + self.money, 2)}')
        print('#' * 30)

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


class Agent:
    def __init__(self, numStocks, windowSize):
        self.numStocks = numStocks
        self.windowSize = windowSize
        self.numActions = 2 * numStocks + 1 # Buy or sell per stock + hold

        self.policyNetwork = self.buildModel()
        self.targetNetwork = self.buildModel()

    def buildModel(self):
        # Builds and LSTM model using the windowSize and numStocks as input and numActions
        # as output
        inputLayer  = Input(shape=(self.windowSize, self.numStocks))
        lstmLayer   = layers.LSTM(32)(inputLayer)
        denseLayer  = layers.Dense(self.numActions, activation='linear')(lstmLayer)
        outputLayer = layers.Softmax()(denseLayer)
        model       = Model(inputs=inputLayer, outputs=outputLayer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def getAction(self, state):
        # Returns an action based on the current state
        return self.policyNetwork.predict(state)

    def updateTargetNetwork(self):
        # Updates the target network with the weights of the policy network
        self.targetNetwork.set_weights(self.policyNetwork.get_weights())


def main():
    numEpisodes = 10
    # Initialise Environment
    sm = StockMarket(numStocks=5,
                     windowSize=10,
                     start=pd.to_datetime('2019-01-01'),
                     end=pd.to_datetime('2020-01-01'),
                     startMoney=10000,
                     buyAmount=1000)

    # Initialise Agent with target and policy networks
    agent = Agent(numStocks=5, windowSize=10)

    # Initialise replay memory
    replayMemory = []

    for episode in range(numEpisodes):
        # Reset the environment
        sm.reset()

        # Run episode
        while True:
            action = sm.action_space.sample()
            state, reward, done, info = sm.step(action)

            if done == True:
                break


if __name__ == '__main__':
    main()