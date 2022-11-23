import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import utils
import preprocess
from config import Config, Interval
cfg = Config()

class Dataset:
    def __init__(self, interval: Interval):
        self.interval = interval
        self.tickers  = utils.getValidTickers(interval)

    def createTrainset(self, maxInterval: int, acceptableReturn: float, maxReturn: float, inputPeriod: int):
        trainSetX, trainSetY = [], []
        tickerStdDict = {}
        for ticker in tqdm(self.tickers[:20]):
            df = self.__loadDataFromCSV(ticker)
            df = self.__addBasicIndicators(df)
            df = self.__clean(df)
            df = self.__findReturnPoints(df, maxInterval, acceptableReturn, maxReturn)
            df, stdDict = self.__standardise(df)
            trainX, trainY = self.__toTrainArray(df, inputPeriod)
            trainSetX += trainX
            trainSetY += trainY
            tickerStdDict[ticker] = stdDict
        trainX, trainY = np.array(trainSetX), np.array(trainSetY)

        np.save(f'{cfg.TRAIN_DIR}/trainX', trainX)
        np.save(f'{cfg.TRAIN_DIR}/trainY', trainY)

        with open('ticker_scalers.pickle', 'wb+') as wf:
            pickle.dump(tickerStdDict, wf)

    def load(self):
        self.trainX = np.load(f'{cfg.TRAIN_DIR}/trainX.npy', allow_pickle=True)
        self.trainY = np.load(f'{cfg.TRAIN_DIR}/trainY.npy', allow_pickle=True)

        with open('ticker_scalers.pickle', 'rb') as rf:
            self.scaler = pickle.load(rf)

    def __toTrainArray(self, df: pd.DataFrame, inputPeriod: int) -> (list, list):
        arr = df.to_numpy()
        arrX, arrY = arr[:,:-1], arr[:,-1]
        trainList = [(arrX[i-inputPeriod:i, :], arrY[i]) for i in range(inputPeriod, len(df))]
        return map(list, zip(*trainList))

    def __standardise(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        stdDict = {}
        for col in df.columns[:-1]:
            stdDict[col] = {'max' : df[col].max(),
                            'min' : df[col].min()}
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df, stdDict

    def __clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(['Dividends', 'Stock Splits', 'Lagging'], axis=1).dropna()

    def __loadDataFromCSV(self, ticker: str) -> pd.DataFrame:
        return utils.loadRawStockData(ticker, self.interval)

    def __addBasicIndicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess.addBaseIndicatorsToDf(df)

    def __findReturnPoints(self, df: pd.DataFrame, maxInterval: int,
                           acceptableReturn: float, maxReturn: float) -> pd.DataFrame:
        # maxInterval - integer value for number of rows
        # acceptableReturn - value for ROI (e.g. 0.1 = 10% return)
        # maxReturn - value for max ROI (to eliminate outliers) same format as acceptableReturn
        return preprocess.findReturnPoints(df, maxInterval, acceptableReturn, maxReturn)


class LSTM:
    def __init__(self):
        pass

    def build(self, x, weights, biases):
        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, n_input, 1)

        # 1-layer LSTM with n_hidden units.
        rnn_cell = rnn.BasicLSTMCell(n_hidden)

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


def main():
    dSet = Dataset(Interval.FIVE_MINUTE)
    #dSet.createTrainset(5, 0.04, 0.2, 20)

    dSet.load()


if __name__ == '__main__':
    main()