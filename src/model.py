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
        return utils.loadStockData(ticker, self.interval)

    def __addBasicIndicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess.addBaseIndicatorsToDf(df)

    def __findReturnPoints(self, df: pd.DataFrame, maxInterval: int,
                           acceptableReturn: float, maxReturn: float) -> pd.DataFrame:
        # maxInterval - integer value for number of rows
        # acceptableReturn - value for ROI (e.g. 0.1 = 10% return)
        # maxReturn - value for max ROI (to eliminate outliers) same format as acceptableReturn
        return preprocess.findReturnPoints(df, maxInterval, acceptableReturn, maxReturn)


class LSTM:
    def __init__(self, inputShape):
        self.build(inputShape)

    def build(self, inputShape):
        inputs = tf.keras.Input(inputShape)
        x1 = tf.keras.layers.LSTM(64)(inputs)
        x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
        x1 = tf.keras.layers.LSTM(32)(x1)
        x1 = tf.keras.layers.Dense(32, activation='relu')(x1)
        x1 = tf.keras.layers.LSTM(16)(x1)
        x1 = tf.keras.layers.Dense(16, activation='relu')(x1)
        x1 = tf.keras.layers.LSTM(8)(x1)
        x1 = tf.keras.layers.Dense(8, activation='relu')(x1)
        x1 = tf.keras.layers.LSTM(4)(x1)
        x1 = tf.keras.layers.Dense(4, activation='relu')(x1)
        outputs = tf.keras.layers.Dense(1)(x1)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['mean_absolute_error'])


def main():
    dSet = Dataset(Interval.FIVE_MINUTE)

    maxIntervalReturn = 5
    acceptableReturnFraction = 0.04
    acceptableMaximumReturnFraction = 0.2
    inputPeriod = 20

    #dSet.createTrainset(maxInterval=maxIntervalReturn,              # Number of time intervals for return to occur
    #                    acceptableReturn=acceptableReturnFraction,  # Minimum acceptable return fraction
    #                    maxReturn=acceptableMaximumReturnFraction,  # Maximum acceptable return fraction (to remove anomalies)
    #                    inputPeriod=inputPeriod)                    # Number of preceeding time intervals to construct input train data from

    input_shape = (20, 14)
    model = LSTM(input_shape)

    dSet.load()

    x = 1


if __name__ == '__main__':
    main()