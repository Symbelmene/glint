import os
from tqdm import tqdm
import pandas as pd


class Ticker:
    def __init__(self, path):
        self.name = path.split('/')[-1].replace('.csv', '')
        self.path = path
        self.data = self.__load()
        self.normalised = False

    def __load(self):
        df = pd.read_csv(self.path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date', drop=True)

    def resample(self, deltaT):
        # TODO Resample data by a given time delta
        pass

    def standardise(self):
        # TODO Standardise data and return the scaler and normalised data
        pass


def main():
    dbPath = 'D:/findata/data/24_HOUR'
    tickerList = [Ticker(f'{dbPath}/{t}') for t in tqdm(os.listdir(dbPath))]
    x = 1


if __name__ == '__main__':
    main()