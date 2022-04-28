import pandas as pd

from config import Config
cfg = Config()

def loadStockData(ticker):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        df = pd.read_csv(f'{cfg.DATA_DIR_RAW_24H}/{ticker}.csv', parse_dates=['Date']).set_index('Date')
    except FileNotFoundError as ex:
        print(ex)
    else:
        return df