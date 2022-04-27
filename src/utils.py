import pandas as pd

from dbg import log
from config import Config
cfg = Config()

def loadStockData(ticker, interval):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        if interval == '24H':
            df = pd.read_csv(f'{cfg.DATA_DIR_RAW_24H}/{ticker}.csv', parse_dates=['Date']).set_index('Date')
        elif interval == '5M':
            df = pd.read_csv(f'{cfg.DATA_DIR_RAW_5M}/{ticker}.csv', parse_dates=['Date']).set_index('Date')
        else:
            log(f'Unrecognised interval {interval}. (5M or 24H)')
            raise KeyError
    except FileNotFoundError as ex:
        print(ex)
    else:
        return df