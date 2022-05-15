import time
from multiprocessing import Pool
from dbg import log
from getdata import updateFinanceDatabase
import utils
from config import Config
cfg = Config()


def main():
    log('Glint is initialising...')
    p = Pool(3)

    # Start finance update process
    p.apply_async(updateFinanceDatabase, args=())

    while True:
        time.sleep(1000)


if __name__ == '__main__':
    #main()
    dfDict = utils.loadAllRawStockData('24H')


    #df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'interval_return', 'cum_return']].iloc[1:,:]
    x = 1


