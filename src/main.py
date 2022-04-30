import time
from multiprocessing import Pool
from dbg import log
from getdata import updateFinanceDatabase


def main():
    log('Glint is initialising...')
    p = Pool(3)

    # Start finance update process
    p.apply_async(updateFinanceDatabase, args=())

    while True:
        time.sleep(1000)


if __name__ == '__main__':
    main()
