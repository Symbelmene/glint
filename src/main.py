import get_ticker_data as gtd
from connectors import PGConn

import yfinance as yf


if __name__ == '__main__':
    gtd.update_all_sectors()
