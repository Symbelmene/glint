import time
import pandas as pd
import yfinance as yf

from connectors import PGConn
from debug import log_message

from config import Config
cfg = Config()


def download_initial_ticker_data(pg_conn, tickers):
    # Download tickers data
    log_message("Downloading data...")
    ticker_data = yf.download(tickers, interval='1h', period='3mo')
    ticker_groups = ticker_data.T.groupby(level=1)
    # Create a database connection
    for ticker, group in ticker_groups:
        log_message(f'Updating data for {ticker}')
        df_ticker = group.T
        df_ticker.columns = df_ticker.columns.droplevel(1)
        df_ticker = df_ticker[df_ticker.notna().all(axis=1)]
        if df_ticker.empty:
            continue

        # Check if data for the ticker already exists in the database
        pg_conn.insert_stock_data(ticker, df_ticker)


def update_sector_tickers(sector_name, conn):
    # Get list of tickers for the sector
    tickers = conn.get_tickers_for_sector(sector_name)
    download_ticker_data(conn, tickers)


def update_all_sectors():
    pg_conn = PGConn()
    sector_list = pg_conn.get_sectors()
    for sector in sector_list:
        update_sector_tickers(sector, pg_conn)


def main():
    update_all_sectors()
    #update_sector_tickers('Information Technology', pg_conn)


if __name__ == "__main__":
    main()
