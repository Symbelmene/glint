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
    ticker_data = yf.download(tickers, interval='1h', period='2y')
    insert_tickers_into_db(pg_conn, ticker_data)


def insert_tickers_into_db(pg_conn, ticker_data):
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


def get_initial_stock_data(sector_name):
    # Database connection parameters
    pg_conn = PGConn()

    # Get list of tickers for the sector
    tickers = pg_conn.get_tickers_for_sector(sector_name)

    # Check current state of database and get tickers to update
    download_initial_ticker_data(pg_conn, tickers)


def update_stock_data():
    # Checks most recent fetch for each ticker and updates the database
    pg_conn = PGConn()

    tickers = pg_conn.get_distinct_tickers_in_db()

    # Most recent date in database
    query = "SELECT MAX(date) FROM stock_data"
    with pg_conn.conn.cursor() as cursor:
        cursor.execute(query)
        last_date = cursor.fetchone()[0]

    # Download data from last_date to now
    log_message("Downloading data...")
    ticker_data = yf.download(tickers, interval='1h', start=last_date)

    date_mask = pd.to_datetime(ticker_data.index).map(lambda i: i.replace(tzinfo=None)) > last_date

    ticker_data = ticker_data[date_mask]

    insert_tickers_into_db(pg_conn, ticker_data)


def main():
    get_initial_stock_data('Information Technology')

    last_updated = time.time()
    while True:
        if time.time() - last_updated > cfg.UPDATE_RATE_DAYS * 86400:
            update_stock_data()
            last_updated = time.time()


if __name__ == "__main__":
    main()