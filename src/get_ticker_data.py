import yfinance as yf
import pandas as pd
from datetime import datetime as dt

from connectors import PGConn


def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


def check_existing_data(ticker, connection, now):
    query = f"SELECT MAX(date) FROM stock_data WHERE ticker = '{ticker}'"
    with connection.cursor() as cursor:
        cursor.execute(query)
        max_date = cursor.fetchone()[0]
        if max_date == now:
            return True, None
        else:
            return False, max_date


def update_database(ticker, data, connection):
    insert_query = f"INSERT INTO stock_data (date, ticker, open, high, low, close, adj_close, volume) " \
                   f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    inserts = [(row.name.date(),
                ticker,
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Adj Close'],
                int(row['Volume'])) for idx, row in data.iterrows()]

    with connection.cursor() as cursor:
        cursor.executemany(insert_query, inserts)

    # Commit the changes to the database
    connection.commit()


def update_sector_tickers(sector_name):
    # Database connection parameters
    pg_conn = PGConn()

    # Date range for data download
    start_date = '2000-01-01'

    # Get list of tickers for the sector
    tickers = pg_conn.get_tickers_for_sector(sector_name)

    # Check if data already exists for tickers and find out most recent date
    now = dt.now().date()
    for ticker in tickers:
        up_to_date, most_recent_date = check_existing_data(ticker, pg_conn.conn, now)
        if up_to_date:
            tickers.remove(ticker)


    # Check current state of database and get tickers to update
    # TODO: Try to get missing_tickers and if unable then remove from ticker list
    # TODO: Get missing ticker data by specifying start and end date
    # TODO: Ensure duplicate date entries are not posted to database

    # Download tickers data
    print("Downloading data...")
    ticker_data = yf.download(tickers, start=start_date)
    ticker_groups = ticker_data.T.groupby(level=1)

    # Create a database connection
    for ticker, group in ticker_groups:
        print(f'Updating data for {ticker}')
        df_ticker = group.T
        df_ticker.columns = df_ticker.columns.droplevel(1)
        df_ticker = df_ticker[df_ticker.notna().all(axis=1)]
        if df_ticker.empty:
            continue

        # Check if data for the ticker already exists in the database
        pg_conn.insert_stock_data(ticker, df_ticker)


def main():
    update_sector_tickers('Information Technology')


if __name__ == "__main__":
    main()