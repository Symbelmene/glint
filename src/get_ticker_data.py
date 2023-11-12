import yfinance as yf
import psycopg2
import pandas as pd
from datetime import datetime as dt


def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


def create_database_connection(db_params):
    connection = psycopg2.connect(**db_params)
    return connection


def check_existing_data(ticker, connection, now):
    query = f"SELECT COUNT(*) FROM stock_data WHERE ticker = '{ticker}'"
    with connection.cursor() as cursor:
        cursor.execute(query)
        count = cursor.fetchone()[0]
    return count > 0


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


def check_database_status(tickers, connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT DISTINCT ON (ticker) * FROM stock_data ORDER BY ticker, date DESC")
        tickers_in_db = cursor.fetchall()
    now = dt.now().date()
    tickers_to_update = [(val[2], val[1]) for val in tickers_in_db if val[1] < now]
    missing_tickers = [ticker for ticker in tickers if ticker not in [val[2] for val in tickers_in_db]]
    return tickers_to_update, missing_tickers


def download_tickers():
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'database': 'findata',
        'user': 'user',
        'password': 'pass'
    }

    connection = create_database_connection(db_params)
    dfTickers = pd.read_csv('../Wilshire-5000-Stocks.csv')
    tickers = [ticker for ticker in dfTickers['Ticker'] if ticker.isalpha()]

    # Date range for data download
    start_date = '2000-01-01'

    # Check current state of database and get tickers to update
    tickers_to_update, missing_tickers = check_database_status(tickers, connection)
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
        dfTicker = group.T.round(3)
        dfTicker.columns = dfTicker.columns.droplevel(1)
        dfTicker = dfTicker[dfTicker.notna().all(axis=1)]

        # Check if data for the ticker already exists in the database
        update_database(ticker, dfTicker, connection)

    # Close the database connection
    connection.close()


def main():
    download_tickers()


if __name__ == "__main__":
    main()