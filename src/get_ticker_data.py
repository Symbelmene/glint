import yfinance as yf

from connectors import PGConn
from debug import log_message


def download_ticker_data(pg_conn, tickers):
    # Download tickers data
    log_message("Downloading data...")
    ticker_data = yf.download(tickers, interval='1h', period='2y')
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


def update_sector_tickers(sector_name):
    # Database connection parameters
    pg_conn = PGConn()

    # Get list of tickers for the sector
    tickers = pg_conn.get_tickers_for_sector(sector_name)

    # Check current state of database and get tickers to update
    # TODO: Try to get missing_tickers and if unable then remove from ticker list
    # TODO: Get missing ticker data by specifying start and end date
    # TODO: Ensure duplicate date entries are not posted to database

    download_ticker_data(pg_conn, tickers)


def main():
    update_sector_tickers('Information Technology')


if __name__ == "__main__":
    main()