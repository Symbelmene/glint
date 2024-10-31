import time
import traceback
import pandas as pd
import psycopg2
import psycopg2 as pg
from tqdm import tqdm
from debug import log_message
from config import Config
cfg = Config()


class PGConn:
    def __init__(self, retry=True):
        try:
            self.conn = pg.connect(dbname='postgres',
                          user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                          host=cfg.STORER_HOST, port=cfg.STORER_PORT)
        except Exception as e:
            if e == psycopg2.OperationalError and retry:
                log_message("Database connection failed. The container may not yet be ready to accept connections."
                            " Retrying in 30 seconds...")
                time.sleep(30)
                self.__init__(retry=False)

        # Check if findata database exists and create it if it doesn't
        if not self.check_database_exists(cfg.STORER_DB_NAME):
            self.create_database(cfg.STORER_DB_NAME)

        self.conn.close()

        self.conn = pg.connect(dbname=cfg.STORER_DB_NAME,
                               user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                               host=cfg.STORER_HOST, port=cfg.STORER_PORT)

        # Check if sector and ticker tables exist and create them if they don't
        self.populate_initial_tables()

    def check_database_exists(self, db_name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            return cursor.fetchone()

    def create_database(self, db_name):
        self.conn.rollback()
        self.conn.autocommit = True

        with self.conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {db_name}")

        self.conn.autocommit = False

    def populate_initial_tables(self):
        populate_base_tables(self.conn)
        create_stock_data_table_day(self.conn)
        # create_stock_data_table_hour(self.conn)

    def get_tickers(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT ticker FROM tickers")
            return [row[0] for row in cursor.fetchall()]

    def get_ticker_id(self, ticker):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT id FROM tickers WHERE ticker = %s", (ticker,))
            return cursor.fetchone()[0]

    def get_sectors(self):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT sector FROM sectors")
            return [row[0] for row in cursor.fetchall()]

    def get_tickers_for_sector(self, sector_name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT t.id, t.ticker FROM tickers t JOIN sectors s ON t.sector_id = s.id WHERE s.sector = %s",
                           (sector_name,))

            return [row[1] for row in cursor.fetchall()]

    def insert_stock_day_data(self, ticker, data):
        ticker_id = self.get_ticker_id(ticker)

        insert_query = f"INSERT INTO stock_data_day (date, ticker_id, open, high, low, close, adj_close, volume) " \
                       f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        inserts = [(pd.to_datetime(row.name), ticker_id, row['Open'], row['High'],
                    row['Low'], row['Close'], row['Adj Close'], int(row['Volume']))
                   for idx, row in data.iterrows()]

        with self.conn.cursor() as cursor:
            cursor.executemany(insert_query, inserts)

        # Commit the changes to the database
        self.conn.commit()

    def insert_stock_hour_data(self, ticker, data):
        ticker_id = self.get_ticker_id(ticker)

        insert_query = f"INSERT INTO stock_data_hour (date, ticker_id, open, high, low, close, adj_close, volume) " \
                       f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        inserts = [(pd.to_datetime(row.name), ticker_id, row['Open'], row['High'],
                    row['Low'], row['Close'], row['Adj Close'], int(row['Volume']))
                   for idx, row in data.iterrows()]

        with self.conn.cursor() as cursor:
            cursor.executemany(insert_query, inserts)

        # Commit the changes to the database
        self.conn.commit()

    def get_most_recent_date_by_ticker(self, as_dataframe=False):
        query = """SELECT DISTINCT ON (ticker) date, ticker FROM stock_data ORDER BY ticker, date DESC"""

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        if as_dataframe:
            headers = [desc.name for desc in cursor.description]
            return pd.DataFrame(data, columns=headers)

        return data

    def find_tickers_with_no_data(self):
        query = """SELECT t.ticker FROM tickers t LEFT JOIN stock_data_day sdd ON t.id = sdd.ticker_id WHERE sdd.id IS NULL"""

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def get_ticker_data_day(self, ticker, as_dataframe=True):
        query = """SELECT sdd.*, t.ticker FROM stock_data_day sdd INNER JOIN tickers t ON sdd.ticker_id = t.id WHERE ticker = %s"""
        try:
            cur = self.conn.cursor()
            cur.execute(query, (ticker,))
            ticker_data = cur.fetchall()
        except Exception as e:
            log_message(f'Error getting all feature details: {e}')
            log_message(traceback.format_exc())
            self.conn.rollback()
            return False

        if as_dataframe:
            headers = [desc.name for desc in cur.description]
            ticker_data = pd.DataFrame(ticker_data, columns=headers)

        return ticker_data.sort_values(by='date')

    def get_ticker_data_hour(self, ticker, as_dataframe=True):
        query = """SELECT sdd.*, t.ticker FROM stock_data_hour sdd INNER JOIN tickers t ON sdd.ticker_id = t.id WHERE ticker = %s"""
        try:
            cur = self.conn.cursor()
            cur.execute(query, (ticker,))
            ticker_data = cur.fetchall()
        except Exception as e:
            log_message(f'Error getting all feature details: {e}')
            log_message(traceback.format_exc())
            self.conn.rollback()
            return False

        if as_dataframe:
            headers = [desc.name for desc in cur.description]
            ticker_data = pd.DataFrame(ticker_data, columns=headers)

        return ticker_data.sort_values(by='date')

    def find_and_remove_duplicate_entries(self):
        tickers = self.get_tickers()
        for ticker in tickers:
            remove_list = []
            td = self.get_ticker_data_day(ticker)
            for name, group in td.groupby('date'):
                if len(group) > 1:
                    remove_list += list(group['id'][1:])
            if remove_list:
                print(f'{ticker}: {len(remove_list)} duplicates')
                self.delete_entry_by_row_ids(remove_list)

    def delete_entry_by_row_ids(self, row_id_list):
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM stock_data_day WHERE id IN %s", (tuple(row_id_list),))
            self.conn.commit()

    def validate(self):
        # Fetches each ticker from the tickers table and carries out the following checks:
        # For each ticker:
        #   For each of stock_data_day and stock_data_hour:
        #       Check for a minimum number of entries
        #       Check that the number of entries in the table is equal to the number of unique dates
        #       Check for gaps in the data

        tickers = self.get_tickers()
        tickers.sort()
        for ticker in tqdm(tickers):
            day_data = self.get_ticker_data_day(ticker)
            if len(day_data) < 300:
                print(f'{ticker} Day Data has less than 300 entries')
                continue
            day_differences = day_data.sort_values(by='date')['date'].diff().dt.days
            num_significant_diffs = len(day_differences[day_differences > 5])
            if num_significant_diffs > 5:
                print(f'{ticker} Day Data has {num_significant_diffs} gaps in the data > 5 days')

            hour_data = self.get_ticker_data_hour(ticker)
            if len(hour_data) < 300:
                print(f'{ticker} Hour Data has less than 300 entries')
                continue
            hour_differences = hour_data.sort_values(by='date')['date'].diff().dt.days
            num_significant_diffs = len(hour_differences[hour_differences > 5])
            if num_significant_diffs > 5:
                print(f'{ticker} Hour Data has {num_significant_diffs} gaps in the data > 1 day')


def populate_base_tables(conn):
    df = pd.read_csv('../stocks.csv')

    # Populate sectors table
    if check_if_table_exists(conn, 'sectors'):
        return True

    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE sectors (id SERIAL PRIMARY KEY, sector VARCHAR(50) UNIQUE)")
        conn.commit()

        # Insert unique sectors into sectors table and return the id
        sector_dict = {}
        for sector in df['Sector'].unique():
            cursor.execute("INSERT INTO sectors (sector) VALUES (%s) RETURNING id", (sector,))
            sector_id = cursor.fetchone()[0]
            sector_dict[sector] = sector_id
            conn.commit()

    log_message("Sectors table populated successfully")

    # Populate ticker table
    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE tickers (id SERIAL PRIMARY KEY, ticker VARCHAR(10) UNIQUE, sector_id INTEGER)")
        conn.commit()

        for idx, row in df.iterrows():
            ticker = row['Ticker']
            sector_id = sector_dict[row['Sector']]
            cursor.execute("INSERT INTO tickers (ticker, sector_id) VALUES (%s, %s)", (ticker, sector_id,))
            conn.commit()

    log_message("Ticker table populated successfully")


def create_stock_data_table_day(conn):
    if check_if_table_exists(conn, 'stock_data_day'):
        return True

    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE stock_data_day (id SERIAL PRIMARY KEY, date TIMESTAMP, ticker_id INT, open NUMERIC, "
                       "high NUMERIC, low NUMERIC, close NUMERIC, adj_close NUMERIC, volume BIGINT)")
        conn.commit()

    log_message("Stock data day table created successfully")


def create_stock_data_table_hour(conn):
    if check_if_table_exists(conn, 'stock_data_hour'):
        return True

    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE stock_data_hour (id SERIAL PRIMARY KEY, date TIMESTAMP, ticker_id INT, open NUMERIC, "
                       "high NUMERIC, low NUMERIC, close NUMERIC, adj_close NUMERIC, volume BIGINT)")
        conn.commit()

    log_message("Stock data hour table created successfully")


def check_if_table_exists(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
        return cursor.fetchone()[0]


if __name__ == '__main__':
    pg_conn = PGConn()
