import pandas as pd
import psycopg2 as pg

from debug import log_message
from config import Config
cfg = Config()


class PGConn:
    def __init__(self):
        self.conn = pg.connect(dbname='postgres',
                               user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                               host=cfg.STORER_HOST, port=cfg.STORER_PORT)

        # Check if findata database exists and create it if it doesn't
        if not self.check_database_exists(cfg.STORER_DB_NAME):
            self.create_database(cfg.STORER_DB_NAME)
        self.conn.close()

        self.conn = pg.connect(dbname=cfg.STORER_DB_NAME,
                               user=cfg.STORER_USER, password=cfg.STORER_PASSWORD,
                               host=cfg.STORER_HOST, port=cfg.STORER_PORT)

        # Check if sector and ticker tables exist and create them if they don't
        self.populate_initial_tables()

        # Create empty stock data table if it doesn't exist

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
        create_stock_data_table(self.conn)

    def get_tickers_for_sector(self, sector_name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT t.id, t.ticker FROM tickers t JOIN sectors s ON t.sector_id = s.id WHERE s.sector = %s",
                           (sector_name,))
            return [row[1] for row in cursor.fetchall()]

    def insert_stock_data(self, ticker, data):
        insert_query = f"INSERT INTO stock_data (date, ticker, open, high, low, close, adj_close, volume) " \
                       f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        inserts = [(pd.to_datetime(row.name), ticker, row['Open'], row['High'],
                    row['Low'], row['Close'], row['Adj Close'], int(row['Volume']))
                   for idx, row in data.iterrows()]

        with self.conn.cursor() as cursor:
            cursor.executemany(insert_query, inserts)

        # Commit the changes to the database
        self.conn.commit()


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


def create_stock_data_table(conn):
    if check_if_table_exists(conn, 'stock_data'):
        return True

    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE stock_data (id SERIAL PRIMARY KEY, date DATETIME, ticker VARCHAR(10), open NUMERIC, "
                       "high NUMERIC, low NUMERIC, close NUMERIC, adj_close NUMERIC, volume BIGINT)")
        conn.commit()

    log_message("Stock data table created successfully")


def check_if_table_exists(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
        return cursor.fetchone()[0]


if __name__ == '__main__':
    pg_conn = PGConn()
