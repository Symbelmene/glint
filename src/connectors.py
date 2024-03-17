import pandas as pd
import psycopg2 as pg


class PGConn:
    def __init__(self):
        self.conn = pg.connect(dbname='findata',
                               user='user',
                               password='pass',
                               host='0.0.0.0',
                               port='5432')

        # Check if sector and ticker tables exist and create them if they don't
        self.populate_initial_tables()

        # Create empty stock data table if it doesn't exist

    def populate_initial_tables(self):
        populate_base_tables(self.conn)
        create_stock_data_table(self.conn)

    def get_tickers_for_sector(self, sector_name):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT t.ticker FROM tickers t JOIN sectors s ON t.sector_id = s.id WHERE s.sector = %s",
                           (sector_name,))
            return [row[0] for row in cursor.fetchall()]

    def insert_stock_data(self, ticker, data):
        insert_query = f"INSERT INTO stock_data (date, ticker, open, high, low, close, adj_close, volume) " \
                       f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        inserts = [(row.name.date(), ticker, row['Open'], row['High'],
                    row['Low'], row['Close'], row['Adj Close'], int(row['Volume']))
                   for idx, row in data.iterrows()]

        with self.conn.cursor() as cursor:
            cursor.executemany(insert_query, inserts)

        # Commit the changes to the database
        self.conn.commit()

    def __del__(self):
        self.conn.close()


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

    print("Sectors table populated successfully")

    # Populate ticker table
    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE tickers (id SERIAL PRIMARY KEY, ticker VARCHAR(10) UNIQUE, sector_id INTEGER)")
        conn.commit()

        for idx, row in df.iterrows():
            ticker = row['Ticker']
            sector_id = sector_dict[row['Sector']]
            cursor.execute("INSERT INTO tickers (ticker, sector_id) VALUES (%s, %s)", (ticker, sector_id,))
            conn.commit()

    print("Ticker table populated successfully")


def create_stock_data_table(conn):
    if check_if_table_exists(conn, 'stock_data'):
        return True

    with conn.cursor() as cursor:
        cursor.execute("CREATE TABLE stock_data (id SERIAL PRIMARY KEY, date DATE, ticker VARCHAR(10), open NUMERIC, "
                       "high NUMERIC, low NUMERIC, close NUMERIC, adj_close NUMERIC, volume BIGINT)")
        conn.commit()

    print("Stock data table created successfully")


def check_if_table_exists(conn, table_name):
    with conn.cursor() as cursor:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
        return cursor.fetchone()[0]


if __name__ == '__main__':
    pg_conn = PGConn()
    pg_conn.populate_ticker_table()
