import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from config import Config
cfg = Config()


def createTables(db):
    cursor = db.cursor()
    # Create empty database tables
    # SECTOR TABLE - sector_id, sector
    cursor.execute("CREATE TABLE sectors ("
                   "sector_id INT AUTO_INCREMENT PRIMARY KEY, "
                   "sector VARCHAR(255))")

    # TICKER TABLE - ticker_id, sector_id, ticker
    cursor.execute("CREATE TABLE tickers (ticker_id INT AUTO_INCREMENT PRIMARY KEY,"
                   "sector_id INT, "
                   "ticker VARCHAR(255),"
                   "FOREIGN KEY (sector_id) REFERENCES sectors(sector_id))")

    # STOCK DATA TABLES - ticker_id, date, open, high, low, close, volume
    cursor.execute("CREATE TABLE stock_data_day (ticker_id INT, date DATE, open FLOAT, high FLOAT, low FLOAT, "
                     "close FLOAT, volume BIGINT UNSIGNED, FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id))")

    # STOCK DATA TABLES - ticker_id, date, open, high, low, close, volume
    cursor.execute("CREATE TABLE stock_data_5min (ticker_id INT, date DATETIME, open FLOAT, high FLOAT, low FLOAT, "
                   "close FLOAT, volume BIGINT UNSIGNED, FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id))")
    db.commit()


def dropAllTables(db):
    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS stock_data_day")
    cursor.execute("DROP TABLE IF EXISTS stock_data_5min")
    cursor.execute("DROP TABLE IF EXISTS tickers")
    cursor.execute("DROP TABLE IF EXISTS sectors")
    db.commit()


def recreateTables(mydb):
    dropAllTables(mydb)
    createTables(mydb)


def addSectorData(mydb, sectorList):
    cursor = mydb.cursor()
    for sector in sectorList:
        cursor.execute(f"INSERT INTO sectors (sector) VALUES ('{sector}')")
    mydb.commit()


def getSectorTable(mydb):
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM sectors")
    return cursor.fetchall()


def addTickerData(mydb, df):
    cursor = mydb.cursor()
    df.dropna(inplace=True)
    for idx, row in df.iterrows():
        sector = row["Sector"]
        ticker = row["Ticker"]
        cursor.execute(f"SELECT sector_id FROM sectors WHERE sector = '{sector}'")
        sector_id = cursor.fetchone()[0]
        cursor.execute(f"INSERT INTO tickers (sector_id, ticker) VALUES ({sector_id}, '{ticker}')")
    mydb.commit()


def addTickerDataToDatabaseDay(mydb, path):
    df, ticker = parseTickerDataFromFile(path)
    cursor = mydb.cursor()
    df.dropna(inplace=True)
    cursor.execute(f"SELECT ticker_id FROM tickers WHERE ticker = '{ticker}'")
    ticker_id = cursor.fetchone()[0]
    values = [(ticker_id, row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]) for idx, row in df.iterrows()]
    cursor.executemany(f"INSERT INTO stock_data_day (ticker_id, date, open, high, low, close, volume) "
                          f"VALUES (%s, %s, %s, %s, %s, %s, %s)", values)
    mydb.commit()

def addTickerDataToDatabase5Min(mydb, path):
    df, ticker = parseTickerDataFromFile(path)
    cursor = mydb.cursor()
    df.dropna(inplace=True)
    cursor.execute(f"SELECT ticker_id FROM tickers WHERE ticker = '{ticker}'")
    ticker_id = cursor.fetchone()[0]
    values = [(ticker_id, row["Datetime"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]) for idx, row in df.iterrows()]
    cursor.executemany(f"INSERT INTO stock_data_5min (ticker_id, date, open, high, low, close, volume) "
                          f"VALUES (%s, %s, %s, %s, %s, %s, %s)", values)
    mydb.commit()


def parseTickerDataFromFile(path):
    df = pd.read_csv(path)
    ticker = path.split('/')[-1].split('.')[0].replace('_', '.')
    return df, ticker


def loadAllTickerData(tickerDir):
    tickerData = [f'{cfg.DATA_DIR_24_HOUR}/{tickerFile}' for tickerFile in os.listdir(tickerDir)[:30]]
    with Pool(cfg.PROCESSES) as p:
        tickerData = list(tqdm(p.imap(parseTickerDataFromFile, tickerData), total=len(tickerData)))
    return tickerData


def logErrorMessage(ticker, ex):
    fileMode = 'w+' if not os.path.exists(f'{cfg.LOG_DIR}/sql_error.log') else 'a'
    with open(f'{cfg.LOG_DIR}/sql_error.log', 'a') as f:
        f.write(f'{ticker} - {ex}\n')


def main():
    mydb = mysql.connector.connect(
        host="localhost",
        user="user",
        password="password",
        database="findata"
    )

    df = pd.read_csv("../big_stock_sectors.csv")
    # Recreate empty tables
    dropAllTables(mydb)
    createTables(mydb)

    # Add sector and ticker key tables
    sectorList = list(df['Sector'].dropna().unique())
    addSectorData(mydb, sectorList)
    addTickerData(mydb, df)

    # Add 24H ticker data to database
    for tickerFile in tqdm(os.listdir(cfg.DATA_DIR_24_HOUR)):
        try:
            addTickerDataToDatabaseDay(mydb, f'{cfg.DATA_DIR_24_HOUR}/{tickerFile}')
        except Exception as ex:
            print(f'ERROR: {tickerFile} failed to load ({ex})')
            logErrorMessage(tickerFile, ex)

    # Add ticker data to database
    for tickerFile in tqdm(os.listdir(cfg.DATA_DIR_5_MINUTE)):
        try:
            addTickerDataToDatabase5Min(mydb, f'{cfg.DATA_DIR_5_MINUTE}/{tickerFile}')
        except Exception as ex:
            print(f'ERROR: {tickerFile} failed to load ({ex})')
            logErrorMessage(tickerFile, ex)

    cursor = mydb.cursor()
    cursor.execute('SELECT table_schema AS "Database", SUM(data_length + index_length) / 1024 / 1024 / 1024 AS "Size (GB)" FROM information_schema.TABLES GROUP BY table_schema')
    print(cursor.fetchall())


if __name__ == '__main__':
    main()