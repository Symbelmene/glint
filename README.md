# glint

Creates two docker containers - SCRAPER & STORER

## SCRAPER

This container download information from the Yahoo Finance stock data API and sends it 
to STORER

### Steps
- Upon first run, the container will download the stock data the specified sector for as far back as yfinance allows for the given time interval
- Once the data has been downloaded, tickers that data has not been sucessfully downloaded will be deleted from the database.
- The container will then wait for the next interval to download the data again.


## STORER

This container acts as a central repository for storing of finance information. It
receives information from SCRAPER and stores it in a database.