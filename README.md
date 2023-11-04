# glint

Creates two docker containers - SCRAPER & STORER

## SCRAPER

This container download information from the Yahoo Finance stock data API and sends it 
to STORER

## STORER

This container acts as a central repository for storing of finance information. It
receives information from SCRAPER and stores it in a database.