import os
import pandas as pd
from tqdm import tqdm
import plotly.express as px

from finclasses import Ticker, Portfolio


def graphicalShortList(path, maxVol=5.0, minRet=0.0, maxRet=10.0):
    tickerList = []
    startTime = pd.to_datetime('2020-01-01')
    for tickerPath in tqdm(os.listdir(path)):
        ticker = Ticker(path + '/' + tickerPath)
        ticker.slice(start=startTime)
        tickerList.append({
            'Ticker': ticker.name,
            'Return': ticker.calc_return(),
            'Volatility': ticker.calc_volatility(),
            'Sharpe': ticker.calc_sharpe()})

    df = pd.DataFrame(tickerList)
    df = df[df['Volatility'] < maxVol]
    df = df[df['Return'] > minRet]
    df = df[df['Return'] < maxRet]

    fig = px.scatter(df, x=df['Volatility'], y=df['Return'],
                     custom_data=['Ticker', 'Return', 'Volatility'])
    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.update_traces(
        hovertemplate="<br>".join([
            "Col1: %{customdata[0]}",
            "Col2: %{customdata[1]}",
            "Col3: %{customdata[2]}",
        ])
    )
    fig.show()


def main():
    dbPath = 'D:/findata/data/24_HOUR'

    graphicalShortList(dbPath,
                       maxVol=2,
                       minRet=0.5,
                       maxRet=4)

    # TODO Display linked graphs of scatter and stock candle
    # TODO Be able to change volatility and return in the dashboard
    # TODO Allow selection of multiple tickeres
    # TODO Add button to auto-optimise portfolio and display weights
    # TODO Dockerise the whole shebang


if __name__ == '__main__':
    main()