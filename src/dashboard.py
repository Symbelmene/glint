import json
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from plot import bollingerBands, ichimoku
from finclasses import Ticker, Portfolio, loadTickers
from config import Config
cfg = Config()

tickerDict = {}
dfSharpe = pd.DataFrame()


def prepSharpeTable(tickers):
    startTime = pd.to_datetime('2020-01-01')
    tickerList = []
    for name, ticker in tqdm(tickers.items()):
        ticker.slice(start=startTime)
        if ticker.data.index.max() - ticker.data.index.min() < timedelta(days=100):
            continue

        try:
            tickerList.append({
                'Ticker': ticker.name,
                'Return': ticker.calc_return(),
                'Volatility': ticker.calc_volatility(),
                'Sharpe': ticker.calc_sharpe()})
        except Exception:
            print(f'Could not analyse stock {name}. Skipping...')
            continue
    return tickerList


def graphicalShortList(maxVol=5.0, minRet=0.0, maxRet=10.0):
    df = dfSharpe
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
    return fig


def dashboard():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    fig = graphicalShortList(maxVol=2, minRet=0.5, maxRet=4)
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=5)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='sharpe-plot',
                figure=fig,
                hoverData={'points': [{'customdata': [None]}]}
            ),
        ], style={'display': 'inline-block', 'width': '49%'}),

        html.Div([
            #dcc.Graph(id='current-ticker-bollinger'),
            dcc.Graph(id='current-ticker-ichimoku'),
        ], style={'display': 'inline-block', 'width': '49%'}),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown("""
                    **Hover Data**
                """),
                html.Pre(id='hover-data', style=styles['pre']),
            ], className='three columns'),
        ])
    ])

    @app.callback(
        Output('hover-data', 'children'),
        Input('sharpe-plot', 'hoverData'))
    def display_hover_data(hoverData):
        return json.dumps(hoverData, indent=2)

    #@app.callback(
    #    Output('current-ticker-bollinger', 'figure'),
    #    Input('sharpe-plot', 'hoverData'))
    #def drawTickerBollinger(hoverData):
    #    if hoverData == None:
    #        tickerName = list(tickerDict.keys())[0]
    #    else:
    #        tickerName = hoverData['points'][0]['customdata'][0]
    #    currTicker = tickerDict[tickerName]
    #    currTicker.preprocess()
    #    fig = bollingerBands(currTicker.data, ticker=tickerName)
    #    return fig

    @app.callback(
        Output('current-ticker-ichimoku', 'figure'),
        Input('sharpe-plot', 'hoverData'))
    def drawTickerIchimoku(hoverData):
        if hoverData == None:
            tickerName = list(tickerDict.keys())[0]
        else:
            tickerName = hoverData['points'][0]['customdata'][0]
        currTicker = tickerDict[tickerName]
        currTicker.preprocess()
        fig = ichimoku(currTicker.data, ticker=tickerName)
        return fig

    app.run_server(debug=True)


def main():
    global tickerDict, dfSharpe
    tickerDict = loadTickers(cfg.DATA_DIR_24_HOUR)
    dfSharpe   = pd.DataFrame(prepSharpeTable(tickerDict))

    dashboard()

    # TODO Display linked graphs of scatter and stock candle
    # TODO Be able to change volatility and return in the dashboard
    # TODO Allow selection of multiple tickers
    # TODO Add button to auto-optimise portfolio and display weights
    # TODO Dockerise the whole shebang
