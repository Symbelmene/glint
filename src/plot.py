import numpy as np
import plotly.graph_objects as go

import utils


def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


def ichimoku(df, ticker=''):
    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

    df1 = df.copy()
    fig = go.Figure()
    df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()

    df = df.groupby('group')

    dfs = []
    for name, data in df:
        dfs.append(data)

    for df in dfs:
        fig.add_traces([go.Scatter(x=df.index, y=df.SpanA,
                                  line=dict(color='rgba(0,0,0,0)'))])

        fig.add_traces([go.Scatter(x=df.index, y=df.SpanB,
                                  line=dict(color='rgba(0,0,0,0)'),
                                  fill='tonexty',
                                  fillcolor=get_fill_color(df['label'].iloc[0]))])

    baseline = go.Scatter(x=df1.index, y=df1['Baseline'], line=dict(color='pink', width=2), name="Baseline")
    conversion = go.Scatter(x=df1.index, y=df1['Conversion'], line=dict(color='black', width=1), name="Conversion")
    lagging = go.Scatter(x=df1.index, y=df1['Lagging'], line=dict(color='purple', width=2), name="Lagging")
    span_a = go.Scatter(x=df1.index, y=df1['SpanA'], line=dict(color='green', width=2, dash='dot'), name="Span A")
    span_b = go.Scatter(x=df1.index, y=df1['SpanB'], line=dict(color='red', width=1, dash='dot'), name="Span B")

    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    fig.update_layout(title=ticker + " Ichimoku",
                      height=1200, width=1800, showlegend=True)

    fig.show()


def bollingerBands(df, ticker=''):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df['Low'],
                            close=df['Close'], name="Candlestick")

    upper_line = go.Scatter(x=df.index, y=df['upper_band'],
                            line=dict(color='rgba(250, 0, 0, 0.75)',
                                      width=1), name="Upper Band")

    mid_line = go.Scatter(x=df.index, y=df['middle_band'],
                          line=dict(color='rgba(0, 0, 250, 0.75)',
                                    width=0.7), name="Middle Band")

    lower_line = go.Scatter(x=df.index, y=df['lower_band'],
                            line=dict(color='rgba(0, 250, 0, 0.75)',
                                      width=1), name="Lower Band")

    fig.add_trace(candle)
    fig.add_trace(upper_line)
    fig.add_trace(mid_line)
    fig.add_trace(lower_line)

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    # USED FOR NON-DAILY DATA : Get rid of empty dates and market closed
    # fig.update_layout(title=ticker + " Bollinger Bands",
    # height=1200, width=1800,
    #               showlegend=True,
    #               xaxis_rangebreaks=[
    #         dict(bounds=["sat", "mon"]),
    #         dict(bounds=[16, 9.5], pattern="hour"),
    #         dict(values=["2021-12-25", "2022-01-01"])
    #     ])

    fig.update_layout(title=ticker + " Bollinger Bands",
                      height=1200, width=1800, showlegend=True)
    fig.show()


def plotStockTickersInPeriod(colName, tickers, interval, sTime, eTime):
    df = utils.loadMultipleDFsAndMergeByColumnName(colName, sTime, eTime, interval, tickers)

    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title=colName)
    fig.update_layout(height=800, width=1600,
                      showlegend=True)
    fig.show()
