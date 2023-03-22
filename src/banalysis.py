import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import plotly.express as px
import utils
import preprocess
from config import Config, Interval
cfg = Config()

import warnings
warnings.simplefilter("ignore")


def mergeDfByColumn(col_name, sdate, edate, *tickers):
    # Will hold data for all dataframes with the same column name
    mult_df = pd.DataFrame()
    for ticker in tickers:
        df = utils.loadStockData(ticker, Interval.DAY)
        df = preprocess.addBaseIndicatorsToDf(df)
        mask = (df.index >= pd.to_datetime(sdate)) & (df.index <= pd.to_datetime(edate))
        mult_df[ticker] = df.loc[mask][col_name]
    return mult_df


def plotStockData(eDate, mult_df, sDate):
    # Plot out prices for each stock
    print('Plotting stock prices in selected date...')
    fig = px.line(mult_df, x=mult_df.index, y=mult_df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.show()

    mult_cum_df = mergeDfByColumn('cum_return', sDate, eDate, *portList)
    # Plot out cumulative returns for each stock since beginning of 2017
    print('Plotting cumulative returns in selected date...')
    fig = px.line(mult_cum_df, x=mult_cum_df.index, y=mult_cum_df.columns)
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.show()


def optimisePortfolio(portfolioList, sDate, eDate, plot=False):
    num_stocks = len(portfolioList)
    print('Loading and preprocessing data...')
    mult_df = mergeDfByColumn('Close', sDate, eDate, *portfolioList)

    if plot:
        plotStockData(eDate, mult_df, sDate)

    returns = np.log(mult_df / mult_df.shift(1))
    mean_ret = returns.mean() * 252  # 252 average trading days per year

    # Generate 11 random values that sum to 1
    print('Plotting Pareto front...')
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    # Provide return of portfolio using random weights over the whole dataset

    p_ret = []  # Returns list
    p_vol = []  # Volatility list
    p_SR  = []  # Sharpe Ratio list
    p_wt  = []  # Stock weights list
    retList = []
    for _ in tqdm(range(10000)):
        # Generate random weights
        p_weights = np.random.random(num_stocks)
        p_weights /= np.sum(p_weights)

        # Add return using those weights to list
        ret_1 = np.sum(p_weights * returns.mean()) * 252
        p_ret.append(ret_1)

        # Add volatility or standard deviation to list
        vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
        p_vol.append(vol_1)

        # Get Sharpe ratio
        SR_1 = (ret_1 - cfg.RISK_FREE_RATE) / vol_1
        p_SR.append(SR_1)

        # Store the weights for each portfolio
        p_wt.append(p_weights)

        retDict = {'return' : ret_1, 'volatility' : vol_1, 'sharpe_ratio': SR_1, 'weights': p_weights}
        retList.append(retDict)

    # Convert to Numpy arrays
    p_ret = np.array(p_ret)
    p_vol = np.array(p_vol)
    p_SR  = np.array(p_SR)
    p_wt  = np.array(p_wt)

    # Create a dataframe with returns and volatility
    ports = pd.DataFrame({'Return': p_ret, 'Volatility': p_vol})

    if plot:
        print('Plotting Pareto Front...')
        fig = px.scatter(ports, x=ports['Volatility'], y=ports['Return'])
        fig.update_layout(height=800, width=1200, showlegend=True)
        fig.show()

    # Return the index of the largest Sharpe Ratio
    SR_idx = np.argmax(p_SR)

    # Find the ideal portfolio weighting at that index
    idealPortfolioWeighting = [p_wt[SR_idx][i] * 100 for i in range(num_stocks)]

    return p_vol[SR_idx], p_ret[SR_idx], idealPortfolioWeighting


def getPortFolioShares(one_price, force_one, wts, prices):
    # Gets number of stocks to analyze
    num_stocks = len(wts)

    # Holds the number of shares for each
    shares = []

    # Holds Cost of shares for each
    cost_shares = []

    i = 0
    while i < num_stocks:
        # Get max amount to spend on stock
        max_price = one_price * wts[i]

        # Gets number of shares to buy and adds them to list
        num_shares = int(max_price / prices[i])

        # If the user wants to force buying one share do it
        if (force_one & (num_shares == 0)):
            num_shares = 1

        shares.append(num_shares)

        # Gets cost of those shares and appends to list
        cost = num_shares * prices[i]
        cost_shares.append(cost)
        i += 1

    return shares, cost_shares


def getPortfolioWeighting(share_cost):
    # Holds weights for stocks
    stock_wts = []
    # All values summed
    tot_val = sum(share_cost)
    print("Total Investment :", tot_val)

    for x in share_cost:
        stock_wts.append(x / tot_val)
    return stock_wts


def getPortfolioValueByDate(date, shares, tickers):
    port_prices = mergeDfByColumn('Close', date, date, *portList)
    # Convert from dataframe to Python list
    port_prices = port_prices.values.tolist()
    # Trick that converts a list of lists into a single list
    port_prices = sum(port_prices, [])
    return port_prices


def markowitzPortfolioOptimisation(portList):
    S_DATE = '2019-01-04'
    E_DATE = '2023-01-28'

    portWeights = [7, 8, 15, 14, 3, 3, 17, 6, 11, 14, 1]

    volatility, annualReturn, sharpeWeights = optimisePortfolio(portList, S_DATE, E_DATE, plot=False)

    # Get all stock prices on the starting date
    port_df_start = mergeDfByColumn('Close', '2022-01-07', '2022-01-07', *portList)

    # Convert from dataframe to Python list
    port_prices = port_df_start.values.tolist()

    # Trick that converts a list of lists into a single list
    port_prices = sum(port_prices, [])

    tot_shares, share_cost = getPortFolioShares(105.64, True, portWeights, port_prices)

    # Get list of weights for stocks
    stock_wts = getPortfolioWeighting(share_cost)

    # Get value at end of year
    portFolioPrices = getPortfolioValueByDate(E_DATE, tot_shares, portList)

    portDict = {'tickers'      : portList,
                'weights'      : portWeights,
                'prices'       : port_prices,
                'distribution' : stock_wts,
                'sharpeweights': sharpeWeights,
                'shares'       : tot_shares}

    portDf = pd.DataFrame(portDict)

    print(tabulate(portDf, headers='keys', tablefmt='psql'))
    print(f'Volatility    : {round(volatility, 3)}')
    print(f'Annual Return : {round(annualReturn, 3)}')


if __name__ == '__main__':
    portList = ['CALX', 'NOVT', 'RGEN', 'LLY',
                'AMD', 'NFLX', 'COST', 'BJ', 'WING',
                'MSCI', 'CBRE']
    markowitzPortfolioOptimisation(portList)