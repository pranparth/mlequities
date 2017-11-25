import pandas as pd
import numpy as np
import math

prices = pd.read_csv("prices-split-adjusted.csv", parse_dates=['date'], index_col='date')
prices = prices['2016']
RISK_FREE_RATE = 1.0033

def returns(ticker):

    assert ticker in prices.symbol.values, "fuck you, I don't have data for this ticker"
    df = prices.loc[prices['symbol'] == ticker]
    x = 1
    returns = []
    while x<=12:
        df1 = df["2016-"+str(x)]
        closing_prices = df1.close.values
        returns.append(closing_prices[-1]/closing_prices[0])
        x += 1
    return returns


def sharpe(portfolio):

    portfolio_expected_return = 0
    std_returns =[]
    weighted_std_returns =[]
    excess_return_matrix = []

    for ticker, weight in portfolio.items():

        stock_returns = returns(ticker)

        avg = np.average(stock_returns)
        std = np.std(stock_returns)
        std_returns.append(std)
        weighted_std_returns.append(weight*std)
        portfolio_expected_return += weight*avg
        excess_return_matrix.append(stock_returns-avg)

    excess_return_matrix = np.array(excess_return_matrix).transpose()
    covariance_matrix = np.dot(excess_return_matrix.transpose(),excess_return_matrix)/excess_return_matrix.shape[0]
    std_products = np.outer(std_returns,std_returns)
    correlation_matrix = np.divide(covariance_matrix,std_products)

    weighted_std_returns = np.array(weighted_std_returns)
    a = np.dot(weighted_std_returns, correlation_matrix)
    portfolio_std = math.sqrt(math.sqrt(np.dot(a,weighted_std_returns.transpose())))


    return (portfolio_expected_return-RISK_FREE_RATE)/portfolio_std










