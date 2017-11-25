"""
AUTHOR: Pranav Parthasarathy
        Vedaad Shakib
        Shubhrakanti Ganguly
        Alexander Bondarenko
        Maksim Ivanov
FUNCTION: Combine and standardize all features, perform PCA on these features
"""
import pandas as pd
import numpy as np
import math

NUM_RANKED = 10
RISK_FREE_RATE = 1.0033

constituents = pd.read_csv("constituents.csv")
fundamentals = pd.read_csv("fundamentals.csv")
prices_split_adjusted = pd.read_csv("prices-split-adjusted.csv")
prices_for_expected_return = pd.read_csv("prices-split-adjusted.csv", parse_dates=['date'], index_col='date')
prices_for_expected_return = prices_for_expected_return['2016']
securities = pd.read_csv("securities.csv")
df_price = pd.read_csv("prices.csv")

# sample portfolio
portfolio = {'TDG': 0.50, 'ALK': 0.25, 'K': .25}

fundamentals = fundamentals[['Ticker Symbol']]
universe = set(securities['Ticker symbol']) \
         & set(prices_split_adjusted['symbol']) \
         & set(fundamentals['Ticker Symbol']) \
         & set(constituents['Symbol'])

def historical_volatility(df):
    p = np.array(df.close)
    lr = np.log(p[1:]) - np.log(p[:-1])
    return np.sum(np.square(lr))
    
def covariance(df1, df2):
    date1 = np.array(df1.date)
    date2 = np.array(df2.date)
    start = max(date1[0], date2[0])
    end = min(date1[-1], date2[-1])
    p1 = np.array(df1[(start <= df1.date) & (df1.date <= end)].close)
    p2 = np.array(df2[(start <= df2.date) & (df2.date <= end)].close)
    lr1 = np.log(p1[1:]) - np.log(p1[:-1])
    lr2 = np.log(p2[1:]) - np.log(p2[:-1])
    return np.sum(lr1*lr2)

vol = []
portfolio_prices = {s: prices_split_adjusted[prices_split_adjusted.symbol == s].sort_values(['date'], ascending=[True]) for s in portfolio.keys()}
for s in universe:
    df = prices_split_adjusted[prices_split_adjusted.symbol == s]
    df.sort_values(['date'], ascending=[True], inplace=True)
    vol.append({
        'symbol': s,
        'volatility': historical_volatility(df),
        'added_volatility': sum(w * covariance(df, portfolio_prices[s]) for s, w in portfolio.items()),
    })
vol = pd.DataFrame(vol, columns = ['symbol', 'volatility', 'added_volatility'])


securities = securities[['Ticker symbol', 'GICS Sector', 'GICS Sub Industry']]

def merge(keep_key, dfs_by_key):
    merged = dfs_by_key[0][1].copy()
    for k, df in dfs_by_key[1:]:
        merged = merged.merge(df, left_on=dfs_by_key[0][0], right_on=k, how='inner')
        if k != keep_key:
            del merged[k]
    if dfs_by_key[0][0] != keep_key:
        del merged[dfs_by_key[0][0]]
    return merged

df = merge('symbol', [('symbol', vol), ('Symbol', constituents), ('Ticker symbol', securities)])

volatility_df = df

securities = pd.read_csv("securities.csv")
fundamentals = pd.read_csv("fundamentals.csv")

df_price = df_price[['symbol','close']].drop_duplicates(subset='symbol', keep = "last")
df = pd.DataFrame()
df["Ticker"] = fundamentals["Ticker Symbol"]
df["EPS"] = fundamentals['Earnings Per Share']
df = df.drop_duplicates(subset='Ticker', keep = "last")
df["Industry"] = np.nan
df["Price"] = np.nan

for x in securities.iterrows():
    truth = df["Ticker"]==x[1]['Ticker symbol']
    try:
        df.loc[df.loc[truth]["Ticker"].index.values[0],'Industry'] = x[1]['GICS Sector']
    except:
        pass

for x in df_price.iterrows():
    truth = df["Ticker"]==x[1]['symbol']
    try:
        df.loc[df.loc[truth]["Ticker"].index.values[0],'Price'] = x[1]['close']
    except:
        pass


df = df.drop(df[df.EPS < 0].index)
df["PE"] = round(df['Price']/df['EPS'], 2)
df = df.dropna()

# Calculate industry 

industries_dict = {sector: np.mean(df[df.Industry == sector].PE.values) for sector in df.Industry.unique()}

volatility_pe_df = volatility_df.merge(df, how='inner', left_on=['symbol'], right_on=['Ticker'])


def get_standarized_PE(ticker, pe= None, industry= None):
    if not pe or not industry:
        assert ticker in df.Ticker.values, "Please provide price and Industry"
        data = df.loc[df["Ticker"] == ticker]
        indus = data["Industry"].values[0]
        PE = data["PE"].values[0]
        avg = industries_dict[indus]
        return (PE - avg)/avg
    else:
        assert industry in industries_dict.keyes(), "You're sector needs to be a GICS Sector"
        avg = industries_dict[industry]
        return (pe - avg)/avg

col = []
for ticker in volatility_pe_df['symbol']:
    col.append(get_standarized_PE(ticker))
volatility_pe_df['std_pe'] = np.array(col)


fifty_two_week_df = pd.read_csv("prices-split-adjusted.csv")
fifty_two_week_df['date'] = pd.to_datetime(fifty_two_week_df.date)

#Keep only the data from 2016 (we only need 1 years worth of data)
mask = (fifty_two_week_df['date'] > '2015-12-31')
fifty_two_week_df = fifty_two_week_df.loc[mask]


#Keep only the high and low and merge it all into one final data frame
s1 = fifty_two_week_df.groupby(['symbol'], sort=False)['close'].max()
s2 = fifty_two_week_df.groupby(['symbol'], sort=False)['close'].min()
final = pd.concat([s1, s2], axis=1).reset_index()
final.columns = ['symbol','high','low']

fifty_two_week_df = final.merge(
    volatility_pe_df, how='inner', left_on=['symbol'], right_on=['symbol'])

sector_feature = []
sector_weights = {}
for ticker in portfolio:
    sector = list(fifty_two_week_df[fifty_two_week_df.symbol == ticker]['Sector'])[0]
    if sector in sector_weights:
        sector_weights[sector] += portfolio[ticker]
    else:
        sector_weights[sector] = portfolio[ticker]
for x in fifty_two_week_df.iterrows():
    if x[1]['Sector'] in sector_weights:
        sector_feature.append(sector_weights[x[1]['Sector']])
    else:
        sector_feature.append(0)
fifty_two_week_df['portfolio_sector'] = sector_feature

"""
standardize un-standardized features
"""
vol_mean, vol_std = fifty_two_week_df['volatility'].mean(), fifty_two_week_df['volatility'].std()
high_low_diff = [x/y for x,y in zip(fifty_two_week_df['high'], fifty_two_week_df['low'])]
high_low_mean, high_low_std = pd.Series(high_low_diff).mean(), pd.Series(high_low_diff).std()
sector_mean, sector_std = fifty_two_week_df['portfolio_sector'].mean(), fifty_two_week_df['portfolio_sector'].std()
std_high_low, std_vol, std_sector = [],[],[]
for x in fifty_two_week_df.iterrows():
	std_high_low.append((x[1]['high']/x[1]['low']-high_low_mean)/high_low_std)
	std_vol.append((x[1]['volatility']-vol_mean)/vol_std)
	std_sector.append((x[1]['portfolio_sector']-sector_mean)/sector_std)
    
fifty_two_week_df['std_high_low'] = np.array(std_high_low)
fifty_two_week_df['std_vol'] = np.array(std_vol)
fifty_two_week_df['std_sector'] = np.array(std_sector)

def top_k_stocks(k):
    """
    ::param k number of stocks to be ranked

    Now make features (52 week high/low + volatility + portfolio_sector + std_pe) and 
    perform Latent Factor Analysis
    """
    X = []
    centroid = {}
    for x in fifty_two_week_df.iterrows():
        point = np.array([
            x[1]['std_high_low'], x[1]['std_vol'], x[1]['std_pe'], x[1]['std_sector']])
        if x[1]['symbol'] in portfolio:
            centroid[tuple(point)] = portfolio[x[1]['symbol']]
        X.append(point)
        
    center = [0,0,0,0]
    for pt in centroid:
        for i in range(4):
            center[i] += centroid[pt]*pt[i]
            
    X = np.array(X)
    U, s, V = np.linalg.svd(X, full_matrices=True)
    shifted_center = []
    for i in range(4):
        shifted_center.append(V[-1][i]*(s[-1]) + center[i])
        
    dist = {}
    for i in range(len(X)):
        c_dist = 0
        for j in range(4):
            c_dist += (X[i][j]-shifted_center[j])**2
        dist[i] = c_dist
    dist = sorted(dist, key = lambda x: dist[x])
    rankings = []
    for i in range(NUM_RANKED):
        rankings.append(fifty_two_week_df.iloc[dist[i]])
    rankings_data = pd.DataFrame(rankings)

    return rankings_data

def returns(ticker):

    assert ticker in prices_for_expected_return.symbol.values, "fuck you, I don't have data for this ticker"
    df = prices_for_expected_return.loc[prices_for_expected_return['symbol'] == ticker]
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

def final_ranking(stocks):
    median_weight = sorted(portfolio.values())[len(portfolio.values())//2]
    ranking = {}
    for stock in stocks:
        portfolio[stock] = median_weight
        total_sum = sum(portfolio.values())
        for ticker in portfolio:
            portfolio[ticker] = portfolio[ticker]/total_sum
        ranking[stock] = sharpe(portfolio)
        del portfolio[stock]
    return sorted(ranking.keys(), key = lambda x: -ranking[x])



            
top_stocks = final_ranking(list(top_k_stocks(NUM_RANKED).symbol.values))

print(top_stocks)


