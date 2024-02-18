import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import pdb

def plot_funding(funding, xvar, yvar1, yvar2, label1, label2, fname, ylim):
    """

    """
    fig = plt.figure(figsize = (12,8))        
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #plt.gcf().autofmt_xdate()
    plt.plot(funding[xvar], funding[yvar1], label = label1, color = 'blue') # blue #1fzzb4
    plt.plot(funding[xvar], funding[yvar2], label = label2, color = 'orange') # #ff7f02
    #plt.legend()
    if ylim:
        plt.ylim(ylim)
    plt.savefig(fname, transparent = True)

def annual_interest(full_day_funding_rate):
    """
    funding: pd.DataFrame
    The rate is already a daily average, so 24h instead of 8h based
    """
    interest_path =  np.cumprod(1 + full_day_funding_rate) - 1
    return interest_path.iloc[-1]


funding = pd.read_csv('funding.csv')
funding['date'] = pd.to_datetime(funding['date'])



# Start of Funding Rates in the data. 
funding = funding.loc[funding['date'] >= '2019-05-01']

# Create a table that summarizes the annual funding rates
# Since the perpetual future data exists from the start date in 2019, 
# 2019 only offers the returns for half a year! So we have to adjust for a full year
funding['year'] = funding['date'].apply(lambda x: x.year)
btc_funding_rate_per_year = funding.groupby('year').apply(lambda x: annual_interest(x['funding_btc']))
eth_funding_rate_per_year = funding.groupby('year').apply(lambda x: annual_interest(x['funding_eth']))
df_funding_rate_per_year = pd.DataFrame({'btc': btc_funding_rate_per_year, 'eth': eth_funding_rate_per_year})

annualization_factor = 12/4
df_funding_rate_per_year.loc[df_funding_rate_per_year.index == 2019] = df_funding_rate_per_year.loc[df_funding_rate_per_year.index == 2019].apply(lambda x: (1 + x)**(annualization_factor) - 1)
df_funding_rate_per_year.to_csv('btc/out/funding_rate_per_year.csv')

pdb.set_trace()
plot_funding(funding, 'date', 'funding_btc', 'funding_eth', 'BTC', 'ETH', 'plots/funding_btc_vs_eth.png', None)
plot_funding(funding, 'date', 'btc_annualized', 'eth_annualized', 'BTC', 'ETH', 'plots/funding_btc_vs_eth_annualized.png', None)

# With ylim - zoom in 
plot_funding(funding, 'date', 'funding_btc', 'funding_eth', 'BTC', 'ETH', 'plots/funding_btc_vs_eth_zoom.png', (-0.005, 0.005))
plot_funding(funding, 'date', 'btc_annualized', 'eth_annualized', 'BTC', 'ETH', 'plots/funding_btc_vs_eth_annualized_zoom.png', (-2, 2))

