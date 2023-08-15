"""
Construct Portfolio of a Long Call and a Delta Hedge
Rebalanced Daily
Analyze PnL 

Then go for Put, Call Spread, Put Spread, Straddle

So for each instrument, observe over time and adjust Portfolio

Use associated Future for risk-free rate
Get risk-free rate for the time of the instrument!
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import pdb
from delta_hedging import delta_hedge
from scipy import stats


dat = pd.read_csv('data/option_transactions.csv')
dat['date'] = pd.to_datetime(dat['time'])
dat.sort_values('timestamp', inplace = True)

# Not every instrument may have a price for a day. We will have to infer this from other instruments using a vola proxy. 
# Or maybe just take instruments which have a daily price
#@Todo: Use Rookley to solve this problem. Or reconstruct vola surface using another method, then price all options around the same time for this.
# Might as well use Rookley at a specific point of the day

calls = dat.loc[(dat['is_call'] == 1) & (dat['days_to_maturity'] > 4)]
print('Calls only')

# Not yet!! Must force all diffs to be 1, cant skip any!!
#date_diffs = calls.groupby('instrument_name')['date'].diff()
#idx = date_diffs[date_diffs == timedelta(1)].index
#consecutive_date_calls = calls.loc[idx]
#calls.groupby(['instrument_name'])['date'].diff()
#calls['date'].diff().dt.days.ne(1).cumsum()

dailies = {}
pnl = {}
counter = 0
for instrument in calls['instrument_name'].unique():
    
    sub = dat[dat['instrument_name'] == instrument]
    sub['future_position'] = None
    print(sub.head())

    # Take last price per day
    last_timestamp_per_day = sub.groupby('time')['timestamp'].max()
    daily = sub.loc[sub['timestamp'].isin(last_timestamp_per_day)]

    # Check that we have daily differences
    #pdb.set_trace()
    only_daily_differences = np.all(daily['date'].dropna().diff().dropna() < timedelta(days = 2))

    if not only_daily_differences:
        continue;

    # Delta Hedge
    daily['future_position'] = daily['delta'] * daily['index_price']

    # Hedge PnL
    #@Todo Check the PnL!!!
    """
    # This should be a more elegant method
    # cash flow from delta hedging
    # compute cash generated from delta re-hedging
    delta_chgs = np.diff(deltas, axis=1)
    delta_rehedge_cfs = -delta_chgs * pxs[:,1:] # pxs are prices
    """
    daily['index_change'] = daily['index_price'].pct_change()
    #daily['future_position_change'] = daily['future_position'].diff()
    #daily['future_pnl'] = daily['future_position_change'] * daily['index_change'] * (-1)
    #daily['future_pnl'].sum()

    # PnL in the hedge position
    # dailies['BTC-8JAN21-28000-C']
    # For above instrument, the initial futures position is 20164. The change in underlying
    # on the next day is 5.71%. Therefore the hedge position has made 1152 on day 2
    daily['hedge_pnl'] = daily['future_position'].shift(1) * daily['index_change']


    # another try via cashflow
    # But then you also have to unravel the cashflow of the last day, so shift and add a zero
    # @Todo: This is missing an unraveling on the last day, where the diff is 0
    #daily['delta_diffs'] = daily['delta'].diff()
    #daily['hedge_cashflow'] = (-1) * daily['delta_diffs'] * daily['index_price']
    n_shares, cost_of_shares, cumulative_cost, interest_cost = delta_hedge(daily['delta'].to_list(), daily['index_price'].to_list(), 0.01, daily['tau'].iloc[0])
    delta_hedge_cost = cumulative_cost[-1]
    pdb.set_trace()

    initial_instrument_price = daily.iloc[0]['instrument_price']
    final_instrument_price = daily.iloc[-1]['instrument_price']

    dailies[instrument] = daily
    # @Todo also have to consider the payoff of the long call here!!
    pnl[instrument] = final_instrument_price - initial_instrument_price + delta_hedge_cost #+ daily['hedge_pnl'].sum()

    #counter += 1
    #if counter >= 1000:
    #    pdb.set_trace()

pdb.set_trace()
pnl_df = pd.DataFrame({'pnl': pnl})
pnl_df.to_csv('out/pnl_df.csv')
print(pnl_df.describe())

print('done')