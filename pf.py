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
from src.delta_hedging import delta_hedge
from src.blackscholes import Call


def assign_groups(df):
    """
    Classify per time-to-maturity in weeks
    """
    df['nweeks'] = 0
    floatweek = 1/52
    df['nweeks'][(df['tau'] <= floatweek)] = 1
    df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
    df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
    df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
    df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8
    return df

def analyze_portfolio(dat, week, iv_var_name):
    """
    dat, pd.DataFrame as from main.py
    week, int, week indicator
    iv_var_name, string, variable name of estimated implied volatility: 
        1) 'predicted_iv' for simple regression
        2) 'rookley_predicted_iv' for rookley
    """

    # Not every instrument may have a price for a day. We will have to infer this from other instruments using a vola proxy. 
    # Or maybe just take instruments which have a daily price
    #@Todo: Use Rookley to solve this problem. Or reconstruct vola surface using another method, then price all options around the same time for this.
    # Might as well use Rookley at a specific point of the day
    dat = dat.rename(columns = {'spot': 'index_price'})

    # Min 4 days to maturity
    calls = dat.loc[(dat['is_call'] == 1) & (dat['tau'] * 365 >= 4)]
    print('Calls only')

    # First, use BS Call value function to get Dollar Value for Call Parameters
    calls['instrument_price'] = calls.apply(lambda x: Call.Price(x['index_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    calls['delta'] = calls.apply(lambda x: Call.Delta(x['index_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)

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
        
        sub = calls[calls['instrument_name'] == instrument]
        sub['future_position'] = None
        print(sub.head())

        # Take last price per day
        #last_timestamp_per_day = sub.groupby('time')['timestamp'].max()
        #daily = sub.loc[sub['timestamp'].isin(last_timestamp_per_day)]
        daily = sub
        # Check that we have daily differences
        #pdb.set_trace()
        #only_daily_differences = np.all(daily['date'].dropna().diff().dropna() < timedelta(days = 2))

        #if not only_daily_differences:
        #    continue;
        #pdb.set_trace()
        # Delta Hedge
        daily['future_position'] = daily['delta'] * daily['index_price']

        # Hedge PnL
        n_shares, cost_of_shares, cumulative_cost, interest_cost = delta_hedge(daily['delta'].to_list(), daily['index_price'].to_list(), 0.01, daily['tau'].iloc[0])
        delta_hedge_cost = cumulative_cost[-1]

        initial_instrument_price = daily.iloc[0]['instrument_price']
        final_instrument_price = daily.iloc[-1]['instrument_price']

        dailies[instrument] = daily
        # @Todo also have to consider the payoff of the long call here!!
        # Delta hedge cost is negative if we made a profit in the hedge position!
        #pnl[instrument] = final_instrument_price - initial_instrument_price - delta_hedge_cost #+ daily['hedge_pnl'].sum()

        # Relative to initial price
        pnl[instrument] = (final_instrument_price - initial_instrument_price - delta_hedge_cost) / initial_instrument_price

        #counter += 1
        #if counter >= 1000:
        #    pdb.set_trace()

    #pdb.set_trace()
    pnl_df = pd.DataFrame({'pnl': pnl})
    pnl_df.to_csv('out/pnl_df' + iv_var_name + '_week=' + str(week) + '.csv')
    print(pnl_df.describe())

    # @Todo: PnL per Group: First time-to-maturity, moneyness. Also show over time. 
    # Compare to Initial IV and Difference between Initial IV to running average. 
    # Check Jackwerth

    return pnl_df

if __name__ == '__main__':

    dat = pd.read_csv('out/fitted_data.csv')#pd.read_csv('data/option_transactions.csv')
    dat['date'] = pd.to_datetime(dat['day'])
    dat = assign_groups(dat)
    
    dat.sort_values('day', inplace = True)

    # Assess difference between Rookley and simple Regression estimated IV
    dat['ivdiff_abs'] = dat['rookley_predicted_iv'] - dat['predicted_iv']
    dat['ivdiff_rel'] = dat['ivdiff_abs'] / dat['rookley_predicted_iv']
    dat[['ivdiff_abs', 'ivdiff_rel']].describe()

    # Find Outliers
    dat[['rookley_predicted_iv', 'predicted_iv']].describe()

    max_iv = 2.5
    min_iv = 0
    iv_vars = ['rookley_predicted_iv', 'predicted_iv']

    for iv_var in iv_vars:
        dat.loc[dat[iv_var] >= max_iv, iv_var] = max_iv 
        dat.loc[dat[iv_var] <= min_iv, iv_var] = min_iv
    #@Todo: Set bounds for prediction in the actual prediction
    pdb.set_trace()
    pnl_per_group = {}
    for week in dat['nweeks'].unique():
        print('Week Group: ', week)
        df = dat[dat['nweeks'] == week]
        df.sort_values('day', inplace = True)
        pnl_per_group[week] = analyze_portfolio(df, week, 'predicted_iv')
        
    for key, val in pnl_per_group.items():
        print('Week: ', key)
        print(val.describe())
        