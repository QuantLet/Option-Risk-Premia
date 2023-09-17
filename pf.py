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
import matplotlib.pyplot as plt
from datetime import timedelta
import pdb
from src.delta_hedging import delta_hedge
from src.blackscholes import Call
from src.helpers import assign_groups, load_expiration_price_history, compute_vola


def analyze_portfolio(dat, week, iv_var_name, expiration_history):
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

    # Introduce expiration prices
    #dat = dat.merge(expiration_history, on ='Date')
    
    # Min 4 days to maturity
    #calls = dat.loc[(dat['is_call'] == 1) & (dat['tau'] * 365 >= 2)]
    calls = dat.loc[(dat['is_call'] == 1)]
    print('Calls only')

    calls['instrument_price_on_expiration'] = calls.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0), axis = 1)

    # First, use BS Call value function to get Dollar Value for Call Parameters
    calls['instrument_price'] = calls.apply(lambda x: Call.Price(x['index_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    calls['delta'] = calls.apply(lambda x: Call.Delta(x['index_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)

    # Not yet!! Must force all diffs to be 1, cant skip any!!
    #date_diffs = calls.groupby('instrument_name')['date'].diff()
    #idx = date_diffs[date_diffs == timedelta(1)].index
    #consecutive_date_calls = calls.loc[idx]
    #calls.groupby(['instrument_name'])['date'].diff()
    #calls['date'].diff().dt.days.ne(1).cumsum()

    collector = []
    dailies = {}
    pnl = {}
    n_shares_dct = {}
    cost_of_shares_dct = {}
    interest_cost_dct = {}
    cumulative_cost_dct = {}
    delta_hedge_cost_dct = {}
    final_instrument_price_dct = {}
    initial_instrument_price_dct = {}
    initial_tau_dct = {}
    start_date_dct = {}
    end_date_dct = {}


    counter = 0
    for instrument in calls['instrument_name'].unique():
        
        idx = calls['instrument_name'] == instrument
        sub = calls.loc[idx]
        sub['future_position'] = None
        print(sub.head())

        #if sub.shape[0] <= 1:
        #    pdb.set_trace()

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

        # @Todo: 
        # Insert another row into daily, then concatenate all later
        # this should include settlement values

        # @Todo: 
        # Make sure that we only have one price per day!!! 
        # Rebalance Daily!!!

        # Hedge PnL

        # Rookley can have some missing deltas (due to missing IVs!!)
        if any(np.isnan(daily['delta'])):
            print('nan!!')
            continue;
            #pdb.set_trace()
        # So we are going one row too far!!
        # Clip (drop last row) for delta hedge calculation
        clipped = daily.iloc[:-1]
        
        n_shares_dct[instrument], cost_of_shares_dct[instrument], cumulative_cost_dct[instrument], interest_cost_dct[instrument] = delta_hedge(daily['delta'].to_list(), daily['index_price'].to_list(), 0.01, daily['tau'].iloc[0])
        delta_hedge_cost = cumulative_cost_dct[instrument][-1]
        initial_instrument_price = daily.iloc[0]['instrument_price']

        #@Todo: Still have to estimate instrument price from expiration price
        final_instrument_price = daily.iloc[-1]['instrument_price_on_expiration']
        initial_tau = daily.iloc[0]['tau']
        

        # Store
        dailies[instrument] = daily
        initial_instrument_price_dct[instrument] = initial_instrument_price
        final_instrument_price_dct[instrument] = final_instrument_price
        initial_tau_dct[instrument] = initial_tau
        start_date_dct[instrument] = daily.iloc[0]['day']
        end_date_dct[instrument] = daily.iloc[-1]['day']

        # @Todo also have to consider the payoff of the long call here!!
        # Delta hedge cost is negative if we made a profit in the hedge position!
        #pnl[instrument] = final_instrument_price - initial_instrument_price - delta_hedge_cost #+ daily['hedge_pnl'].sum()

        # Relative to initial price
        # ABSOLUTE
        # Maybe goes to infinity for small initial_instrument_price 
        pnl[instrument] = (final_instrument_price - initial_instrument_price - delta_hedge_cost) #/ initial_instrument_price

        if np.isnan(pnl[instrument]):
            print('delta hedge cost is nan')
            pdb.set_trace()
            pnl[instrument] = np.nan

        #counter += 1
        #if counter >= 1000:
        #    pdb.set_trace()

    pnl_df = pd.DataFrame({'pnl': pnl})
    pnl_df.to_csv('out/pnl_df' + iv_var_name + '_week=' + str(week) + '.csv')
    print(pnl_df.describe())

    pdb.set_trace()
    #test_out = pd.DataFrame([pnl, n_shares_dct, cost_of_shares_dct, cumulative_cost_dct, interest_cost_dct, initial_instrument_price_dct, final_instrument_price_dct])
    #columns = ['pnl', 'n_shares', 'cost_of_shares', 'cumulative_cost', 'interest_cost', 'initial_instrument_price', 'final_instrument_price']

    #dailies_df = pd.DataFrame(data = dailies.values(), index = dailies.keys())

    # n_shares_dct, cost_of_shares_dct, cumulative_cost_dct, interest_cost_dct, 
    # 'n_shares', 'cost_of_shares', 'cumulative_cost', 'interest_cost',
    perf_overview = pd.DataFrame(data = [pnl,  initial_instrument_price_dct, final_instrument_price_dct, initial_tau_dct, start_date_dct, end_date_dct], index = ['pnl',  'initial_instrument_price', 'final_instrument_price', 'tau', 'start_date', 'end_date']).T
    perf_overview['return_on_init_price'] = perf_overview['pnl'] / perf_overview['initial_instrument_price']
    perf_overview['ndays'] = perf_overview['tau'] * 365
    #perf_overview['ndays_rounded'] = round(perf_overview['tau'], 2)

    grp = perf_overview.groupby('ndays')['pnl']
    print(grp.describe())

    # perf_overview.loc[(perf_overview['ndays'] > 1) & (perf_overview['ndays'] < 2)]['pnl'].mean()
    perf_overview.loc[(perf_overview['ndays'] > 1) & (perf_overview['ndays'] < 2)]['pnl'].describe()

    # @Todo: Now relate this plot to the IV over Realized Vola premium!!
    plt.plot(pd.to_datetime(perf_overview['start_date']), perf_overview['pnl'])
    plt.ylim(-5000, 5000)
    plt.show()
    
    plt.plot(pd.to_datetime(perf_overview['start_date']), perf_overview['return_on_init_price'])
    plt.ylim(-2, 2)
    plt.show()
    
    
    #perf_overview.loc[perf_overview['ndays'] == 9].describe()

    # Instead: Add all None's in the last row if daily data frame, then add the columns

    # @Todo: PnL per Group: First time-to-maturity, moneyness. Also show over time. 
    # Compare to Initial IV and Difference between Initial IV to running average. 
    # Check Jackwerth

    #@Todo: 
    # 1) Performance over Time (use start date and end date)
    # 2) Relate this to IV vs Realized Vola Premium
    # 3) Take actual expiration price for final instrument price instead of an estimation!!!
    # 4) For #1, Calculate Mean IV and Realized Variance!
    # 5) Restrict for ATM instruments, only count each once!
    # Exclude Outliers

    return pnl_df

if __name__ == '__main__':

    # Load Expiration Price History
    expiration_price_history = load_expiration_price_history()

    # Load Fitted Data from main.py
    dat = pd.read_csv('out/fitted_data.csv')#pd.read_csv('data/option_transactions.csv')
    dat['date'] = pd.to_datetime(dat['day'])

    if not 'nweeks' in dat.columns:
        pdb.set_trace()
        dat = assign_groups(dat)
    
    dat.sort_values('day', inplace = True)

    # Merge Expiration Prices to Transactions
    #pdb.set_trace()
    #window_size = 2
    dat = dat.merge(expiration_price_history, on ='Date')
    vola_df = dat.copy(deep=True)

    # IV vs RV for fixed amount of days
    for i in range(10):
        daily_volas = vola_df.loc[round(vola_df['tau'] * 365) == i]
        if daily_volas.shape[0] > 10:
            avg_daily_vola = daily_volas.groupby('day')['rookley_predicted_iv', 'expiration_price'].mean().reset_index()
            avg_daily_vola['rolling_iv'] = avg_daily_vola['rookley_predicted_iv']#.mean() #.rolling(window_size).mean()# * (365/window_size)
            avg_daily_vola['rolling_rv'] = avg_daily_vola['expiration_price'].pct_change().rolling(2).std() * (365)**0.5 #compute_vola(avg_daily_vola['expiration_price'], 1) 
            avg_daily_vola['date'] = pd.to_datetime(avg_daily_vola['day'])

            fig = plt.figure(figsize = (10,7))
            plt.plot(avg_daily_vola['date'], avg_daily_vola['rolling_iv'], label = 'IV')
            plt.plot(avg_daily_vola['date'], avg_daily_vola['rolling_rv'], label = 'RV')
            plt.title('Days to Maturity: ' + str(i))
            plt.legend()
            plt.savefig('plots/iv_vs_rv_ndays=' + str(i) + '.png')
            #vola_df['rolling_daily_real_vola'] = compute_vola(avg_daily_vola['rookley_predicted_iv'], window_size)

            avg_daily_vola['voldiff'] = avg_daily_vola['rolling_iv'] - avg_daily_vola['rolling_rv']
            print(avg_daily_vola['voldiff'].describe())

            fig = plt.figure(figsize = (10,7))
            plt.plot(avg_daily_vola['date'], avg_daily_vola['voldiff'])
            plt.title('Days to Maturity: ' + str(i))
            plt.legend()
            plt.savefig('plots/voladiff_ndays=' + str(i) + '.png')
            

    # Assess difference between Rookley and simple Regression estimated IV
    dat['ivdiff_abs'] = dat['rookley_predicted_iv'] - dat['predicted_iv']
    dat['ivdiff_rel'] = dat['ivdiff_abs'] / dat['rookley_predicted_iv']
    dat[['ivdiff_abs', 'ivdiff_rel']].describe()

    # Find Outliers
    print(dat[['rookley_predicted_iv', 'predicted_iv']].describe())

    max_iv = 2.5
    min_iv = 0
    iv_vars = ['rookley_predicted_iv', 'predicted_iv']

    for iv_var in iv_vars:
        dat.loc[dat[iv_var] >= max_iv, iv_var] = max_iv 
        dat.loc[dat[iv_var] <= min_iv, iv_var] = min_iv

    #@Todo: Set bounds for prediction in the actual prediction
    #dat = dat.loc[pd.notna(dat['rookley_predicted_iv'])]
    #rook = analyze_portfolio(dat, 'all', 'rookley_predicted_iv')
    pdb.set_trace()
    rookley_missing_instruments = dat.loc[dat['rookley_predicted_iv'].isna(), 'instrument_name']
    rookley_filtered_dat = dat.loc[~dat['instrument_name'].isin(rookley_missing_instruments)]
    rook_test = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv', expiration_history = expiration_price_history)

    test = analyze_portfolio(dat, 'all', 'predicted_iv', expiration_history = expiration_price_history)
    
    pdb.set_trace()
    pnl_per_group = {}
    for week in dat['nweeks'].unique():
        print('Week Group: ', week)
        df = dat[dat['nweeks'] == week]
        df.sort_values('day', inplace = True)
        pnl_per_group[week] = analyze_portfolio(df, week, 'predicted_iv')
    pdb.set_trace()
    for key, val in pnl_per_group.items():
        print('Week: ', key)
        print(val.describe())
        
    pdb.set_trace()