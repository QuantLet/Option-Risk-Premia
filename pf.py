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
from src.delta_hedging import delta_hedge, simple_delta_hedge
from src.blackscholes import Call
from src.helpers import assign_groups, load_expiration_price_history, compute_vola


def analyze_portfolio(dat, week, iv_var_name, calls = True):
    """
    dat, pd.DataFrame as from main.py
    week, int, week indicator
    iv_var_name, string, variable name of estimated implied volatility: 
        1) 'predicted_iv' for simple regression
        2) 'rookley_predicted_iv' for rookley
    calls: Boolean, if False then puts
    """

    dat = dat.rename(columns = {'spot': 'index_price'})
    
    # Min 4 days to maturity
    options = dat.loc[(dat['is_call'] == 1) & (dat['tau'] * 365 >= 2)]
    #options = dat.loc[(dat['is_call'] == 1)]
    print('Calls only')

    options['instrument_price_on_expiration'] = options.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0), axis = 1)

    # First, use BS Call value function to get Dollar Value for Call Parameters
    options['instrument_price'] = options.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    options['delta'] = options.apply(lambda x: Call.Delta(x['expiration_price'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)

    collector = []
    dailies = {}
    pnl = {}
    pnl_relative = {}
    cumulative_cost_dct = {}
    delta_hedge_cost_dct = {}
    final_instrument_price_dct = {}
    initial_instrument_price_dct = {}
    initial_tau_dct = {}
    initial_strike_dct = {}
    initial_moneyness_dct = {}
    start_date_dct = {}
    end_date_dct = {}


    counter = 0
    for instrument in options['instrument_name'].unique():
        
        idx = options['instrument_name'] == instrument
        daily = options.loc[idx]
        daily['future_position'] = None
        
        # Hedge PnL
        # Rookley can have some missing deltas (due to missing IVs!!)
        if any(np.isnan(daily['delta'])):
            print('nan!!')
            continue;
            #pdb.set_trace()
        
        # Execute Hedge
        hedge_payoff_vec = simple_delta_hedge(daily)
        hedge_payoff_sum = hedge_payoff_vec.sum()

        # Helpers
        initial_instrument_price = daily.iloc[0]['instrument_price']
        final_instrument_price = daily.iloc[-1]['instrument_price_on_expiration']
        initial_tau = daily.iloc[0]['tau']
        initial_strike = daily.iloc[0]['strike']
        initial_moneyness = daily.iloc[0]['moneyness']

        # Store
        dailies[instrument] = daily
        initial_instrument_price_dct[instrument] = initial_instrument_price
        final_instrument_price_dct[instrument] = final_instrument_price
        initial_tau_dct[instrument] = initial_tau
        initial_strike_dct[instrument] = initial_strike
        initial_moneyness_dct[instrument] = initial_moneyness
        start_date_dct[instrument] = daily.iloc[0]['day']
        end_date_dct[instrument] = daily.iloc[-1]['day']
        delta_hedge_cost_dct[instrument] = hedge_payoff_sum 

        # Absolute Pnl
        pnl[instrument] = (final_instrument_price - initial_instrument_price - hedge_payoff_sum)
        
        # Relative to initial price
        if initial_instrument_price > 0.01:
            pnl_relative = pnl[instrument] / initial_instrument_price
        else:
            pnl_relative = np.nan

    pnl_df = pd.DataFrame({'pnl': pnl})
    pnl_df.to_csv('out/pnl_df' + iv_var_name + '_week=' + str(week) + '.csv')
    print(pnl_df.describe())

    perf_overview = pd.DataFrame(data = [pnl,  initial_instrument_price_dct, final_instrument_price_dct, initial_tau_dct, start_date_dct, end_date_dct], index = ['pnl',  'initial_instrument_price', 'final_instrument_price', 'tau', 'start_date', 'end_date']).T
    perf_overview.to_csv('out/perf_overview' + iv_var_name + '_calls=' + str(calls) +'_week=' + str(week) + '.csv')

    overview = pd.DataFrame({'pnl': pnl, 'pnl_relative': pnl_relative,'tau': initial_tau_dct, 'moneuyness': initial_moneyness_dct, 'strike': initial_strike_dct})
    overview['ndays'] = overview['tau'] * 365
    #overview.groupby('ndays').describe()

    over = assign_groups(overview)
    print(over.groupby('nweeks').describe())
    over.to_csv('out/overview' + iv_var_name + '_calls=' + str(calls) + '_week=' + str(week) + '.csv')

    return perf_overview

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
            plt.savefig('plots/iv_vs_rv_ndays=' + str(i) + '.png', transparent = True)
            #vola_df['rolling_daily_real_vola'] = compute_vola(avg_daily_vola['rookley_predicted_iv'], window_size)

            avg_daily_vola['voldiff'] = avg_daily_vola['rolling_iv'] - avg_daily_vola['rolling_rv']
            print(avg_daily_vola['voldiff'].describe())

            fig = plt.figure(figsize = (10,7))
            plt.plot(avg_daily_vola['date'], avg_daily_vola['voldiff'])
            plt.title('Days to Maturity: ' + str(i))
            plt.legend()
            plt.savefig('plots/voladiff_ndays=' + str(i) + '.png', transparent = True)
            
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
    rookley_missing_instruments = dat.loc[dat['rookley_predicted_iv'].isna(), 'instrument_name']
    rookley_filtered_dat = dat.loc[~dat['instrument_name'].isin(rookley_missing_instruments)]
    
    for is_calls in [True, False]:

        # Run Analysis for Rookley and Regression
        rookley_performance_overview = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv', is_calls)
        regression_performance_overview = analyze_portfolio(dat, 'all', 'predicted_iv', is_calls)

        # @Todo: Now relate this plot to the IV over Realized Vola premium!!
        fig = plt.figure(figsize = (10,7))
        
        plt.subplot(2, 1, 1)
        plt.plot(pd.to_datetime(rookley_performance_overview['start_date']), rookley_performance_overview['pnl'])
        plt.ylim(-5000, 5000)
        plt.savefig('plots/rookley_pnl_calls=' + str(is_calls) + '.png')
        
        plt.subplot(2, 1, 2)
        plt.plot(pd.to_datetime(regression_performance_overview['start_date']), regression_performance_overview['pnl'])
        plt.ylim(-5000, 5000)
        plt.savefig('plots/regression_pnl_calls=' + str(is_calls) + '.png')
