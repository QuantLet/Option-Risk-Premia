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
import math
from datetime import timedelta
import pdb
from src.plots import simple_3d_plot, plot_performance, grouped_boxplot
from src.blackscholes import Call, Put
from src.helpers import assign_groups, load_expiration_price_history, compute_vola
from src.zero_beta_straddles import get_call_beta, get_put_beta, get_straddle_weights

def greeks(options, iv_var_name):
    """

    """
    #First, use BS Call value function to get Dollar Value for Call Parameters
    options['instrument_price_on_expiration'] = options.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0) if x['is_call'] == 1 else Put.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0), axis = 1)
    options['instrument_price'] = options.apply(lambda x: Call.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else Put.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    options['delta'] = options.apply(lambda x: Call.Delta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else Put.Delta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)

    # Calculate Call Beta, Put Beta for every day
    options['call_beta'] = options.apply(lambda x: get_call_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else np.nan, axis = 1)
    #options['put_beta'] = options.apply(lambda x: get_put_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 0 else np.nan, axis = 1)
    #get_put_beta()
    return options

def analyze_portfolio(dat, week, iv_var_name, center_on_expiration_price, first_occurrence_only = True, synthetic_matches_only = True, long = False):
    """
    dat, pd.DataFrame as from main.py
    week, int, week indicator
    iv_var_name, string, variable name of estimated implied volatility: 
        1) 'predicted_iv' for simple regression
        2) 'rookley_predicted_iv' for rookley
    calls: Boolean, if False then puts

    first_occurrence_only: Using only the first observation of an instrument
    synthetic_matches_only: Always use Put-Call-Parity to find matching Straddle instrument instead of existing instruments (snapped at different times)
    long: long straddle, else short straddle
    """
    if not center_on_expiration_price:
        dat['spot'] = dat['index_price']

    dat['ndays'] = dat['tau'] * 365
    dat.sort_values('day', inplace=True)
    #sub = dat.loc[(dat['ndays'] >= 0) & (dat['ndays']<= 30)]
    #sub = dat

    # Get first mentioning of each option
    #pdb.set_trace()
    if first_occurrence_only:
        sub = dat.groupby('instrument_name').first().reset_index()
        print('Using first occurrence of an instrument!')
    else:
        sub = dat
        print('Not using unique instruments! Using all!')

    # Locate on the last expiration price
    #sub['spot']

    # Restrict Moneyness
    sub = sub.loc[(sub['moneyness'] <= 1.3) & sub['moneyness'] >= 0.7]

    # Min 4 days to maturity
    existing_options = sub.copy(deep = True)
    #options = dat.loc[dat['tau'] * 365 >= 7]

    # Construct Pairs - Match Calls and Puts Names so that they have the same strike and maturity
    existing_options['pair_name'] = existing_options.apply(lambda x: x['instrument_name'].replace('-C', '-P') if x['is_call'] == 1 else x['instrument_name'].replace('-P', '-C'), axis = 1)

    # For which paired instruments do we not have observations?
    if synthetic_matches_only:
        print('Synthetic Matches Only!')
        #missing_options = existing_options.copy(deep = True)
        # @Todo: Looks like this doesnt work yet. Didnt change the output!
        missing_options = existing_options.copy(deep = True)
        missing_options = missing_options[['instrument_name', 'pair_name', 'is_call', 'strike', 'spot', 'tau', 'iv', 'expiration_price']]
    else:    
        missing_options = existing_options[~existing_options['pair_name'].isin(existing_options['instrument_name'])]

    # Reverse instrument name and pair name and then fill with Put-Call-Parity
    #missing_options = missing_options.rename(columns = {'instrument_name': 'pair_name', 'pair_name': 'instrument_name'})
    missing_options[['instrument_name', 'pair_name']] = missing_options[['pair_name', 'instrument_name']]


    # Reverse Puts and Calls
    missing_options['is_call'] = abs(missing_options['is_call'] - 1)

    # Overwrite Spot and Moneyness 
    missing_options['moneyness'] = missing_options['strike'] / missing_options['spot']
    
    # Combine with existing options
    #options = pd.concat([existing_options, missing_options], ignore_index=True)

    exist = greeks(existing_options, iv_var_name)
    missing = greeks(missing_options, iv_var_name)

    # @Todo Check this one:
    # BTC-24SEP21-26000-C
    # Looks like wrong tau

    counter = 0
    out_dct = {}
    # Looping over Calls only to match Puts
    # @Todo: Check how this behaves for all options, not just Calls
    print('Check behavior for all options, not just Calls! This is not gonna work because we are expecting call_df to exist initially!')
    print('Check Formula for Weights!!!')
    for i in range(len(exist)): #.loc[options['is_call'] == 1]
        try:

            #@Hint: We can also match by index here as each index in existing_options correspoinds to missing_options!
            # Why does call_df have less columns than exist?!
            #print('Why does call_df have less columns than exist?!')
            # Ahh! Well its either in call_df or in put_df now

            # No Rebalancing implemented so far!
            opt = exist.iloc[i]
            
            if opt['is_call'] == 1:
                call_df = exist.iloc[i]
                put_df = missing.iloc[i]

            elif opt['is_call'] == 0:
                put_df = exist.iloc[i]
                call_df = missing.iloc[i]

            else:
                raise ValueError('is_call is not binary!')

            # For Key-Name in dict
            call_name = call_df['instrument_name']
            put_name = put_df['instrument_name']

            if call_name == 'BTC-4DEC21-57000-C' or put_name == 'BTC-4DEC21-57000-C':
                print("here")
                pdb.set_trace()

            
            # Only take the first row! No rebalancing performed at the time
            call_price = call_df['instrument_price']
            call_beta = call_df['call_beta']
            
            spot = call_df['spot']
            put_price = put_df['instrument_price']
            put_beta = get_put_beta(call_price, put_price, spot, call_beta)
            put_df['put_beta'] = put_beta
            call_df['put_beta'] = np.nan

            call_weight, put_weight = get_straddle_weights(call_beta, put_beta)

            if math.isinf(call_weight) or math.isinf(put_weight) or call_weight < 0 or put_weight < 0:
                print('Inf!!')
                pdb.set_trace()

            # Sell call_weight of Calls and put_weight of Puts
            call_df['weight'] = call_weight
            call_df['cost_base'] = call_df['weight'] * call_df['instrument_price']
            if long:
                call_df['payoff'] = call_df['instrument_price_on_expiration'] - call_df['instrument_price'] 
            else:
                call_df['payoff'] = call_df['instrument_price'] - call_df['instrument_price_on_expiration']
            call_df['ret'] = call_df['payoff'] / call_df['instrument_price']
            call_df['weighted_ret'] = call_df['ret'] * call_weight
            call_df['weighted_payoff'] = call_df['payoff'] * call_weight

            put_df['weight'] = put_weight
            put_df['cost_base'] = put_df['weight'] * put_df['instrument_price']
            if long:
                put_df['payoff'] = put_df['instrument_price'] - put_df['instrument_price_on_expiration']
            else:
                put_df['payoff'] = put_df['instrument_price'] - put_df['instrument_price_on_expiration']
            put_df['ret'] = put_df['payoff'] / put_df['instrument_price']
            put_df['weighted_ret'] = put_df['ret'] * put_weight
            put_df['weighted_payoff'] = put_df['payoff'] * put_weight

            
            cols = ['instrument_name', 'spot', 'instrument_price', 'instrument_price_on_expiration', 'call_beta', 'put_beta', 'cost_base','payoff', 'weight', 'weighted_payoff', 'ret', 'weighted_ret']
            call_sub = call_df[cols]
            put_sub = put_df[cols]
            

            out = call_sub.to_frame().join(put_sub.to_frame(), lsuffix = '_call', rsuffix = '_put', how = 'outer').T

            #out = call_sub.join(put_sub, lsuffix = '_call', rsuffix = '_put', how = 'outer')
            out[['day', 'days_to_maturity', 'moneyness', 'tau']] = opt[['day', 'days_to_maturity', 'moneyness', 'tau']]

            #out = pd.concat([call_sub, put_sub], ignore_index=True, suf)
            out['combined_payoff'] = out['weighted_payoff'].sum() 
            out['combined_ret'] = out['combined_payoff'] / (out['cost_base'].sum()) #(out['instrument_price_call'] * call_weight + out['instrument_price_put'] * put_weight)
            
            daily_factor = call_df['tau'] * 365
            out['combined_payoff_daily'] = out['combined_payoff'] / daily_factor
            out['combined_ret_daily'] = out['combined_ret'] / daily_factor
            
            keyname = str(call_name) + ' + ' + str(put_name) 
            out_dct[keyname] = out
            #out_l.append(out)

            #print('\nCall: ', call_df[['instrument_name', 'spot', 'instrument_price', 'instrument_price_on_expiration', 'call_beta']])
            #print('\nPut: ', put_df[['instrument_name', 'spot', 'instrument_price', 'instrument_price_on_expiration', 'put_beta']])
            #print('\n Call Weight, Put Weight: ', call_weight, ' ... ' ,'put_weight')
        except Exception as e:
            print(e)
            pdb.set_trace()
    print('done')
    return out_dct



        


if __name__ == '__main__':

    # Params
    center_on_expiration_price = True

    # Load Expiration Price History
    expiration_price_history = load_expiration_price_history()

    # Load Fitted Data from main.py
    dat = pd.read_csv('out/raw_transactions.csv')#pd.read_csv('data/option_transactions.csv')
    dat['date'] = pd.to_datetime(dat['day'])

    if not 'nweeks' in dat.columns:
        pdb.set_trace()
        dat = assign_groups(dat)
    
    dat.sort_values('day', inplace = True)
    
    # Merge Expiration Prices to Transactions
    dat['maturitydate'] = dat['maturitydate_trading'].apply(lambda x: str(x)[:10])
    dat = dat.merge(expiration_price_history, left_on ='maturitydate', right_on = 'Date')
    dat.rename(columns = {'Date_x': 'Date'}, inplace=True)
    
    if center_on_expiration_price:
        # Use expiration date also as spot for each day ('trade around spot")
        dat = dat.merge(expiration_price_history.rename(columns={'expiration_price':'spot'}), left_on = 'day', right_on = 'Date')

    # Drop all mentionings of 'Date', since it is only associated to expiration
    #dat.drop(columns = ['Date_x', 'Date_y'], inplace = True)

    vola_df = dat.copy(deep=True)
    
    # Find Outliers
    print(dat['iv'].describe())

    max_iv = 2.5
    min_iv = 0
    iv_vars = ['iv']

    for iv_var in iv_vars:
        dat.loc[dat[iv_var] >= max_iv, iv_var] = max_iv 
        dat.loc[dat[iv_var] <= min_iv, iv_var] = min_iv

    #@Todo: Set bounds for prediction in the actual prediction
    #rookley_missing_instruments = dat.loc[dat['rookley_predicted_iv'].isna(), 'instrument_name']
    #rookley_filtered_dat = dat.loc[~dat['instrument_name'].isin(rookley_missing_instruments)]
    


    # Run Analysis for Rookley and Regression
    #rookley_performance_overview = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv')
    performance_overview_l = analyze_portfolio(dat, 'all', 'iv', center_on_expiration_price)

    collected = []
    for key, val in performance_overview_l.items():
        # First row summarizes results
        collected.append(val.iloc[0])
    performance_overview = pd.DataFrame(collected)
    performance_overview.to_csv('out/performance_overview.csv')
    #performance_overview = pd.concat(collected, ignore_index=True).reset_index()
    
    #test = pd.DataFrame.from_dict(regression_performance_overview_l, orient = 'index')
    #regression_performance_overview = pd.DataFrame([regression_performance_overview_l.values()], index = regression_performance_overview_l.keys())
    #regression_performance_overview = pd.concat(regression_performance_overview_l, ignore_index = True)
    #print(performance_overview[['combined_payoff', 'combined_ret', 'tau']].describe())

    # Investigate too low days to maturity
    # Maybe the -29 comes from taking days to maturity until end-of-month....

    #pdb.set_trace()
    #performance_overview.apply(lambda x: math.isinf(performance_overview['combined_payoff'][x]), axis = 1)
    #inf_idx = performance_overview['combined_payoff'][np.isinf(performance_overview['combined_payoff'])].index
    #infsub = performance_overview.loc[inf_idx]

    #performance_overview.loc[performance_overview['days_to_maturity'] > 2]
    #performance_overview.loc[performance_overview['days_to_maturity'] == -29]
    #performance_overview.loc[performance_overview['tau'] < 0]
    #print('Currently Shorting Straddles, but should be inverting that!')

    #performance_overview.loc[performance_overview['days_to_maturity'] != -29][['combined_payoff', 'combined_ret', 'tau', 'moneyness']].describe()

    # The transposed / pd.Series transformation fucks up the data type, so got to force float

    # Invert 
    #print('Invert Payoff and Returns!!')
    #performance_overview['combined_ret'] = performance_overview['combined_ret'] * (-1)
    #performance_overview['combined_payoff'] = performance_overview['combined_payoff'] * (-1)
    #performance_overview['combined_ret_daily'] = performance_overview['combined_ret_daily'] * (-1)
    #performance_overview['combined_payoff'] = performance_overview['combined_payoff_daily'] * (-1)

    # IRR
    #(performance_overview['combined_ret'] * 365) / (performance_overview['tau'])
    #performance_overview['daily_ret'] = performance_overview['combined_ret'] / performance_overview['tau']
    #performance_overview.loc[(performance_overview['daily_ret'] > -100) & (performance_overview['daily_ret'] <= 100)]['daily_ret'].dropna().describe()
    #pdb.set_trace()
    # Much stronger than [0.9, 1.1]
    atm_sub = performance_overview.loc[(performance_overview['moneyness'] >= 0.95) & (performance_overview['moneyness'] <= 1.05)][['tau', 'moneyness', 'combined_payoff', 'combined_ret', 'combined_payoff_daily','combined_ret_daily']]
    atm_sub = performance_overview.loc[(performance_overview['moneyness'] >= 0.95) & (performance_overview['moneyness'] <= 1.05)][['combined_payoff', 'combined_ret', 'tau','moneyness', 'combined_payoff_daily','combined_ret_daily']]
    grouped_boxplot(atm_sub, 'combined_ret', 'tau', -2, 2, 'atm')
    grouped_boxplot(atm_sub, 'combined_payoff', 'tau', -5000, 5000, 'atm')
    grouped_boxplot(atm_sub, 'combined_ret_daily', 'tau', -2, 2, 'atm')
    grouped_boxplot(atm_sub, 'combined_payoff_daily', 'tau', -5000, 5000, 'atm')
    #pdb.set_trace()
    

    # HOW CAN WE HAVE a return outside of [-1,1]?!
    
    perf = performance_overview.copy(deep = True)
    #perf['combined_payoff_adj'] = perf['combined_payoff']
    #perf[perf['combined_payoff_adj'] > 500] = 500
    #perf[perf['combined_payoff_adj'] < -2] = -2
    #perf.describe()

    # Something is wrong here - we should have a range from -40k to 27k
    #perf.loc[perf['combined_payoff'] == perf['combined_payoff'].max()].iloc[0]
    #perf.loc[perf['combined_payoff'] == perf['combined_payoff'].min()]


    des = performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)][['combined_payoff', 'combined_ret', 'tau','moneyness']].groupby(['tau', 'moneyness']).describe()
    print(des.to_string())
    des.to_csv('out/Zero_Beta_Performance_Overview_Summary_Statistics.csv')
    performance_overview.to_csv('out/PerformanceOverview.csv')
  
    
    # Prepare Performance Plots
    performance_overview = assign_groups(performance_overview)
    performance_overview['day'] = pd.to_datetime(performance_overview['day'])
    
    
    # Add Boxplots for Performance Overview per Tau!!!
    grouped_boxplot(performance_overview, 'combined_payoff_daily', 'tau', -5000, 5000)
    grouped_boxplot(performance_overview, 'combined_ret_daily', 'tau', -2, 2)
    grouped_boxplot(performance_overview, 'combined_payoff_daily', 'nweeks', -5000, 5000)
    grouped_boxplot(performance_overview, 'combined_ret_daily', 'nweeks', -2, 2)
    
    # If we don't round here, then it gets too messy. Adjust the X Axis otherwise...
    #
    grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_payoff', 'moneyness', -5000, 5000)
    grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_ret', 'moneyness', -2, 2)
    grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_payoff_daily', 'moneyness', -5000, 5000)
    grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_ret_daily', 'moneyness', -2, 2)
    #@Todo: Ensure that x axis is date! its too tight!
    #@Todo: Invert Returns and Combined Payoff for long straddles!
    # @Todo Introduce interest rate!
    #@Todo: Color Moneyness ranges in plots!
    #pdb.set_trace()
    
    # Make PNL plot over Tau and Moneyness
    simple_3d_plot(performance_overview['tau'], performance_overview['moneyness'], performance_overview['combined_payoff'], 'plots/3d_combined_payoff.png', 'Tau', 'Moneyness', 'Payoff', -5000, 5000)
    simple_3d_plot(performance_overview['tau'], performance_overview['moneyness'], performance_overview['combined_ret'], 'plots/3d_combined_return.png', 'Tau', 'Moneyness', 'Return', -2, 2)
    
    # Per Tau
    plot_performance(performance_overview, 'tau')

    # Per Week
    plot_performance(performance_overview, 'nweeks')
