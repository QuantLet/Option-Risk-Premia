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
from src.zero_beta_straddles import get_call_beta, get_combined_beta, get_put_beta, get_straddle_weights, get_beta

def greeks(options, iv_var_name):
    """

    """
    #First, use BS Call value function to get Dollar Value for Call Parameters
    options['instrument_price_on_expiration'] = options.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0) if x['is_call'] == 1 else Put.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0), axis = 1)
    options['instrument_price'] = options.apply(lambda x: Call.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else Put.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    options['delta'] = options.apply(lambda x: Call.Delta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else Put.Delta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    
    # Calculate Call Beta, Put Beta for every day. Both calculations work
    options['old_beta'] = options.apply(lambda x: get_call_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else np.nan, axis = 1)
    options['beta'] = options.apply(lambda x: get_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau'], x['is_call']), axis = 1)
    #options['put_beta'] = options.apply(lambda x: get_put_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 0 else np.nan, axis = 1)
    #get_put_beta()
    return options

def get_synthetic_options(existing_options, fotm, move_strike = 1.2, move_iv = 1.3):
    """
    Create a synthetic Pair via Put-Call-Parity

    If fotm is active, then create FOTM option that changes strike and IV by a factor

    """
    # For which paired instruments do we not have observations?
    print('Synthetic Matches Only!')
    #missing_options = existing_options.copy(deep = True)
    # @Todo: Looks like this doesnt work yet. Didnt change the output!
    missing_options = existing_options.copy(deep = True)
    missing_options = missing_options[['instrument_name', 'pair_name', 'is_call', 'strike', 'spot', 'tau', 'iv', 'expiration_price']]

    if fotm:
        missing_options['strike'] = missing_options['strike'] * move_strike
        missing_options['iv'] = missing_options['iv'] * move_iv
    else:
        # If not FOTM, then we are inverting the instrument name / type (matching puts to calls and vice versa)
        # For FOTM instruments, we are retaining the type of instrument.

        # Reverse instrument name and pair name and then fill with Put-Call-Parity
        missing_options[['instrument_name', 'pair_name']] = missing_options[['pair_name', 'instrument_name']]

        # Reverse Puts and Calls
        missing_options['is_call'] = abs(missing_options['is_call'] - 1)

    # Overwrite Spot and Moneyness 
    missing_options['moneyness'] = missing_options['strike'] / missing_options['spot']

    return missing_options


def portfolio_calculations(pf_df, weight, long):
    """
    pf_df: call_df, put_df
    weight from straddle_weight
    crash_resistant, long: boolean
    """

    # Sell call_weight of Calls and put_weight of Puts
    pf_df['weight'] = weight
    
    pf_df['cost_base'] = pf_df['weight'] * (pf_df['instrument_price'])
    if long:
        pf_df['payoff'] = pf_df['instrument_price_on_expiration'] - pf_df['instrument_price'] 
        pf_df['direction'] = 'long'
    else:
        pf_df['payoff'] = pf_df['instrument_price'] - pf_df['instrument_price_on_expiration']
        pf_df['direction'] = 'short'

    pf_df['ret'] = pf_df['payoff'] / pf_df['instrument_price']
    pf_df['weighted_ret'] = pf_df['ret'] * weight
    pf_df['weighted_payoff'] = pf_df['payoff'] * weight

    return pf_df


def analyze_portfolio(dat, week, iv_var_name, center_on_expiration_price, first_occurrence_only = True, long = False, crash_resistant = True):
    """
    dat, pd.DataFrame as from main.py
    week, int, week indicator
    iv_var_name, string, variable name of estimated implied volatility: 
        1) 'predicted_iv' for simple regression
        2) 'rookley_predicted_iv' for rookley
    calls: Boolean, if False then puts

    first_occurrence_only: Using only the first observation of an instrument
    synthetic_matches_only: Always use Put-Call-Parity to find matching Straddle instrument instead of existing instruments (snapped at different times)
    --> now synthetic is always on
    long: long straddle, else short straddle
    crash_resistant: using FOTM options to protect against crash. For now pretending that the premium is 0 and strike is 1000 further out, meaning
    that the max loss per short position is 1000
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

    # Overwrite Moneyness with chosen Spot 
    # (main.py uses dat['index_price'] for moneyness )
    sub['moneyness'] = sub['strike'] / sub['spot']

    # Restrict Moneyness
    sub = sub.loc[(sub['moneyness'] <= 1.3) & sub['moneyness'] >= 0.7]

    # Min 4 days to maturity
    existing_options = sub.copy(deep = True)

    # Construct Pairs - Match Calls and Puts Names so that they have the same strike and maturity
    existing_options['pair_name'] = existing_options.apply(lambda x: x['instrument_name'].replace('-C', '-P') if x['is_call'] == 1 else x['instrument_name'].replace('-P', '-C'), axis = 1)
    missing_options = get_synthetic_options(existing_options, fotm = False)

    # Create FOTM Pairs for existing and missing options each
    #raise ValueError('LOOK!')
    print("We dont have a proper separation between calls and puts here!!!")
    # This should be converted into X * strike for calls and Y * strike for Puts. Then select just as for the "exist" dataframe.
    # This matching doesnt work yet. 
    fotm_calls = get_synthetic_options(existing_options, True, 1.25, 1)
    fotm_puts = get_synthetic_options(missing_options, True, 0.75, 1)

    
    exist = greeks(existing_options, iv_var_name)
    missing = greeks(missing_options, iv_var_name)
    fotm_c = greeks(fotm_calls, iv_var_name)
    fotm_p = greeks(fotm_puts, iv_var_name)

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
            
            # This should fix the current problem. Now need to adjust the [i]
            if opt['is_call'] == 1:
                call_df = exist.iloc[i]
                put_df = missing.iloc[i]
                fotm_call = fotm_c.iloc[i]
                fotm_put = fotm_p.iloc[i]

            elif opt['is_call'] == 0:
                put_df = exist.iloc[i]
                call_df = missing.iloc[i]
                fotm_put = fotm_c.iloc[i]
                fotm_call = fotm_p.iloc[i]

            else:
                raise ValueError('is_call is not binary!')

            # For Key-Name in dict
            call_name = call_df['instrument_name']
            put_name = put_df['instrument_name']

            # Only take the first row! No rebalancing performed at the time
            call_price = call_df['instrument_price']
            call_beta = call_df['beta']
            
            spot = call_df['spot']
            put_price = put_df['instrument_price']

            # Using new method now, applied in "greeks" function
            #put_beta = get_put_beta(call_price, put_price, spot, call_beta)
            #put_df['old_beta'] = put_beta
            #call_df['put_beta'] = np.nan

            #print('need to adjust straddle weights for FOTM Options too!!')
            call_weight, put_weight = get_straddle_weights(call_df['beta'], put_df['beta'])


            # Test implementing the adjusted beta calculation for crash-resistant Straddles
            if crash_resistant:
                print('Testing Crash Resistance')
                #print('GOT IT: HAVE TO USE WHOLE POSITION BETA, MEANING S/(P1 - P2) * (DELTA1 - DELTA2)')

                # Call Beta Adj is too often negative!! Check this!
                call_beta_adj = get_combined_beta(call_df['instrument_price'], fotm_c.iloc[i]['instrument_price'],call_df['delta'], fotm_c.iloc[i]['delta'], spot)
                put_beta_adj = get_combined_beta(put_df['instrument_price'], fotm_p.iloc[i]['instrument_price'],put_df['delta'], fotm_p.iloc[i]['delta'], spot)
                
                # @Todo: The fotm_c df doesnt ahve a beta here yet!!
                #call_beta_adj = call_df['beta'] - fotm_c.iloc[i]['beta']
                #put_beta_adj = put_df['beta'] - fotm_p.iloc[i]['beta']
                if call_beta_adj < 0 or put_beta_adj > 0:
                    print('So now the difference in Call Beta is going to be negative!!')
                    pdb.set_trace()
                crash_resistant_call_weight, crash_resistant_put_weight = get_straddle_weights(call_beta_adj, put_beta_adj)
                #print(crash_resistant_call_weight, crash_resistant_put_weight)
                #print(call_weight, put_weight)
            

            # Get FOTM Call and Put Price
            # Restrict max Payoff with the FOTM Strike (e.g. 2000)
            # Consider Cost of FOTM Option

            call_df = portfolio_calculations(call_df, call_weight, False)            
            put_df = portfolio_calculations(put_df, put_weight, False)

            fotm_call_df = portfolio_calculations(fotm_c.iloc[i], call_weight, True)
            fotm_put_df = portfolio_calculations(fotm_p.iloc[i], put_weight, True)

            
            cols = ['instrument_name', 'spot', 'strike', 'moneyness', 'tau', 'instrument_price', 'instrument_price_on_expiration', 'direction', 'beta', 'cost_base','payoff', 'weight', 'weighted_payoff', 'ret', 'weighted_ret']
            call_sub = call_df[cols]
            put_sub = put_df[cols]
            fotm_call_sub = fotm_call_df[cols]
            fotm_put_sub = fotm_put_df[cols]            

            # Test
            out = pd.DataFrame({'atm_call': call_sub, 'fotm_call': fotm_call_sub, 'atm_put': put_sub, 'fotm_put': fotm_put_sub}).T
            #out = call_sub.to_frame().join(put_sub.to_frame(), lsuffix = '_call', rsuffix = '_put', how = 'outer').T
            out['day'] = opt['day']
            out['days_to_maturity'] = opt['days_to_maturity']

            
            out['combined_payoff'] = out['weighted_payoff'].sum() 
            out['combined_ret'] = out['combined_payoff'] / (out['cost_base'].sum()) #(out['instrument_price_call'] * call_weight + out['instrument_price_put'] * put_weight)
            
            daily_factor = call_df['tau'] * 365
            out['combined_payoff_daily'] = out['combined_payoff'] / daily_factor
            out['combined_ret_daily'] = out['combined_ret'] / daily_factor
            
            keyname = str(call_name) + ' + ' + str(put_name) 
            out_dct[keyname] = out

            
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
    print('Loading BTC Data')
    dat = pd.read_csv('out/raw_deribit_transactions.csv') # deribit_transactions_eth for Ethereum
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

    # Run Analysis for Rookley and Regression
    #rookley_performance_overview = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv')
    performance_overview_l = analyze_portfolio(dat, 'all', 'iv', center_on_expiration_price)

    pdb.set_trace()
    collected = []
    for key, val in performance_overview_l.items():
        # First row summarizes results
        collected.append(val.iloc[0])
    performance_overview = pd.DataFrame(collected)
    performance_overview.to_csv('out/performance_overview.csv')

    # Plot Straddle Returns for inspection
    # @Todo: Should also do this for each day and instrument only once!
    performance_overview['combined_ret'].plot.kde()

    # Invert 
    #print('Invert Payoff and Returns!!')
    #performance_overview['combined_ret'] = performance_overview['combined_ret'] * (-1)
    #performance_overview['combined_payoff'] = performance_overview['combined_payoff'] * (-1)
    #performance_overview['combined_ret_daily'] = performance_overview['combined_ret_daily'] * (-1)
    #performance_overview['combined_payoff'] = performance_overview['combined_payoff_daily'] * (-1)

    atm_sub = performance_overview.loc[(performance_overview['moneyness'] >= 0.95) & (performance_overview['moneyness'] <= 1.05)][['tau', 'moneyness', 'combined_payoff', 'combined_ret', 'combined_payoff_daily','combined_ret_daily']]
    atm_sub = performance_overview.loc[(performance_overview['moneyness'] >= 0.95) & (performance_overview['moneyness'] <= 1.05)][['combined_payoff', 'combined_ret', 'tau','moneyness', 'combined_payoff_daily','combined_ret_daily']]
    grouped_boxplot(atm_sub, 'combined_ret', 'tau', -2, 2, 'atm')
    grouped_boxplot(atm_sub, 'combined_payoff', 'tau', -5000, 5000, 'atm')
    grouped_boxplot(atm_sub, 'combined_ret_daily', 'tau', -2, 2, 'atm')
    grouped_boxplot(atm_sub, 'combined_payoff_daily', 'tau', -5000, 5000, 'atm')

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
    #simple_3d_plot(performance_overview['tau'], performance_overview['moneyness'], performance_overview['combined_payoff'], 'plots/3d_combined_payoff.png', 'Tau', 'Moneyness', 'Payoff', -5000, 5000)
    #simple_3d_plot(performance_overview['tau'], performance_overview['moneyness'], performance_overview['combined_ret'], 'plots/3d_combined_return.png', 'Tau', 'Moneyness', 'Return', -2, 2)
    
    # Per Tau
    plot_performance(performance_overview, 'tau')

    # Per Week
    plot_performance(performance_overview, 'nweeks')
