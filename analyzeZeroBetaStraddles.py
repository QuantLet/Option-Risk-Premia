"""
Construct Portfolio of a Long Call and Long Put, resulting in a Straddle
This Straddle has a zero-Beta exposure to the underlying index
A zero-beta Straddle is also Delta-Neutral

Analyze empirical PnL: is it close to the risk-free rate? 
"""

from numpy.core.defchararray import center
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
from datetime import timedelta
from pathlib import Path
from math import ceil
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

def get_synthetic_options(existing_options, fotm, move_strike_call = 1.2, move_strike_put = 0.8, move_iv = 1.3):
    """
    Create a synthetic Pair via Put-Call-Parity

    If fotm is active, then create FOTM option that changes strike and IV by a factor

    """
    print('Synthetic Matches Only!')
    missing_options = existing_options.copy(deep = True)
    missing_options = missing_options[['instrument_name', 'pair_name', 'is_call', 'index_change_rel', 'strike', 'spot', 'tau', 'iv', 'expiration_price', 'ndays_ceiling']]

    if fotm:
        missing_options['strike'] = missing_options.apply(lambda x: x['strike'] * move_strike_call if x['is_call'] == 1 else x['strike'] * move_strike_put, axis = 1)
        missing_options['iv'] = missing_options['iv'] * move_iv
    else:
        # If not FOTM, then we are inverting the instrument name / type (matching puts to calls and vice versa)
        # For FOTM instruments, we are retaining the type of instrument.

        # Reverse instrument name and pair name and then fill with Put-Call-Parity
        missing_options[['instrument_name', 'pair_name']] = missing_options[['pair_name', 'instrument_name']]

        # Reverse Puts and Calls
        missing_options['is_call'] = abs(missing_options['is_call'] - 1)

    # Overwrite Spot and Moneyness 
    missing_options['moneyness'] = round(missing_options['strike'] / missing_options['spot'], 2)

    return missing_options


def portfolio_calculations(pf_df, weight, is_long):
    """
    pf_df: call_df, put_df
    weight from straddle_weight
    crash_resistant, long: boolean
    """

    # Sell call_weight of Calls and put_weight of Puts
    pf_df['weight'] = weight
    
    pf_df['cost_base'] = pf_df['weight'] * (pf_df['instrument_price'])
    if is_long:
        pf_df['payoff'] = pf_df['instrument_price_on_expiration'] - pf_df['instrument_price'] 
        pf_df['direction'] = 'long'
    else:
        pf_df['payoff'] = pf_df['instrument_price'] - pf_df['instrument_price_on_expiration']
        pf_df['direction'] = 'short'

    pf_df['ret'] = pf_df['payoff'] / pf_df['instrument_price']
    pf_df['weighted_ret'] = pf_df['ret'] * weight
    pf_df['weighted_payoff'] = pf_df['payoff'] * weight

    return pf_df


def analyze_portfolio(dat, week, iv_var_name, center_on_expiration_price, first_occurrence_only = True, long = True, crash_resistant = True, fees = True):
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
    crash_resistant: using FOTM options to protect against crash. 
    """
    if long == False:
        print('Try investing the received premium in the index!')


    if not center_on_expiration_price:
        dat['spot'] = dat['index_price']

    dat['ndays'] = dat['tau'] * 365
    dat['ndays_ceiling'] = dat['ndays'].apply(lambda x: ceil(x)) #ceil(dat['ndays'])
    dat.sort_values('day', inplace=True)

    if first_occurrence_only:
        sub = dat.groupby('instrument_name').first().reset_index()
        print('Using first occurrence of an instrument!')
    else:
        sub = dat
        print('Not using unique instruments! Using all!')

    # Overwrite Moneyness with chosen Spot 
    # (main.py uses dat['index_price'] for moneyness )
    sub['moneyness'] = round(sub['strike'] / sub['spot'], 2)

    # Restrict Moneyness / already happening in parent function
    #sub = sub.loc[(sub['moneyness'] <= 1.3) & (sub['moneyness'] >= 0.7) & (sub['tau'] <= 0.3)]
    
    # Min 4 days to maturity
    existing_options = sub.copy(deep = True)

    # Construct Pairs - Match Calls and Puts Names so that they have the same strike and maturity
    existing_options['pair_name'] = existing_options.apply(lambda x: x['instrument_name'].replace('-C', '-P') if x['is_call'] == 1 else x['instrument_name'].replace('-P', '-C'), axis = 1)
    
    missing_options = get_synthetic_options(existing_options, fotm = False)

    # Create FOTM Pairs for existing and missing options each
    fotm_exist = get_synthetic_options(existing_options, True)
    fotm_missing = get_synthetic_options(missing_options, True)

    exist = greeks(existing_options, iv_var_name)
    missing = greeks(missing_options, iv_var_name)
    fotm_e = greeks(fotm_exist, iv_var_name)
    fotm_m = greeks(fotm_missing, iv_var_name)
    
    counter = 0
    out_dct = {}
    
    for i in range(len(exist)): #.loc[options['is_call'] == 1]
        try:

            # No Rebalancing implemented!
            opt = exist.iloc[i]
            
            if opt['is_call'] == 1:
                call_df = exist.iloc[i]
                put_df = missing.iloc[i]
                fotm_call = fotm_e.iloc[i]
                fotm_put = fotm_m.iloc[i]

            elif opt['is_call'] == 0:
                put_df = exist.iloc[i]
                call_df = missing.iloc[i]
                fotm_put = fotm_e.iloc[i]
                fotm_call = fotm_m.iloc[i]

            else:
                raise ValueError('is_call is not binary!')

            # For Return on Cash, Fees
            index_price = opt['index_price']
            index_change_rel = opt['index_change_rel']

            # For Key-Name in dict
            call_name = call_df['instrument_name']
            put_name = put_df['instrument_name']

            # Only take the first row! No rebalancing performed at the time
            call_price = call_df['instrument_price']
            call_beta = call_df['beta']
            
            spot = call_df['spot']
            put_price = put_df['instrument_price']

            #print('need to adjust straddle weights for FOTM Options too!!')
            call_weight, put_weight = get_straddle_weights(call_df['beta'], put_df['beta'])

            # Test implementing the adjusted beta calculation for crash-resistant Straddles
            if crash_resistant:
                #print('GOT IT: HAVE TO USE WHOLE POSITION BETA, MEANING S/(P1 - P2) * (DELTA1 - DELTA2)')

                # Call Beta Adj is too often negative!! Check this!
                call_beta_adj = get_combined_beta(call_df['instrument_price'], fotm_call['instrument_price'],call_df['delta'], fotm_call['delta'], spot)
                put_beta_adj = get_combined_beta(put_df['instrument_price'], fotm_put['instrument_price'],put_df['delta'], fotm_put['delta'], spot)
                
                if call_beta_adj < 0 or put_beta_adj > 0:
                    print('So now the difference in Call Beta is going to be negative!!')
                    # Perhaps unselect these imbalanced trades 
                crash_resistant_call_weight, crash_resistant_put_weight = get_straddle_weights(call_beta_adj, put_beta_adj)
                
                # Otherwise: Could just be taking a subset when the calculation is done
                # Check if Zero-Beta-Straddle is feasible, else just continue
                if crash_resistant_call_weight is None or crash_resistant_put_weight is None:
                    print('\nStraddle not feasible: ', opt)
                    continue
            else:
                call_beta_adj = np.nan
                put_beta_adj = np.nan

                # Check if Zero-Beta-Straddle is feasible, else just continue
                if call_weight is None or put_weight is None:
                    print('\nStraddle not feasible: ', opt)
                    continue
            
            # Get FOTM Call and Put Price
            # Restrict max Payoff with the FOTM Strike (e.g. 2000)
            # Consider Cost of FOTM Option

            # Change to accept parent parameter of is_long
            not_long = not long

            call_df = portfolio_calculations(call_df, call_weight, long)            
            put_df = portfolio_calculations(put_df, put_weight, long)

            fotm_call_df = portfolio_calculations(fotm_call, call_weight, not_long)
            fotm_put_df = portfolio_calculations(fotm_put, put_weight, not_long)
            
            cols = ['instrument_name', 'index_change_rel', 'spot', 'strike', 'moneyness', 'tau', 'instrument_price', 'instrument_price_on_expiration', 'direction', 'beta', 'cost_base','payoff', 'weight', 'weighted_payoff', 'ret', 'weighted_ret']
            call_sub = call_df[cols]
            put_sub = put_df[cols]
            fotm_call_sub = fotm_call_df[cols]
            fotm_put_sub = fotm_put_df[cols]     
            out = pd.DataFrame({'atm_call': call_sub, 'fotm_call': fotm_call_sub, 'atm_put': put_sub, 'fotm_put': fotm_put_sub}).T

            if fees:
                n_options = 2
                btc_fee_per_option = 0.0003
                fees_abs = n_options * btc_fee_per_option * index_price
                print("make sure that fees are max 12.5% of option price!")
            else:
                fees_abs = 0
            print('Fees: ', fees_abs)

            if long == False: #  and counter == 0
                counter += 1
                # Use Cash if we are Short
                collected_premia = out['instrument_price'].sum()
                profit_on_cash = index_change_rel * collected_premia
            else:
                profit_on_cash = 0
            
            out['day'] = opt['day']
            out['days_to_maturity'] = opt['days_to_maturity']
            
            out['combined_payoff'] = out['weighted_payoff'].sum() + profit_on_cash - fees_abs
            out['combined_ret'] = out['combined_payoff'] / (abs(out['cost_base'].sum())) 

            # If we center on expiration price, then we are resetting the time value of a call to the beginning of a day. 
            if center_on_expiration_price:
                daily_factor = call_df['ndays_ceiling']
            else:
                daily_factor = call_df['tau'] * 365

            out['combined_payoff_daily'] = out['combined_payoff'] / daily_factor
            out['combined_ret_daily'] = out['combined_ret'] / daily_factor

            out['call_beta_adj'] = call_beta_adj
            out['put_beta_adj'] = put_beta_adj
            
            keyname = str(call_name) + ' + ' + str(put_name) 
            out_dct[keyname] = out

        except Exception as e:
            print(e)
            print('Error!')
            #pdb.set_trace()
    print('done')
    return out_dct



        


if __name__ == '__main__':

    # Output Paths must exist:
    # /plots/vanilla
    # /plots/crash_resistant
    required_paths = ['plots/vanilla/fees',
                      'plots/vanilla/no_fees',
                      'plots/crash_resistant/fees',
                      'plots/crash_resistant/no_fees',
                      'out/vanilla/fees',
                      'out/vanilla/no_fees', 
                      'out/crash_resistant/fees',
                      'out/crash_resistant/no_fees']

    for req_path in required_paths:
        Path(req_path).mkdir(parents=True, exist_ok=True)

    # Params
    center_on_expiration_price = True

    # Iterate over crash resistance and fees:
    crash_resistance = False
    apply_fees = True

    for crash_resistance, apply_fees in zip([True, True, False, False], [True, False, True, False]):

        # Output Directories
        if crash_resistance:
            out_dir_start = 'out/crash_resistant/'
            plot_dir_start = 'plots/crash_resistant/'
        else:
            out_dir_start = 'out/vanilla/'
            plot_dir_start = 'plots/vanilla/'
        if apply_fees:
            out_dir = out_dir_start + 'fees/'
            plot_dir = plot_dir_start + 'fees/'
        else:
            out_dir = out_dir_start + 'no_fees/'
            plot_dir = plot_dir_start + 'no_fees/'

        # config
        base_currency = 'btc'
        print("###")
        print(base_currency)
        print("###")
        if base_currency == 'eth':
            f = 'eth/out/raw_deribit_transactions_eth.csv'
        elif base_currency == 'btc':
            f = 'btc/out/raw_deribit_transactions.csv'
        else: 
            raise NotImplementedError()


        # Load Expiration Price History
        expiration_price_history = load_expiration_price_history(currency = base_currency)

        # Load Fitted Data from main.py
        dat = pd.read_csv(f) # deribit_transactions_eth for Ethereum
        dat['date'] = pd.to_datetime(dat['day'])
        dat = dat.loc[dat['day'] >= '2019-05-01']
        print('Using data past 2019-05-01 because its the inception point of the perpetual futures.')
        
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

            print('WARNING: CHECK IF THIS IS A CORRECT OVERWRITE!!')

        # Daily Returns of the Index. Required for Cash Investment on short side. Drop last observation where the return would be NaN.
        dat['index_change_abs'] = dat['expiration_price'] - dat['index_price'] #dat['index_price'].diff().shift(-1)
        dat['index_change_rel'] = dat['index_change_abs'] / dat['index_price']

        vola_df = dat.copy(deep=True)
        
        # Find Outliers
        print(dat['iv'].describe())

        # Get average premium on IV if we go to a moneyness of 1.3 (or 0.7)
        print('Summary of 30% OTM IV for Text:')
        print(dat.loc[dat['moneyness'] == 1.3]['iv'].describe())
        print(dat.loc[dat['moneyness'] == 0.7]['iv'].describe())
        print(dat.loc[dat['moneyness'] == 1.0]['iv'].describe())

        max_iv = 2.5
        min_iv = 0.1
        min_moneyness = 0.7
        max_moneyness = 1.3
        iv_vars = ['iv']

        # Pre Filtering
        print('Pre-filtering distribution')
        print(dat['iv'].describe().round(2).to_latex(multicolumn = True))
        print(dat['tau'].describe().round(2).to_latex(multicolumn = True))

        # Find share of observations where IV is larger than max_iv
        # or smaller than min_iv
        #pdb.set_trace()
        iv_outlier_sub = dat.loc[(dat['iv'] >= max_iv) | (dat['iv'] <= min_iv)]
        print(iv_outlier_sub.describe())
        print('Amount of Observations outside of allowed IV range: ', iv_outlier_sub.shape[0]/dat.shape[0])

        moneyness_outlier_sub = dat.loc[(dat['moneyness'] >= max_moneyness) | (dat['moneyness'] <= min_moneyness)]
        print(moneyness_outlier_sub.describe())
        print('Amount of Observations outside of allowed Tau range: ', moneyness_outlier_sub.shape[0]/dat.shape[0])

        dat = dat.loc[(dat['iv'] <= max_iv) & (dat['iv'] >= min_iv) & (dat['moneyness'] <= max_moneyness) & (dat['moneyness'] >= min_moneyness)]
        print('Amount of Observations in our set post filtering: ', dat.shape[0])

        # Post Filtering
        print('Post-filtering distribution')
        print(dat['iv'].describe().round(2).to_latex(multicolumn = True))
        print((dat['tau']*365).describe().round(2).to_latex(multicolumn = True))

        #for iv_var in iv_vars:
        #    dat.loc[dat[iv_var] >= max_iv, iv_var] = max_iv 
        #    dat.loc[dat[iv_var] <= min_iv, iv_var] = min_iv

        # Run Analysis for Rookley and Regression
        #rookley_performance_overview = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv')
        performance_overview_l = analyze_portfolio(dat, week = 'all', iv_var_name = 'iv', center_on_expiration_price = center_on_expiration_price, 
                            first_occurrence_only = True, long = True, crash_resistant = crash_resistance, fees = apply_fees)

        collected = []
        for key, val in performance_overview_l.items():
            # First row summarizes results
            collected.append(val.iloc[0])
        performance_overview = pd.DataFrame(collected)
        performance_overview.to_csv(out_dir + 'performance_overview.csv')

        # We are not just dropping bad performers here: For some instruments, beta-neutral weights are not possible. 
        # For these, we will have NaN weights, which then translate into NaN payoff and consequently inf or -inf returns.
        # Drop those cases.
        # Ensure range limits.
        performance_overview.replace([np.inf, -np.inf], np.nan, inplace=True)
        performance_overview.dropna(subset=["combined_payoff_daily", "combined_ret_daily"], how="all", inplace=True)

        atm_sub = performance_overview.loc[(performance_overview['moneyness'] >= 0.95) & (performance_overview['moneyness'] <= 1.05)][['combined_payoff', 'combined_ret', 'tau','moneyness', 'combined_payoff_daily','combined_ret_daily']]
        grouped_boxplot(atm_sub, 'combined_ret', 'tau', -2, 2, plot_dir, 'atm', crash_resistance)
        grouped_boxplot(atm_sub, 'combined_payoff', 'tau', -5000, 5000, plot_dir, 'atm', crash_resistance)
        grouped_boxplot(atm_sub, 'combined_ret_daily', 'tau', -2, 2, plot_dir, 'atm', crash_resistance)
        grouped_boxplot(atm_sub, 'combined_payoff_daily', 'tau', -5000, 5000, plot_dir, 'atm', crash_resistance)

        des = performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)][['combined_payoff', 'combined_ret', 'tau','moneyness']].groupby(['tau', 'moneyness']).describe()
        print(des.to_string())
        des.to_csv(out_dir + 'Zero_Beta_Performance_Overview_Summary_Statistics.csv')
        performance_overview.to_csv(out_dir + 'PerformanceOverview.csv')
    
        # Prepare Performance Plots
        performance_overview = assign_groups(performance_overview)
        performance_overview['day'] = pd.to_datetime(performance_overview['day'])
            
        # Add Boxplots for Performance Overview per Tau!!!
        grouped_boxplot(performance_overview, 'combined_payoff_daily', 'tau', -5000, 5000, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview, 'combined_ret_daily', 'tau', -2, 2, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview, 'combined_payoff_daily', 'nweeks', -5000, 5000, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview, 'combined_ret_daily', 'nweeks', -2, 2, plot_dir, '', crash_resistance)
        
        # If we don't round here, then it gets too messy. Adjust the X Axis otherwise...
        grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_payoff', 'moneyness', -5000, 5000, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_ret', 'moneyness', -2, 2, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_payoff_daily', 'moneyness', -5000, 5000, plot_dir, '', crash_resistance)
        grouped_boxplot(performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)], 'combined_ret_daily', 'moneyness', -2, 2, plot_dir, '', crash_resistance)
        
