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
from src.blackscholes import Call, Put
from src.helpers import assign_groups, load_expiration_price_history, compute_vola
from src.zero_beta_straddles import get_call_beta, get_put_beta, get_straddle_weights


def analyze_portfolio(dat, week, iv_var_name):
    """
    dat, pd.DataFrame as from main.py
    week, int, week indicator
    iv_var_name, string, variable name of estimated implied volatility: 
        1) 'predicted_iv' for simple regression
        2) 'rookley_predicted_iv' for rookley
    calls: Boolean, if False then puts
    """

    #dat = dat.rename(columns = {'spot': 'index_price'})
 
    # Run for dailies first
    print('Only Dailies!')

    dat['ndays'] = dat['tau'] * 365
    sub = dat.loc[(dat['ndays'] >= 0) & (dat['ndays']<= 1)]

    # Min 4 days to maturity
    existing_options = sub
    #options = dat.loc[dat['tau'] * 365 >= 7]

    # Construct Pairs - Match Calls and Puts Names so that they have the same strike and maturity
    existing_options['pair_name'] = existing_options.apply(lambda x: x['instrument_name'].replace('-C', '-P') if x['is_call'] == 1 else x['instrument_name'].replace('-P', '-C'), axis = 1)

    # For which paired instruments do we not have observations?
    missing_options = existing_options[~existing_options['pair_name'].isin(existing_options['instrument_name'])]

    # Reverse instrument name and pair name and then fill with Put-Call-Parity
    missing_options = missing_options.rename(columns = {'instrument_name': 'pair_name', 'pair_name': 'instrument_name'})

    # Reverse Puts and Calls
    missing_options['is_call'] = abs(missing_options['is_call'] - 1)

    # Overwrite Moneyness
    missing_options['moneyness'] = missing_options['strike'] / missing_options['spot']

    # Combine with existing options
    options = pd.concat([existing_options, missing_options], ignore_index=True)


    # First, use BS Call value function to get Dollar Value for Call Parameters
    options['instrument_price_on_expiration'] = options.apply(lambda x: Call.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0) if x['is_call'] == 1 else Put.Price(x['expiration_price'], x['strike'], 0, x[iv_var_name], 0), axis = 1)
    options['instrument_price'] = options.apply(lambda x: Call.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else Put.Price(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)
    options['delta'] = options.apply(lambda x: Call.Delta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']), axis = 1)

    # Calculate Call Beta, Put Beta for every day
    options['call_beta'] = options.apply(lambda x: get_call_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 1 else np.nan, axis = 1)
    #options['put_beta'] = options.apply(lambda x: get_put_beta(x['spot'], x['strike'], 0, x[iv_var_name], x['tau']) if x['is_call'] == 0 else np.nan, axis = 1)


    # @Todo Check this one:
    # BTC-24SEP21-26000-C
    # Looks like wrong tau

    counter = 0
    out_dct = {}
    # Looping over Calls only to match Puts
    for instrument in options.loc[options['is_call'] == 1]['instrument_name'].unique():
        try:
            # We could also just price Puts by Put-Call-Parity (same IV)

            call_idx = options['instrument_name'] == instrument
            #daily = options.loc[idx]
            
            # Perform Rebalancing / @Todo: No rebalancing so far!!

            # Match
            call_df = options.loc[call_idx]
            matching_put_names = call_df['pair_name'].unique()
            if len(matching_put_names) == 1:
                call_name = call_df['instrument_name'].unique()[0]
                matching_put_name = matching_put_names[0]
            else:
                print('Couldnt find matching Instrument')
                continue
            put_df = options.loc[options['instrument_name'] == matching_put_name]
            if put_df.shape[0] != 1:
                print('No Put')
                pdb.set_trace()
                # Match here via Put-Call-Parity
                continue

            if call_df.shape[0] > 1:
                print("here")
                pdb.set_trace()

            call_price = call_df.iloc[0]['instrument_price']
            call_beta = call_df.iloc[0]['call_beta']
            spot = call_df.iloc[0]['spot']
            put_price = put_df.iloc[0]['instrument_price']
            put_beta = get_put_beta(call_price, put_price, spot, call_beta)
            put_df['put_beta'] = put_beta

            call_weight, put_weight = get_straddle_weights(call_beta, put_beta)

            # Sell call_weight of Calls and put_weight of Puts
            call_df['payoff'] = call_df['instrument_price'] - call_df['instrument_price_on_expiration']
            call_df['ret'] = call_df['payoff'] / call_df['instrument_price']
            call_df['weighted_ret'] = call_df['ret'] * call_weight
            call_df['weighted_payoff'] = call_df['payoff'] * call_weight

            put_df['payoff'] = put_df['instrument_price'] - put_df['instrument_price_on_expiration']
            put_df['ret'] = put_df['payoff'] / put_df['instrument_price']
            put_df['weighted_ret'] = put_df['ret'] * put_weight
            put_df['weighted_payoff'] = put_df['payoff'] * put_weight
            
            #perf = call_df['weighted_payoff'] + put_df['weighted_payoff']
            call_sub = call_df[['Date', 'moneyness', 'days_to_maturity', 'tau', 'instrument_name', 'spot', 'instrument_price', 'instrument_price_on_expiration', 'call_beta', 'payoff', 'weighted_payoff', 'ret', 'weighted_ret']].reset_index()
            put_sub = put_df[['instrument_name', 'spot', 'instrument_price', 'instrument_price_on_expiration', 'put_beta', 'payoff', 'weighted_payoff', 'ret', 'weighted_ret']].reset_index()

            out = call_sub.join(put_sub, lsuffix = '_call', rsuffix = '_put', how = 'outer')

            #out = pd.concat([call_sub, put_sub], ignore_index=True, suf)
            out['combined_payoff'] = out['weighted_payoff_call'] + out['weighted_payoff_put']
            out['combined_ret'] = out['combined_payoff'] / (out['instrument_price_call'] * call_weight + out['instrument_price_put'] * put_weight)

            keyname = str(call_name) + ' + ' + str(matching_put_name) 
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
    dat['maturitydate'] = dat['maturitydate_trading'].apply(lambda x: str(x)[:10])
    dat = dat.merge(expiration_price_history, left_on ='maturitydate', right_on = 'Date')
    dat.rename(columns = {'Date_x': 'Date'}, inplace=True)
    vola_df = dat.copy(deep=True)
    
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
    


    # Run Analysis for Rookley and Regression
    #rookley_performance_overview = analyze_portfolio(rookley_filtered_dat, 'all', 'rookley_predicted_iv')
    performance_overview_l = analyze_portfolio(dat, 'all', 'predicted_iv')
    
    collected = []
    for key, val in performance_overview_l.items():
        collected.append(val)
    performance_overview = pd.concat(collected, ignore_index=True)   
    #test = pd.DataFrame.from_dict(regression_performance_overview_l, orient = 'index')
    #regression_performance_overview = pd.DataFrame([regression_performance_overview_l.values()], index = regression_performance_overview_l.keys())
    #regression_performance_overview = pd.concat(regression_performance_overview_l, ignore_index = True)
    print(performance_overview[['combined_payoff', 'combined_ret', 'tau']].describe())

    # Investigate too low days to maturity
    # Maybe the -29 comes from taking days to maturity until end-of-month....
    print('Maturitydate problem should come from stratify_instruments')

    #performance_overview.loc[performance_overview['days_to_maturity'] > 2]
    #performance_overview.loc[performance_overview['days_to_maturity'] == -29]
    #performance_overview.loc[performance_overview['tau'] < 0]
    #print('Currently Shorting Straddles, but should be inverting that!')

    #performance_overview.loc[performance_overview['days_to_maturity'] != -29][['combined_payoff', 'combined_ret', 'tau', 'moneyness']].describe()

    performance_overview['rounded_tau'] = round(performance_overview['tau'], 2)
    performance_overview['rounded_moneyness'] = round(performance_overview['moneyness'], 2)
    des = performance_overview.loc[(performance_overview['moneyness'] >= 0.7) & (performance_overview['moneyness'] <= 1.3)][['combined_payoff', 'combined_ret', 'rounded_tau','rounded_moneyness']].groupby(['rounded_tau', 'rounded_moneyness']).describe()
    print(des.to_string())
    des.to_csv('out/Zero_Beta_Performance_Overview_Summary_Statistics.csv')

    

    # Make this plot over Tau and Moneyness

    performance_overview['Date'] = pd.to_datetime(performance_overview['Date'])
    for tau in performance_overview['rounded_tau'].unique():

        tau_sub = performance_overview.loc[performance_overview['rounded_tau'] == tau]
        tau_label = str(tau)

        # @Todo: Now relate this plot to the IV over Realized Vola premium!!
        fig = plt.figure(figsize = (10,7))
            
        plt.subplot(2, 1, 1)
        plt.plot(tau_sub['Date'], tau_sub['combined_payoff'], label = tau_label)
        plt.ylim(-5000, 5000)
        
        plt.subplot(2,1,2)
        plt.plot(tau_sub['Date'], tau_sub['combined_ret'], label = tau_label)
        plt.ylim(-2, 2)

        plt.legend()
        plt.savefig('plots/' + 'zero_beta_straddle_tau=' + tau_label + '.png')
