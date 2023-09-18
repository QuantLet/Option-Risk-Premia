# Set US locale for datetime.datetime.strptime (Conflict with MAER/MAR)
import platform
import locale
from numpy.lib.function_base import insert
plat = platform.system()
if plat == 'Darwin':
    print('Using Macos Locale')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
else:
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np 
import time
import sys
import pdb
import pickle
import gc
import csv
import os
import math
import statsmodels as sm

from pathlib import Path
from src.brc import BRC
from src.helpers import decompose_instrument_name, find_nearest_location
from src.blackscholes import Call, Put
from scipy import stats
from statsmodels import api
from src.surface_plot_regression import surface_plot
from src.plots import simple_3d_plot
from src.helpers import assign_groups
#from statsmodels.api import OLS
#from src.vola_plots import plot_atm_iv_term_structure


"""

@ Todo:
1) Often, for higher maturities, finding an instrument in the strategy_range will fail because the spd is not interpolated!!
2) Probability for realization of profit.

From last session:
make a film out of the hockey graphics, maybe also with moneyness on x axis
boxplot for the multiple scatter plots (especially on put side as in example)
convert .tex file to keynote
correct epsilon, insert 'small' adj before infinitesimally

Generally:
- Implied Binomial Trees
- Historical SPD
- Skewness / Kurtosis Trades

todo: 
    annualize returns!!
    get realized variance! plot vs garch variance and vs implied volatility
    
# can only read profit using pickle!!

"""
def extend_polynomial(x, y):
    """
    Extend Smile, first and second derivative so that spd exists completely for large tau
    x = M_std
    y = first
    """
    
    polynomial_coeff=np.polyfit(x,y,2)
    xnew=np.linspace(0.6,1.4,100)
    ynew=np.poly1d(polynomial_coeff)
    #plt.plot(xnew,ynew(xnew),x,y,'o')
    #plt.title('interpolated smile')
    #plt.show()
    return xnew, ynew(xnew)

def gaussian_kernel(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return stats.norm.cdf(u_m) * stats.norm.cdf(u_t)

def epanechnikov(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return (3/4) * (1-u_m)**2 * (3/4) * (1-u_t)**2

def smoothing_rookley(df, m, t, h_m, h_t, kernel=gaussian_kernel, extend = False, boot = None):
    # M = np.array(df.M)
    # Before
    M = np.array(df.moneyness)
    if boot is None:
        y = np.array(df.iv)
    else:
       # print('using bootstrapped IV')
        y = np.array(boot)

    # After Poly extension
    if extend:
        print('Extending Moneyness and IV in smoothing technique!')
        M, y = extend_polynomial(M, y) # np.polyfit(np.array(df.moneyness, df.tau), df.iv, 2)
    T = df.tau.values#[df.tau.values[0]] * len(M) #np.array(df.tau)
    n = len(M)

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X4 = T-t
    X5 = (T-t)**2
    X6 = X2*X4
    X = np.array([X1, X2, X3, X4, X5, X6]).T

    # the kernel lays on M_j - m, T_j - t
    #ker = new_epanechnikov(X[:,5])
    ker = kernel(M, m, h_m, T, t, h_t)
    #test = gausskernel(X[:,5])
    W = np.diag(ker)

    # Compare Kernels
    # This kernel gives too much weight on far-away deviations
    #plt.scatter(M, ker, color = 'green')
    #plt.scatter(M, X[:,5], color = 'red')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.show()

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    # This is our estimated vs real iv 

    #iv_est = np.dot(X, beta)
    #plt.scatter(df.moneyness, df.mark_iv, color = 'red')
    #plt.scatter(df.moneyness, iv_est, color = 'green')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.title('est vs real iv and current location')
    #plt.show()
    
    
    return beta[0], beta[1], 2*beta[2], beta[3], beta[4], beta[5]

def rookley(df, h_m=0.01, h_t=0.01, gridsize=149, kernel='epak'):
    # gridsize is len of estimated smile

    """
    Solution: Instead of adjusting Rookley for non-fix taus, 
    just use a regression a la 
    iv ~ const + moneyness + moneyess**2 + tau + tau**2 + interact + error
    then predict
    evaluate estimated iv at those tau and moneyness, which are closest to our instruments


    """
    
    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    #tau = df.tau.iloc[0]
    M_min, M_max = min(df.moneyness), max(df.moneyness)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.moneyness), max(df.moneyness)
    M_std = np.linspace(M_std_min, M_std_max, num=num)

    # if all taus are the same
    tau_min, tau_max = min(df.tau[(df.tau > 0)]), max(df.tau)
    tau = np.linspace(tau_min, tau_max, gridsize)

    x = zip(M_std, tau)
    sig = np.zeros((num, 6)) # fill

    # TODO: speed up with tensor instead of loop
    for i, (m, t) in enumerate(x):
        sig[i] = smoothing_rookley(df, m, t, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1] #/ np.std(df.moneyness)
    second = sig[:, 2] #/ np.std(df.moneyness)
    first_tau = sig[:, 3]
    second_tau = sig[:, 4]
    interaction = sig[:, 5]
    
    """
    pdb.set_trace()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(first, first_tau, smile)
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

    #plt.plot(df.moneyness, df.iv, 'ro', ms=3, alpha=0.3, color = 'green')
    #plt.plot(df.moneyness, smile, 'ro', ms=3, alpha=0.3, color = 'red')
    #plt.plot(df.moneyness, first)
    #plt.plot(df.moneyness, second)
    #plt.show()
    """

    S_min, S_max = min(df.index_price), max(df.index_price)
    K_min, K_max = min(df.strike), max(df.strike)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)
    #pdb.set_trace()

    """
    plt.scatter(df.moneyness, df.iv, label = 'IV')
    plt.show()
    plt.plot(M, smile, label = 'smile')
    plt.plot(M, first, label = 'first derivative')
    plt.plot(M, second, label = 'second derivative')
    plt.legend()
    plt.show()
    """

    return smile, first, second, M, S, K, M_std, tau

def exogenous(T, M):
    """
    both variables, tau and moneyness, are pd.series
    for construction of iv surface
    """
    if isinstance(T, float) & isinstance(M, float):
        # For predictions of single instruments
        # But should vectorize this
        X1 = 1
    else:
        # Is pd.Series
        X1 = np.ones(len(T))
    X2 = M 
    X3 = M**2
    X4 = T
    X5 = T**2
    X6 = X2*X4
    X = np.array([X1, X2, X3, X4, X5, X6]).T
    #X = api.add_constant(X)
    
    return X


def calibrate_on_iv_surface(df, predict_sub, curr_day, do_plot = True):
    """
    Calculate IV surface based on 2nd order Regression
    IV ~ Intercept + Tau + Tau**2 + Moneyness + Moneyness**2 + M * T + Error

    #@Todo: Clip Outliers, dont regard Moneyness >= cutoff
    """
    IV = np.array(df['iv'])
    M = np.array(df['moneyness'])
    T = np.array(df['tau'])
    X = exogenous(T, M)
    
    model = api.OLS(IV, X)
    
    fit = model.fit()
    summary = fit.summary()
    print(summary)
    
    
    # Prediction of individual instrument
    exog_for_prediction = exogenous(predict_sub['tau'], predict_sub['moneyness'])#exogenous(0.05, 1.1)
    
    out = predict_sub.copy(deep = True)
    predicted = fit.predict(exog_for_prediction)
    out['predicted_iv'] = predicted
    #pdb.set_trace()
    #out.set_index(df.index, inplace = True)

    
    if do_plot:
        simple_3d_plot(out['tau'], out['moneyness'], out['predicted_iv'], 'out/' + curr_day.strftime('%Y-%m-%d') + '.png')
    
    return predicted


def filter_sub(_df):
    # Subset

    # Only calls, tau in [0, 0.25] and fix one day (bc looking at intra day here)
    # (_df['is_call'] == 1) & --> Dont need to filter for calls because that happens in the R script
    # Also need to consider put-call parity there
    sub = _df[(_df['moneyness'] >= 0.7) & (_df['moneyness'] < 1.3) &(_df['iv'] > 0) & (_df['iv'] <= 2.5)]# &

    print('not filtering sub for tau!!')
    #if tau > 0:
    #    sub = sub[sub['tau'] == tau]

    nrows = sub.shape[0]
    if nrows == 0:
        raise(ValueError('Sub is empty'))
   
    sub['moneyness'] = round(sub['moneyness'], 3)
    sub['index_price'] = round(sub['index_price'], 2)

    sub = sub.drop_duplicates()
    
    print(sub.describe())

    #if nrows > 50000:
    #    print('large df, subsetting')
    #    sub = sub.sample(10000)
    #    print(sub.describe())

    return sub

def classify_options(dat):
    """
    Classify Options to prepare range trading
    """
    dat['option_type'] = ''
    dat['option_type'][(dat['moneyness'] < 0.9)]                                = 'FOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.9) & (dat['moneyness'] < 0.95)]   = 'NOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.95) & (dat['moneyness'] < 1)]     = 'ATM Put'
    dat['option_type'][(dat['moneyness'] >= 1) & (dat['moneyness'] < 1.05)]     = 'ATM Call'
    dat['option_type'][(dat['moneyness'] >= 1.05) & (dat['moneyness'] < 1.1)]   = 'NOTM Call'
    dat['option_type'][(dat['moneyness'] >= 1.1)]                               = 'FOTM Call'
    return dat


def hist_iv(df):
    sub = df[df.mark_iv > 0]
    sub = sub.sort_values('maturitydate_char')
    #o = sub.groupby(['maturitydate_char', 'instrument_name', 'date']).mean()['iv']
    o = sub.groupby(['maturitydate_char', 'instrument_name']).mean()['iv']
    return o.to_frame()

def term_structure(df, curr_date, bw):
    """
    ATM Call Prices for different clusters of Tau
    Weeks until Maturity: 1, 2, 4, 8
    """
    # Last price is spot of highest timestamp
    max_ts_idx = df['timestamp'].idxmax()
    last_spot = df.loc[max_ts_idx]['index_price']
    
    # Find Closest Strike to last Spot
    strikes = df['strike'].unique()
    dist = abs(last_spot - strikes)
    min_dist_idx = np.argmin(dist)
    atm_strike = strikes[min_dist_idx]

    # Get Term Structure
    atm = df[df['strike'] == atm_strike]
    atm['iv'] = atm['iv'].astype(float) / 100
    
    fig = plt.figure()
    plt.subplot(111)
    plt.scatter(atm['nweeks'], atm['iv'], color = 'blue')
    plt.ylim(0, 0.85)

    #ax = plt.axes(projection='3d')
    #ax.scatter3D(df['strike'], df['tau'], df['instrument_price'])
    #plt.ylim(0, 10)
    #plt.xlim(0.9, 1.1)
    #plt.xlabel('Moneyness')

    fname = 'termstructure_reloaded/_bw-' + str(bw) + '_date-' + str(curr_date) + '.png'
    print(fname)
    plt.savefig(fname, transparent = True)

    return None

def plot_group(dat, amount_variable = 'amount', date_variable = 'day', plot_dir = 'plots/'):
    contract_overview = dat.groupby([date_variable, 'instrument_name', 'is_call'])[amount_variable].sum()

    calls = contract_overview.iloc[contract_overview.index.get_level_values('is_call') == 1]
    puts = contract_overview.iloc[contract_overview.index.get_level_values('is_call') == 0]

    calls_per_day = pd.DataFrame(calls).reset_index().groupby(date_variable)[amount_variable].sum()
    puts_per_day = pd.DataFrame(puts).reset_index().groupby(date_variable)[amount_variable].sum()

    plt.figure(figsize=(8, 6))
    plt.plot(calls_per_day, label = 'Call ' + amount_variable)
    plt.plot(puts_per_day, label = 'Put ' + amount_variable)
    plt.legend()

    fname = plot_dir + amount_variable + '_' + date_variable + '.png'
    plt.savefig(fname)
    print('Saved as: ', fname)

    pdb.set_trace()
    return calls, puts

def pct_change(before, after):
    """

    """
    if before is None or after is None:
        return None
    else:
        return (after - before) / before

def run(curr_day):
    print("Entering Main Loop")

    errors = []
    realized_vola = []
    historical_iv = []

    # Initiate BRC instance to query data. First and Last day are stored.
    brc = BRC()

    if curr_day < brc.last_day:
        print(curr_day)
        
        try:
            # make sure days are properly set
            curr_day_starttime = curr_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            curr_day_endtime = curr_day.replace(hour = 23, minute = 59, second = 59, microsecond = 0)

            # Debug
            #curr_day_starttime = datetime.datetime(2020, 4, 5, 0, 0, 0)
            #curr_day_endtime = datetime.datetime(2020, 4, 5, 23, 59, 59)

            print('\nStarting Simulation from ', curr_day_starttime, ' to ', curr_day_endtime)
            
            dat = brc._run(starttime = curr_day_starttime,
                            endtime = curr_day_endtime) 
            dat = pd.DataFrame(dat)

            assert(dat.shape[0] != 0)
    
            # Convert dates, utc
            dat['date'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), dat['timestamp']))
            dat_params  = decompose_instrument_name(dat['instrument_name'], dat['date'])
            dat         = dat.join(dat_params)

            # Drop all spoofed observations - where timediff between two orderbooks (for one instrument) is too small
            dat['timestampdiff'] = dat['timestamp'].diff(1)
            dat = dat[(dat['timestampdiff'] > 2)]

            dat['index_price']   = dat['index_price'].astype(float)

            # To check Results after trading 
            dates                       = dat['date']
            dat['strdates']             = dates.dt.strftime('%Y-%m-%d') 
            maturitydates               = dat['maturitydate_trading']
            dat['maturitydate_char']    = maturitydates.dt.strftime('%Y-%m-%d')

            # Calculate mean instrument price
            dat['instrument_price'] = dat['price'] * dat['index_price'] 

            # Prepare for moneyness domain restriction (0.8 < m < 1.2)
            dat['moneyness']    = round(dat['strike'] / dat['index_price'], 2)
            df                  = dat[['_id', 'index_price', 'amount', 'strike', 'maturity', 'is_call', 'tau', 'iv', 'date', 'moneyness', 'instrument_name', 'days_to_maturity', 'maturitydate_char', 'maturitydate_trading', 'timestamp','instrument_price']]    
                        
            ## Isolate vars
            df['iv'] = df['iv'] / 100
            df['iv'][(df['iv'] < 0.01)] = 0
            vola = df['iv'].astype(float)/100

            # Clusters for Tau: 1, 2, 4, 8 Weeks
            df['day'] = curr_day
            df['nweeks'] = 0
            floatweek = 1/52
            df['nweeks'][(df['tau'] <= floatweek)] = 1
            df['nweeks'][(df['tau'] > floatweek) & (df['tau'] <= 2 * floatweek)] = 2
            df['nweeks'][(df['tau'] > 2 * floatweek) & (df['tau'] <= 3 * floatweek)] = 3
            df['nweeks'][(df['tau'] > 3 * floatweek) & (df['tau'] <= 4 * floatweek)] = 4
            df['nweeks'][(df['tau'] > 4 * floatweek) & (df['tau'] <= 8 * floatweek)] = 8

            # Get Daily OI changes
            #pdb.set_trace()
            #df.groupby('instrument_name')['amount'].sum()

            # Get Greeks, especially Delta
            # Find sum of amount per instrument until expiration
            # Look at change in underlying around each expiration

            # Save output, concatenate to single DF later
            
            return df

            #term_structure(df, curr_day, r_bandwidth)
            #plot_atm_iv_term_structure(df, curr_day, r_bandwidth)
           
            #f = filter_sub(df, curr_day_starttime, curr_day_endtime, 0)
            #f.to_csv('out/filtered_' + str(curr_day_starttime) + '.csv')
            #trisurf(f['moneyness'], f['tau'], f['mark_iv'], 'moneyness', 'tau', 'vola', 'pricingkernel/plots/empirical_vola_smile_' + curr_day_starttime.strftime('%Y-%m-%d'), False)

        except Exception as e:
            print('Download or Processing failed!\n')
            print(e)  
        
        """
        # As taus are ascending, once we do not find one instrument for a specific taus it is unlikely to find one for the following
        # as the SPDs degenerate with higher taus.
        for tau in unique_taus:
            try:
                #r.XFGSPDcb2()
                print(tau)

                sub = filter_sub(df, curr_day_starttime, curr_day_endtime, tau)
                #s, sub  = spdbl(df, curr_day_starttime, curr_day_endtime, tau, int_rate, blockplots = True, bootstrap = False, physical_density=None)
                #area = verify_density(s)

                # Prepare Confidence Band Calculation for the whole Day
                conf_fname = prepare_confidence_band_data(sub)

                # last one on day which we observed
                observation_price = sub['index_price'].tail(1) 

                # need at least one day for the physical density, which is fixed in there!
                time_to_maturity_in_days = sub.days_to_maturity.unique()[0]
                rdate = base.as_Date(curr_day.strftime('%Y-%m-%d'))
                rdate_f = base.format(rdate, '%Y-%m-%d')

                # Only continue if spd fulfills conditions of a density
                # and we have more than 1 day until maturity
                if time_to_maturity_in_days > 1:

                    # Todo: Compare svcj results to old hd results
                    for simmethod in simmethods:

                        #hockeystick(sub, tau, curr_day, r_bandwidth)

                        # Todo:
                        # Tau needs to match for sp500 data and deribit data!!
                        # is like 0.39 for sp500 data!!

                        print(conf_fname)
                        
                        #print('Use Synthetic BTC Index here')
                        # For Mongo deribit_orderbooks
                        spd_btc, tau_btc = bootstrap(conf_fname, 'data/BTC_USD_Quandl.csv', rdate_f, tau, simmethod, r, 'out/deribit/', r_bandwidth)
                        
                        if spd_btc is not None:

                            # Saving Moneyness, SPD, PK, Confidence Bands
                            spd_btc.to_csv('out/movies/btc_pk_' + str(tau) + '_' + str(curr_day_starttime) + '.csv')
                            
                            # Also save data for Vola Smile and Term Structure
                            sub.to_csv('out/movies/sub_' + str(tau) + '_' + str(curr_day_starttime) + '.csv')


                        #pdb.set_trace()
                        # For SP500 data
                        #spd_sp500, tau_sp500 = bootstrap('data/SP500_OMON_sep_multi.csv', 'data/gspc.csv', rdate_f, tau, simmethod, r, 'out/sp500/', r_bandwidth)
                        #if spd_sp500 is not None:
                        #    spd_sp500.to_csv('out/movies/sp500_pk_' + str(tau_sp500[0]) + '_' + str(curr_day_starttime) + '.csv')

                        # Combine in Plot
                        #if spd_btc is not None and spd_sp500 is not None:
                        #    plot_epks(spd_btc, spd_sp500, tau_btc[0], tau_sp500[0], simmethod, curr_day_starttime, r_bandwidth)
                        
                else:
                    print('SPD is not a valid density, proceeding with the next one')

            except Exception as e:
                print('error: ', e)
                errors.append(e)
                with open("out/errors.txt", "wb") as fp:   #Pickling
                    pickle.dump(errors, fp)
            
        #finally:
        #    curr_day += datetime.timedelta(1)
        """


if __name__ == '__main__':
    print('starting non multi main...')

    brc = BRC()

    # Debugging Start, End
    startdate = datetime.datetime(2021,1,1)
    enddate = datetime.datetime(2023,1,1)
    run_dates = [startdate]
    curr_date = startdate
    ndays_shift = 2
    do_plot = False
    use_rookley = True
    print('Rookley activated?! ', use_rookley)

    out = []
    
    while curr_date < enddate:
        curr_date += datetime.timedelta(1)

        run_dates.append(curr_date)

    # Create a dataframe of instruments which should exist until maturity. 
    # Then compare the current day to it and query all for that we require a price

    for d in run_dates:
        print('day: ', d)
        out.append(run(d))
    transaction_df = pd.concat(out, axis = 0, ignore_index = True)
    instrument_collector = []

        
    # Enforce Daily Existence of each instrument
    def stratify_instruments(enf):

        maturitydate = enf['maturitydate_trading'].iloc[0]
        strike = enf['strike'].iloc[0]
        is_call = enf['is_call'].iloc[0]
        rng = pd.date_range(enf['day'].min(), maturitydate)
        
        wrk = pd.DataFrame({'Date': rng, 'strike': strike, 'maturitydate_trading': maturitydate, 'is_call': is_call})

        # Add days to maturity
        wrk['days_to_maturity'] = wrk.apply(lambda x: x['maturitydate_trading'].day - x['Date'].day, axis = 1)

        return wrk

    instrument_collector = transaction_df.groupby(['instrument_name']).apply(lambda x: stratify_instruments(x))
    instrument_df = instrument_collector.reset_index()

    #instrument_df = pd.concat(instrument_collector, axis = 0, ignore_index = True)

    counter = -1
    collected_out = []
    for d in run_dates:
        counter += 1
        #if counter == 100:
        #    break;
        try:
            print(d)
            #out.append(run(d))
            
            # 1) ENSURE THAT EACH INSTRUMENT EXISTS UNTIL MATURITY 
            daily_instruments = instrument_df.loc[(instrument_df['Date'] == d)]
            #requried_instruments = required_instruments_df['instrument_name'].unique()

            # Feed static properties of required instruments to the prediction / vola fit
            # Feed all variables that we need later
            #daily_instruments = required_instruments_df[['instrument_name', 'Date', 'strike', 'maturitydate_trading', 'is_call', 'days_to_maturity']].drop_duplicates()

            #pdb.set_trace()
            # 2) Assign None to instrument_price if it doesn't exist

            #_____________________________________

            # Filter around specific times of the day, possibly when the most trading activity occurs. 
            # Otherwise we have too much variation in this!
            filtered = filter_sub(out[counter])
            #instrument_df = pd.concat(out, axis = 0, ignore_index = True)
            
            # Python Daily observation close to specific time
            # https://stackoverflow.com/questions/42208206/find-daily-observation-closest-to-specific-time-for-irregularly-spaced-data
            filtered.set_index('date', inplace = True)
            sub = filtered.iloc[filtered.index.indexer_between_time("15:30:00", "16:30:00")]

            # Use last price to determine moneyness for daily_instruments
            last_spot = sub['index_price'][-1]
            daily_instruments['spot'] = last_spot # for Black Scholes Call 
            daily_instruments['moneyness'] = daily_instruments['strike'] / last_spot
            
            # maturitydate - current day to tau!
            Tdiff = (daily_instruments['maturitydate_trading'] - d)
            sec_to_date_factor   = 60*60*24
            _Tau                = list(map(lambda x: (x.days + (x.seconds/sec_to_date_factor)) / 365, Tdiff))#Tdiff/365 #list(map(lambda x: x.days/365, Tdiff)) # else: Tdiff/365
            
            daily_instruments['tau'] = _Tau
            daily_instruments = assign_groups(daily_instruments)

            #daily_instruments = sub.copy(deep = True) # HERE!!!
            predicted_iv = calibrate_on_iv_surface(sub, predict_sub = daily_instruments, curr_day = d)
            daily_instruments['predicted_iv'] = predicted_iv
            daily_instruments['day'] = d
            #pdb.set_trace()
            # Add _id to pred
            
            
            if use_rookley:
                #@Todo: Compare this to rookley
                #@Todo: Minimum amount of observations!
                #pdb.set_trace()


                # @Todo: this needs to be fitted on daily_instruments!!!
                predicted = sub.copy(deep = True)
                for week in sub['nweeks'].unique():
                    #test_rookley = smoothing_rookley(sub, sub['moneyness'], sub['tau'], 0.1, 0.1)
                    # Run separately for each week and predict
                    idx = sub.loc[sub['nweeks'] == week].index
                    if len(idx) > 0:
                        rookley_fit = rookley(sub.loc[idx])
                        moneyness_fit = rookley_fit[3]
                        iv_fit = rookley_fit[0]
                        # for prediction, just find location of moneyness of the value to-be-predicted
                        # then take the vola at the same point
                        # GOT AN ERROR HERE: 
                        # for each element in sub['moneyness'], find the closest matching one in moneyness_fit
                        # Then select iv at the same location - should have length of 333 for the first one!
                        daily_idx = daily_instruments.loc[daily_instruments['nweeks'] == week].index
                        moneyness_loc = daily_instruments.loc[daily_idx].apply(lambda x: find_nearest_location(moneyness_fit,x['moneyness'] ), axis = 1)
                        #find_nearest(moneyness_fit, x0)
                        
                        # Choose iv at location of moneyness_loc
                        #@Todo: ERROR HERE!!!!
                        daily_instruments.loc[daily_idx, 'rookley_predicted_iv'] = iv_fit[moneyness_loc]
                    else:
                        print('Not subscriptable')
                        #pdb.set_trace()

                    # Join on _id with pred

            #if counter == 10:
            #    pdb.set_trace()
            #filtered_pred = pred.loc[(pred['moneyness'] <= 1.3) & (pred['predicted_iv'] <= 2.5)]
            collected_out.append(daily_instruments)

        except Exception as e:
            print('error in : ', e)
            #pdb.set_trace()
    
    # Compare Rookley IV vs predicted_iv
    out_df = pd.concat(collected_out, ignore_index = True)
    pd.DataFrame(out_df).to_csv('out/fitted_data.csv')
