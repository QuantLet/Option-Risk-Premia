import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def decompose_instrument_name(_instrument_names, tradedate, round_tau_digits = 4):
    """
    Input:
        instrument names, as e.g. pandas column / series
        in this format: 'BTC-6MAR20-8750-C'
    Output:
        Pandas df consisting of:
            Decomposes name of an instrument into
            Strike K
            Maturity Date T
            Type (Call | Put)
    """
    try:
        _split = _instrument_names.str.split('-', expand = True)
        _split.columns = ['base', 'maturity', 'strike', 'is_call'] 
        
        # call == 1, put == 0 in is_call
        _split['is_call'] = _split['is_call'].replace('C', 1)
        _split['is_call'] = _split['is_call'].replace('P', 0)

        # Calculate Tau; being time to maturity
        #Error here: time data '27MAR20' does not match format '%d%b%y'
        _split['maturitystr'] = _split['maturity'].astype(str)
        # Funny Error: datetime does recognize MAR with German AE instead of A
        maturitydate        = list(map(lambda x: datetime.datetime.strptime(x, '%d%b%y') + datetime.timedelta(hours = 8), _split['maturitystr'])) # always 8 o clock
        reference_date      = tradedate.dt.date #list(map(lambda x: x.dt.date, tradedate))#tradedate.dt.date # Round to date, else the taus are all unique and the rounding creates different looking maturities
        Tdiff               = pd.Series(maturitydate).dt.date - reference_date #list(map(lambda x: x - reference_date, maturitydate))
        Tdiff               = Tdiff[:len(maturitydate)]
        sec_to_date_factor   = 60*60*24
        _Tau                = list(map(lambda x: (x.days + (x.seconds/sec_to_date_factor)) / 365, Tdiff))#Tdiff/365 #list(map(lambda x: x.days/365, Tdiff)) # else: Tdiff/365
        _split['tau']       = _Tau
        _split['tau']       = round(_split['tau'], round_tau_digits)

        # Strike must be float
        _split['strike'] =    _split['strike'].astype(float)

        # Add maturitydate for trading simulation
        _split['maturitydate_trading'] = maturitydate
        _split['days_to_maturity'] = list(map(lambda x: x.days, Tdiff))

        print('\nExtracted taus: ', _split['tau'].unique(), '\nExtracted Maturities: ',_split['maturity'].unique())

    except Exception as e:
        print('Error in Decomposition: ', e)
    finally:
        return _split

def decompose_future_name(_instrument_names, tradedate, round_tau_digits = 4):
    """
    Input:
        instrument names, as e.g. pandas column / series
        in this format: 'BTC-6MAR20-8750-C'
    Output:
        Pandas df consisting of:
            Decomposes name of an instrument into
            Strike K
            Maturity Date T
            Type (Call | Put)
    """
    try:
        _split = _instrument_names.str.split('-', expand = True)
        _split.columns = ['base', 'maturity'] 

        # Calculate Tau; being time to maturity
        _split['maturitystr'] = _split['maturity'].astype(str)
       
        maturitydate        = list(map(lambda x: datetime.datetime.strptime(x, '%d%b%y') + datetime.timedelta(hours = 8), _split['maturitystr'])) # always 8 o clock
        reference_date      = tradedate.dt.date
        Tdiff               = pd.Series(maturitydate).dt.date - reference_date 
        Tdiff               = Tdiff[:len(maturitydate)]
        sec_to_date_factor  = 60*60*24
        _Tau                = list(map(lambda x: (x.days + (x.seconds/sec_to_date_factor)) / 365, Tdiff))
        _split['tau']       = _Tau
        _split['tau']       = round(_split['tau'], round_tau_digits)

        # Add maturitydate for trading simulation
        _split['maturitydate_trading'] = maturitydate
        _split['days_to_maturity'] = list(map(lambda x: x.days, Tdiff))

        print('\nExtracted taus: ', _split['tau'].unique(), '\nExtracted Maturities: ',_split['maturity'].unique())

    except Exception as e:
        print('Error in Decomposition: ', e)
    finally:
        return _split


def find_instrument_before_cutoff(df):
    """
    https://stackoverflow.com/questions/42208206/find-daily-observation-closest-to-specific-time-for-irregularly-spaced-data
    """
    df['Time'] = df['date']
    df.set_index('Time', inplace = True)
    df.sort_index(inplace=True)  # Sort indices of original DF if not in sorted order
    # Create a lookup dataframe whose index is offsetted by 16 hours
    lookup = pd.DataFrame(dict(Time=pd.to_datetime(pd.unique(df.index.date)) + pd.tseries.offsets.Hour(16)))
    # Find values in original within +/- 30 minute interval of lookup 
    df.reindex(lookup['Time'], method='nearest', tolerance=pd.Timedelta('60Min'))
    # Find values in original within 30 minute interval of lookup (backwards)
    pd.merge_asof(lookup, df.reset_index(), on='Time', tolerance=pd.Timedelta('60Min'))
    # Tolerance of +/- 30 minutes from 16:00:00
    return df.iloc[df.index.indexer_between_time("15:30:00", "16:30:00")]

def find_nearest_location(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx - 1 #array[idx-1]
    else:
        return idx #array[idx]

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

def load_expiration_price_history(currency):
    """
    Expiration Price History from Deribit. 
    Used to Price Instruments around Expiration
    """
    if currency == 'btc':
        expi_dir = 'data/Expiration Price History.csv'
    elif currency == 'eth':
        expi_dir = 'data/eth_expiration_history.csv'
    expi_raw = pd.read_csv(expi_dir)
    expi_raw.rename(columns = {'Price': 'expiration_price'}, inplace = True)
    #expi_raw['Date'] = pd.to_datetime(expi_raw['Date'])
    expi_rev = expi_raw[::-1]
    return expi_rev

def compute_vola(x, window = 1):

    dpy = 365 #252  # trading days per year
    ann_factor = (dpy / window)**0.5

    df = pd.DataFrame({'price' : x})
    df['real_vol'] = df['price'].pct_change().std() * ann_factor # .rolling(window)

    return df['real_vol']

