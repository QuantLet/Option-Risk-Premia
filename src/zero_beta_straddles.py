from matplotlib.pyplot import thetagrids
import numpy as np
from scipy.stats import norm

from src.blackscholes import Call, Put, d1
import pdb

# Easier Version for both, Call and Put Betas together
def get_beta(s, k, r, vol, tau, is_call, _lambda = 0, beta_s = 1):
    """
    Calculate Black Scholes Beta for Put OR Call
    as Equation (11) in Shumway: Expected Option Returns

    s: spot
    k: strike
    r: interest rate
    vol: IV
    tau: time-to-maturity
    is_call: boolean. 1 for call, 0 for put
    _lambda: dividend yield is always 0 for BTC, ETH
    beta_s: always equals1, following page 14 Shumway
    """

    N = norm.cdf
    if is_call == 1:
        price = Call.Price(s, k, r, vol, tau)
        delta = Call.Delta(s, k, r, vol, tau)
    elif is_call == 0:
        price = Put.Price(s, k, r, vol, tau)
        delta = Put.Delta(s, k, r, vol, tau)
    else:
        raise NotImplementedError('Neither Put nor Call')

    beta = (s/price) * delta * beta_s
    return beta

# Easier Version for both, Call and Put Betas together
def get_combined_beta(p1, p2, delta1, delta2, spot ,_lambda = 0, beta_s = 1):
    """
    Calculate Black Scholes Beta for Put OR Call
    as Equation (11) in Shumway: Expected Option Returns

    s: spot
    k: strike
    r: interest rate
    vol: IV
    tau: time-to-maturity
    is_call: boolean. 1 for call, 0 for put
    _lambda: dividend yield is always 0 for BTC, ETH
    beta_s: always equals1, following page 14 Shumway
    """

    beta = (spot/(p1 - p2)) * (delta1 - delta2) * beta_s
    return beta

def get_call_beta(s, k, r, vol, tau, _lambda = 0, beta_s = 1):
    """
    Calculate Black Scholes Call Beta
    as Equation (11) in Shumway: Expected Option Returns

    s: spot
    k: strike
    r: interest rate
    vol: IV
    tau: time-to-maturity
    _lambda: dividend yield is always 0 for BTC, ETH
    beta_s: always equals1, following page 14 Shumway
    """

    N = norm.cdf
    call_price = Call.Price(s, k, r, vol, tau)

    # Argument for Cumulative Normal Distribution
    #x = d1(s, k, r, vol, tau)
    x = (np.log(s/k) + ((r - _lambda + (0.5 * (vol**2))) * tau)) / (vol * tau**0.5)
    cdf = N(x)

    beta_c = (s/call_price) * cdf * beta_s
    return beta_c

def get_put_beta(call_price, put_price, spot, call_beta, beta_s = 1):
    """
    Calculated via Put-Call-Parity from Call Beta (get_call_beta)
    beta_s = 1 because it is the beta of holding the spot
    """
    put_beta = (1/put_price) * ((call_price * call_beta) - (spot * beta_s))
    return put_beta#max(put_beta, 0.01)

def get_straddle_weights(call_beta, put_beta, min_weight = 0.01, max_weight = 1):
    """
    Find Call Weights (theta) and Put Weights so that the overall market beta of holding both is 0
    theta is the fraction fo the straddle's value in call options
    returns call weight, put weight

    put_beta needs to be min 0.01, or this goes all to zero
    """
    call_weight = ((-1) * put_beta) / (call_beta - put_beta)
    put_weight = 1 - call_weight

    if call_weight <= min_weight or call_weight >= max_weight or put_weight <= min_weight or put_weight >= max_weight:
        call_weight, put_weight = None, None

    """
    if call_weight <= min_weight:
        print('Call Weight under Minimum: ', call_weight)
        call_weight = 0
    if call_weight >= max_weight:
        print('Call Weight over Maximum: ', call_weight)
        call_weight = max_weight
    if put_weight <= min_weight:
        print('Put Weight under Minimum: ', put_weight)
        put_weight = 0
    if put_weight >= max_weight:
        print('Put Weight over Maximum: ', put_weight)
        put_weight = max_weight
    """

    return call_weight, put_weight

def get_call_prices_via_put_call_parity(put_price, s, k, r, tau):
    """

    """
    call_price = put_price + s - (k * np.exp(-r * tau))
    return call_price

def get_put_prices_via_put_call_parity(call_price, s, k, r, tau):
    """

    """
    put_price = call_price - s + (k * np.exp(-r * tau))
    return put_price

"""
def zero_beta_straddle_return(spot, call_price, call_beta, ):

    Equation (18) Shumway
"""



if __name__ == '__main__':
    print('Starting Zero-Beta Test')
    spot = 10000
    strike = 10000
    interest_rate = 0
    vola = 0.5
    tau = 1/365

    call_price = Call.Price(spot, strike, interest_rate, vola, tau)
    put_price = Put.Price(spot, strike, interest_rate, vola, tau)

    call_beta = get_call_beta(spot, strike, interest_rate, vola, tau)
    put_beta = get_put_beta(call_price, put_price, spot, call_beta)

    print('\nCall Price: ', call_price, '\nPut Price: ', put_price, '\nCall Beta: ', call_beta, '\nPut Beta: ', put_beta)

    call_weights, put_weights = get_straddle_weights(call_beta, put_beta)
    print('\nTheta: ', call_weights, '\n1 - Theta: ', put_weights)

    # The return of an (already rebalanced?!) zero-beta straddle is defined in 
    # Equation (18)