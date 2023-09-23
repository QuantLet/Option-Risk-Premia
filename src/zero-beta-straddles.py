import numpy as np
from scipy.stats import norm

from blackscholes import Call, Put, d1
import pdb

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
    return put_beta

def get_straddle_weights(call_beta, put_beta):
    """
    Find Call Weights (theta) and Put Weights so that the overall market beta of holding both is 0
    theta is the fraction fo the straddle's value in call options
    returns call weight, put weight
    """
    theta = ((-1) * put_beta) / (call_beta - put_beta)
    return theta, 1 - theta


# Will need the functions below in case their is no put-call pair with
# similar strike when constructing straddles
def call_price_via_parity():
    """
    Calculate Call Price via Put-Call-Parity

    """

def put_price_via_parity():
    """
    Calculate Put Price via Put-Call-Parity
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