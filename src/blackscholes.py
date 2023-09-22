from scipy import stats as scistat
import statsmodels.api as sm
from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r+sigma*sigma/2)*T)/(sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)

'''
Input parameters:
S -> asset price
K -> strike price
r -> interest rate
sigma -> volatility
T -> time to maturity
'''
class Call:        
    def Price(S, K, r, sigma, T):
        return np.maximum(S - K, 0) if T==0 else S*scistat.norm.cdf(d1(S, K, r, sigma, T)) - K*np.exp(-r*T)*scistat.norm.cdf(d2(S, K, r, sigma, T))

    def Delta(S, K, r, sigma, T):
        return scistat.norm.cdf(d1(S, K, r, sigma, T))

    def Gamma(S, K, r, sigma, T):
        return scistat.norm.pdf(d1(S, K, r, sigma, T))/(S*sigma*np.sqrt(T))

    def Vega(S, K, r, sigma, T):
        return S*scistat.norm.pdf(d1(S, K, r, sigma, T))*np.sqrt(T)

    def Theta(S, K, r, sigma, T):
        aux1 = -S*scistat.norm.pdf(d1(S, K, r, sigma, T))*sigma/(2*np.sqrt(T))
        aux2 = -r*K*np.exp(-r*T)*scistat.norm.cdf(d2(S, K, r, sigma, T))
        return aux1+aux2

    def Rho(S, K, r, sigma, T):
        return K*T*np.exp(-r*T)*scistat.norm.cdf(d2(S, K, r, sigma, T))

    '''
    Range calculations
    '''
    def Get_range_value(Smin, Smax, Sstep, K, r, sigma, T, num_curves, value="Price"):
        ssize = int((Smax - Smin) / Sstep)
        vec = np.linspace(Smin, Smax, ssize)
        vecT = np.linspace(0,T,num_curves, endpoint=True)
        if value=="Price":
            return vec,vecT, [[Call.Price(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Delta":
            return vec, vecT, [[Call.Delta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Gamma":
            return vec, vecT, [[Call.Gamma(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Vega":
            return vec, vecT, [[Call.Vega(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Theta":
            return vec, vecT, [[Call.Theta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Rho":
            return vec, vecT, [[Call.Rho(S, K, r, sigma, t) for S in vec] for t in vecT]

'''
Input parameters:
S -> asset price
K -> strike price
r -> interest rate
sigma -> volatility
T -> time to maturity
'''
class Put:
    def Price(S, K, r, sigma, T):
        return np.maximum(K-S,0) if T==0 else K*np.exp(-r*T)*scistat.norm.cdf(-1*d2(S, K, r, sigma, T)) - S*scistat.norm.cdf(-1*d1(S, K, r, sigma, T))

    def Delta(S, K, r, sigma, T):
        return scistat.norm.cdf(d1(S, K, r, sigma, T)) - 1

    def Gamma(S, K, r, sigma, T):
        return scistat.norm.pdf(d1(S, K, r, sigma, T))/(S*sigma*np.sqrt(T))

    def Vega(S, K, r, sigma, T):
        return S*scistat.norm.pdf(d1(S, K, r, sigma, T))*np.sqrt(T) * 0.01

    def Theta(S, K, r, sigma, T):
        aux1 = -S*scistat.norm.pdf(d1(S, K, r, sigma, T))*sigma/(2*np.sqrt(T))
        aux2 = r*K*np.exp(-r*T)*scistat.norm.cdf(-1*d2(S, K, r, sigma, T))
        return aux1+aux2

    def Rho(S, K, r, sigma, T):
        return -K*T*np.exp(-r*T)*scistat.norm.cdf(-1*d2(S, K, r, sigma, T))

    def Get_range_value(Smin, Smax, Sstep, K, r, sigma, T, num_curves, value="Price"):
        ssize = int((Smax - Smin) / Sstep)
        vec = np.linspace(Smin, Smax, ssize)
        vecT = np.linspace(0,T,num_curves, endpoint=True)
        if value=="Price":
            return vec,vecT, [[Put.Price(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Delta":
            return vec, vecT, [[Put.Delta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Gamma":
            return vec, vecT, [[Put.Gamma(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Vega":
            return vec, vecT, [[Put.Vega(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Theta":
            return vec, vecT, [[Put.Theta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value=="Rho":
            return vec, vecT, [[Put.Rho(S, K, r, sigma, t) for S in vec] for t in vecT]
