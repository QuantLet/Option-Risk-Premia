# Compare Performance of analyzeZeroBetaStraddles

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import csv
import statsmodels.api as sm
import pylab
from scipy import stats 
from pathlib import Path

def QQ_plot(data, nn, fname):
    # Source: https://stackoverflow.com/questions/13865596/quantile-quantile-plot-using-scipy
    fig = plt.figure(figsize = (10,7))

    # Sort as increasing
    y = np.sort(data)
    
    # Compute sample mean and std
    mean, std = np.mean(y), np.std(y)
    
    # Compute set of Normal quantiles
    ppf = stats.t(df = nn, loc = mean, scale = std).ppf
    #ppf = stats.norm(loc=mean, scale=std).ppf # Inverse CDF
    N = len(y)
    x = [ppf( i/(N+2) ) for i in range(1,N+1)]

    # Make the QQ scatter plot
    plt.scatter(x, y)
    
    # Plot diagonal line
    dmin, dmax = np.min([x,y]), np.max([x,y])
    diag = np.linspace(dmin, dmax, 1000)
    plt.plot(diag, diag, color='red', linestyle='--')
    plt.gca().set_aspect('equal')
    
    # Add labels
    plt.xlabel('Normal quantiles')
    plt.ylabel('Sample quantiles')
    plt.show()
    #plt.savefig(fname)


def wilcoxon_test(dat, out_path, min_moneyness = 0.7, max_moneyness = 1.3):
    """
    Groups returns by levels of moneyness (in steps of 0.05)
    performs kernel density estimation
    performs a Wilcoxon test on a population median around 0
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    pdct = {}
    df = dat.copy(deep = True)

    # Loop over moneyness levels
    for mn in df['moneyness'].sort_values().unique():
        if mn >= min_moneyness and mn <= max_moneyness:

            fig = plt.figure(figsize = (20, 14))
            higher = round(mn + 0.05, 2)
            sub = df.loc[(df['moneyness'] <= higher) & (df['moneyness'] >= mn)]
            result = stats.wilcoxon(sub['combined_ret_daily'], alternative='less')
            
            print(sub.describe())
            print('Moneyness Level: ', mn)
            print("Wilcoxon-statistic value: ", result.statistic)
            print("P-Value: ", result.pvalue)
            pdct[mn] = result.pvalue

    print(pdct)

    # Cool! 
    fig = plt.figure(figsize = (20, 14))
    sub = df.loc[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
    wil_stat, p_value = stats.wilcoxon(sub['combined_ret_daily'], alternative='less')
    sub['combined_ret_daily'].plot.kde()
    plt.title('n = ' + str(sub.shape[0]))
    plt.xlim((-1, 1))
    plt.savefig(out_path + '/moneyness_between_95_and_105' + '.png', transparent = True)
    print("Wilcoxon-statistic value: ", wil_stat)

    with open(out_path + '/wilcoxon-tests.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, pdct.keys())
        w.writeheader()
        w.writerow(pdct)
    
    return pdct

# Config
# Choose BTC or ETH
base = 'btc/' # 'eth/'

pf = 'performance_overview.csv'
p1 = base + 'out/vanilla/fees/' + pf
p2 = base + 'out/vanilla/no_fees/' + pf
p3 = base + 'out/crash_resistant/fees/' + pf
p4 = base + 'out/crash_resistant/no_fees/' + pf

vanilla_fee = pd.read_csv(p1)
vanilla_no_fee = pd.read_csv(p2)
crash_resistant_fee = pd.read_csv(p3)
crash_resistant_no_fee = pd.read_csv(p4)

for df in [vanilla_fee, vanilla_no_fee, crash_resistant_fee, crash_resistant_no_fee]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["combined_payoff_daily", "combined_ret_daily"], how="all", inplace=True)

print(vanilla_fee.loc[(vanilla_fee['moneyness'] >= 0.95) & (vanilla_fee['moneyness'] <= 1.05)][['combined_ret_daily']].describe())
print(crash_resistant_fee.loc[(crash_resistant_fee['moneyness'] >= 0.95) & (crash_resistant_fee['moneyness'] <= 1.05)][['combined_ret_daily']].describe())

# Test significance
p1_values = wilcoxon_test(vanilla_no_fee, out_path = base + 'out/vanilla/no_fees/density')
p2_values = wilcoxon_test(vanilla_fee, out_path = base + 'out/vanilla/fees/density')
p3_values = wilcoxon_test(crash_resistant_fee, out_path = base + 'out/crash_resistant/fees/density')
p4_values = wilcoxon_test(crash_resistant_no_fee, out_path = base + 'out/crash_resistant/no_fees/density')
pv = pd.DataFrame({'Vanilla ex Fee': p1_values, 'Vanilla cum Fee': p2_values, 'Crash Resistant ex Fee': p3_values, 'Crash Resistant cum Fee': p4_values})
pv.round(4).to_csv(base + 'out/pvalues.csv')

vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()

vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()

