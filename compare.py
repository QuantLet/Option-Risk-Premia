# Compare Performance of analyzeZeroBetaStraddles

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pdb
import time
import csv
from scipy import stats 
from scipy.stats import ttest_1samp
from pathlib import Path

def ttest(dat, out_path, min_moneyness = 0.7, max_moneyness = 1.3):
    """
    Groups returns by levels of moneyness (in steps of 0.05)
    performs kernel density estimation
    performs a left-sided t-test on a population mean of 0
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)
    pdct = {}
    df = dat.copy(deep = True)
    for mn in df['moneyness'].sort_values().unique():
        if mn >= min_moneyness and mn <= max_moneyness:

            fig = plt.figure(figsize = (20, 14))
            higher = round(mn + 0.05, 2)
            sub = df.loc[(df['moneyness'] <= higher) & (df['moneyness'] >= mn)]
            sub['combined_ret_daily'].plot.kde()
            plt.title('n = ' + str(sub.shape[0]))
            plt.xlim((-1, 1))
            plt.savefig(out_path + '/moneyness>=' + str(mn) + '&moneyness<=' + str(higher) + '.png')
            t_stat, p_value = ttest_1samp(sub['combined_ret_daily'], popmean=0, alternative = 'less', nan_policy = 'omit') #, alternative = 'less'
            print(sub.describe())
            print('Moneyness Level: ', mn)
            print("T-statistic value: ", t_stat)
            print("P-Value: ", p_value)
            pdct[mn] = p_value
            #time.sleep(3)

    print(pdct)

    # Cool! 
    fig = plt.figure(figsize = (20, 14))
    sub = df.loc[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
    t_stat, p_value = ttest_1samp(sub['combined_ret_daily'], popmean=0, alternative = 'less', nan_policy = 'omit') #, alternative = 'less'
    sub['combined_ret_daily'].plot.kde()
    plt.title('n = ' + str(sub.shape[0]))
    plt.xlim((-1, 1))
    plt.savefig(out_path + '/moneyness_between_95_and_105' + '.png')

    with open(out_path + '/t-tests.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, pdct.keys())
        w.writeheader()
        w.writerow(pdct)
    
    return pdct

# Config
# Choose BTC or ETH
base = 'eth/'


pf = 'performance_overview.csv'
p1 = base + 'out/vanilla/fees/' + pf
p2 = base + 'out/vanilla/no_fees/' + pf
p3 = base + 'out/crash_resistant/fees/' + pf
p4 = base + 'out/crash_resistant/no_fees/' + pf

vanilla_fee = pd.read_csv(p1)
vanilla_no_fee = pd.read_csv(p2)
crash_resistant_fee = pd.read_csv(p3)
crash_resistant_no_fee = pd.read_csv(p4)

# InterestRates start on this date
#print('Restricting date!')
#vanilla_fee['date'] = pd.to_datetime(vanilla_fee['day'])
#dat = vanilla_fee.loc[vanilla_fee['day'] >= '2019-05-01']
#pdb.set_trace()

for df in [vanilla_fee, vanilla_no_fee, crash_resistant_fee, crash_resistant_no_fee]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["combined_payoff_daily", "combined_ret_daily"], how="all", inplace=True)

vanilla_fee[['combined_payoff']].describe()
vanilla_no_fee[['combined_payoff']].describe()

# Test significance
p1_values = ttest(vanilla_no_fee, out_path = base + 'out/vanilla/no_fees/density')
p2_values = ttest(vanilla_fee, out_path = base + 'out/vanilla/fees/density')
p3_values = ttest(crash_resistant_fee, out_path = base + 'out/crash_resistant/fees/density')
p4_values = ttest(crash_resistant_no_fee, out_path = base + 'out/crash_resistant/no_fees/density')

pv = pd.DataFrame({'Vanilla ex Fee': p1_values, 'Vanilla cum Fee': p2_values, 'Crash Resistant ex Fee': p3_values, 'Crash Resistant cum Fee': p4_values})
pv.round(4).to_csv(base + 'out/pvalues.csv')

pdb.set_trace()
vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()

vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()

