# Compare Performance of analyzeZeroBetaStraddles

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
from scipy import stats 
from scipy.stats import ttest_1samp
import time

p1 = 'out/vanilla/fees/performance_overview.csv'
p2 = 'out/vanilla/no_fees/performance_overview.csv'

vanilla_fee = pd.read_csv(p1)
vanilla_no_fee = pd.read_csv(p2)

# InterestRates start on this date
#print('Restricting date!')
#vanilla_fee['date'] = pd.to_datetime(vanilla_fee['day'])
#dat = vanilla_fee.loc[vanilla_fee['day'] >= '2019-05-01']
#pdb.set_trace()

for df in [vanilla_fee, vanilla_no_fee]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["combined_payoff_daily", "combined_ret_daily"], how="all", inplace=True)

vanilla_fee[['combined_payoff']].describe()
vanilla_no_fee[['combined_payoff']].describe()

# Test significance
out_path = 'out/vanilla/fees/density'
Path(out_path).mkdir(parents=True, exist_ok=True)
pdct = {}
df = vanilla_no_fee.copy(deep = True)
for mn in df['moneyness'].sort_values().unique():

    fig = plt.figure(figsize = (20, 14))
    higher = mn + 0.05
    sub = df.loc[(df['moneyness'] <= higher) & (df['moneyness'] >= mn)]
    sub['combined_ret_daily'].plot.kde()
    plt.title('n = ' + str(sub.shape[0]))
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
sub = df.loc[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
t_stat, p_value = ttest_1samp(sub['combined_ret_daily'], popmean=0, alternative = 'less', nan_policy = 'omit') #, alternative = 'less'



pdb.set_trace()
vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_payoff'].describe()

vanilla_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()
vanilla_no_fee.loc[vanilla_fee['days_to_maturity'] <= 20].groupby('days_to_maturity')['combined_ret'].describe()

# Get average option returns etc. 
# average index returns