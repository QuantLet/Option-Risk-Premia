# Empirial analysis of the market volatility premium for digital assets (BTC, ETH)

1) Theoretical Reasoning
- Options are theoretically priced via no-arbitrage arguments. 
- In reality, some concepts underlying option pricing do not hold: Especially no transaction cost, continuous rebalancing and knowledge of volatility
- Future volatility is unknown and even the present one is difficult to determine (different windows)
- For options on equity indices, it is well known that diversions from the theoretical reasonings exist: A portfolio of a long call and delta hedge should earn the risk free rate, but this is not the case. A portfolio of straddles loses approx. 3% weekly...
- How does this look like for digital assets? Do we see a market volatility premium?

2) Derivation
- BlackScholes implies that the present value of a call is the discounted value of the replicating portfolio. 
- Therefore, (theoretically) holding the long call and replicating the PF should yield the risk-free rate.
- But does this really count in practice? 

3) Steps
- Load transactions from DB
- At the cutoff time, cumulate all existing calls and perform Rookley
- With the Rookley smoothing, we have a market-based price for each call for each day
- Simulate holding the long call and replicating the PF with a delta-hedge
- Assess the performance


on Filtering...
It is important to select a range of moneyness in which our results are evaluated. 
First, in order to make instruments comparable. Second, to mitigate the selection bias due to Missing-not-at-Random instruments and changes in instrument frequency and available Strikes. 

4) Further Thoughts & Interpretation: 
- How does this relate to the Vola Premium between IV and Realized Vola?! Is the resulting performance explained by the surplus?!
- Premia looks high, but pricing in crypto-specific risk, fat tail events, counterparty risk (single exchange in Panama), hack risks etc. 
- Adjusting for the crypto-specific risk, probably not that much higher than in traditional equity markets... maybe compare to some exotic currencies or commodities


Todo
- Verify Delta Hedge
- Add Interest Rates / This is coded in src/brc.py but on test mode and needs to be integrated
- More complex Portfolios, e.g. Call Spread etc. 
- Add Puts
- New Performance Measures - compare to literature 
- Add Moneyness Level / Are FOTM Options related to less vola premium?

    # @Todo: PnL per Group: First time-to-maturity, moneyness. Also show over time. 
    # Compare to Initial IV and Difference between Initial IV to running average. 
    # Check Jackwerth

    #@Todo: 
    # 1) Performance over Time (use start date and end date)
    # 2) Relate this to IV vs Realized Vola Premium
    # 3) Take actual expiration price for final instrument price instead of an estimation!!!
    # 4) For #1, Calculate Mean IV and Realized Variance!
    # 5) Restrict for ATM instruments, only count each once!
    # Exclude Outliers


Thoughts on Option Beta
    -  Are Betas of 50 for ATM Calls realistic? d1 is correct, but a call price of 100 relative to the spot magnifies the Beta a lot. 
    e.g. call_price = 100, spot = 10000 delivers beta of 10000/100 * 0.5 = 50
    Possible Answer: 
    If we buy the underlying at the spot of 10000 and at option maturity it rose to 10500, then the return of holding the underlying woulud be 5%.
    Lets assume a risk free rate of 0, so the excess return in CAPM would also be 5%.
    Holding the option, however, would result in a return of 500% (paid 100 for the option, payoff is 500). The excess return would also be 
    500%. So we have a factor of 100 between the excess returns of the two portfolios.
    Following http://www.timworrall.com/fin-40008/optionrisk.pdf on page 2, the elasticity of an option is Omega = (Spot / Call Price) * OptionDelta.
    That is the same formula which we use for the option beta calculation (Shumway). The rest is explained on the following pages.
    Beta_s is the beta of the market portfolio, which in our example would be 5%. So the Beta of the Option would be (10000/100) * 5% = 5


Thoughts on Negative Vola Premium
perhaps: Vola Premium = RV - IV

Next Steps:
Ensure matching for missing puts (match with put-call-parity)
Rerun from 2017
Performance Measures

Get IRR: Combined Return over Tau

Check if Expiration Price is correct!!!!
Also whats used for the final pricing!!!

Show the PnL over time and relate to Vola Premium (IV vs RV)

Translate the IV to percent change and compare to the percent change of expiration price 

Plot: Vola Premium (IV - RV) AND daily Straddle Returns for Tau in a single plot
Maybe group by Tau

Get Correlation between Vola Premium and Straddle Returns
Does Vola Premium Today have predictive power for Straddle Returns tomorrow?

# Vola Methods:
    - Rookley
    - Regression IV Surface
    - None 
    Try None and see if results are robust


# Next important Thing:
Condition on where Moneyness is in [0.95, 1.05] and plot the Returns over Tau! 
That one has the biggest potential!

# Make a run with and without overwriting spot with expiration price. We are cutting off some of the time-to-maturity here. 

So ATM Straddles are losing lots of value. Introduce Crash-resistant Straddles!
Check with the paper why ATM Straddles are losing so much value. 

# Thoughts on Centering with respect to Expiration Price
- Pro: Nice to start on the same time every day
- Con: Centering might distort the results, since IV is dependent on Moneyness (and we are re-centering the spot, which changes moneyness)
- Con: When trading a call and a put from a different time, then they have a different spot. "unrealistic" straddles
Solution: Run the analysis centered around expiration price and then confirm results by checking if they match with a subset (window) around the expiration price
Another Solution: ALWAYS use Put-Call-Parity, so that we are always dealing with 2 instruments around the same spot!

# Graphics
- Combined Payoff vs Combined Return plot should look the same!! 
- So maybe filter around times?! 

# Plan
Use every option exactly how and when it traded and complete the straddle using put call parity 
Get average return of the instruments 
How much do puts lose on avg?
Statistical Tests
Crash-free Straddles

Check Formula for Weights!!!

