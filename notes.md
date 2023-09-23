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
