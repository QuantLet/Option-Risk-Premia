
The valuation of Bitcoin has been assessed in terms of ...

Another, so far rarely used pathway is the assessment of the risk-return characteristics of Bitcoin through the lens of the options written on it, which allow new model-free insights into the existence and degree of a leverage effect, the existence and size of risk premia, resulting market inefficiencies (measurable in economic loss / money lost) and to explore the option (pun intended)/possibility of curing named inefficiencies under transaction cost and liquidity constraints. 

This reasoning follows the insights of Shumway and Bakshi, who separately evaluated option returns in equity markets by means of delta-neutral option positions in the early 2000s. 

Bakshi and Shumway both found that expected call returns exceed those of the underlying and that expected put returns underperform the underlying. Both also found that the effect increases (decreases) with the strike. 
Bakshi found that zero-beta, at-the-money straddle positions produce average losses of approximately three percent per week and suggests that the reason may be a systematic volatility premium being priced in options. 

Following Shumway, the risk-return characteristics of options can be separated in two elements, a leverage effect and a curvature effect due to the sensitivity of option payoffs to higher moments of the underlying. 
First, the holder (buyer) of an option is subject to a leverage effect since exposure to the underlying can be obtained through the payment of a premium, which is small compared to the price of the underlying. 
If there is a leverage effect, then (long) call options should outperform their underlying and (long) put options should underperform their underlying. Furthermore, the effect should increase (decrease) with the strike price for calls (puts).
This reasoning allows a simple test of existence and size of the leverage effect and thus conclusions about the level of speculation associated to Bitcoin valuation. 

(The second component, related to the sensitivity of options towards higher moments, implies that options are redundant assets...)

If options were redundant assets, then apart of the leverage effect, delta-neutral option portfolios should earn the risk-free rate (no premium) on average.

# compare page 2 end
1) get raw returns of calls and puts
avg return grouped by option_type
also get returns of the index
for sp500: calls make about 2% per week, puts lose about 9% per week. the index makes XY

# compare early page 3
are the returns of zero-beta-straddles significantly different than the risk-free rate?!
What are the returns and are they robust (vanilla vs crash-resistant)?!

from shumway: These results indicate that options are earning low returns for reasons which extend beyond their ability to insure crashes. Moreover, the results are not eliminated when transaction costs are considered. 

Shumway considers following strategy: 
sell each month equal numbers of atm calls and atm puts at the bid, then purchase the same number of otm puts at the ask and invest the remianing premium and principal in the index. sharpe ratio outperforms investing in the index (twice as high).
Perhaps should consider doing this on the short side in analyzeZeroBetaStraddles.py .


# The Risk-free rate in Crypto Markets
Evidently, there is no established risk-free rate in crypto (or digital asset) markets. In the absence of riskless bond, we refer to the next best thing as a proxy for the risk free rate, which is the replication of the market-implied yield through a position which only has counterparty risk. To be fair, counterparty risk also exists for risk-free bonds. 

Crypto exchanges often provide a so-called perpetual future, which is a synthetic (typically cash-settled instrument) whose return is tied to the underlying via a funding rate. Most exchanges (quote here) provide perpetual futures for Bitcoin and Ethereum. 
The price of each perpetual future is linked to the underlying by adjusting a variable interest rate, called the funding rate, that fluctuates depending on the deviation between the price of the perpetual future and the underlying. The funding rate increases the cost of holding positions against the deviation, meaning that if the price of the perpetual future is higher than the underlying, the funding rate is positive which causes the market participants who are long to pay the short side in a contractually predefined frequency. Broadly said, the more extreme the deviation, the higher the funding rate and thus the higher the incentive to trade against the deviation and thus converge the price of the perpetual future towards its underlying. 
The risk-free rate is approximated using the simplest and most liquid way of realising the funding rate, a carry trade. In such a trade, the trader shorts (longs) the perpetual future if the funding rate is positive (negative) and simulateously buys (sells) the underlying such that the market delta of the combined position is zero. 






Option Returns are used for three reasons.
1) tests basd on option returns permit us to dirctly examine whether the leverage effect is priced without imposing any particular model
2) violations of market efficiency can be directly measured in economic terms
2.5) can mispricing, if it exists, be exploited or corrected in the presence of transaction cost
3) we can consider the economic payoff for taking very particular risks




Options enable the pricing of bespoke risks. 
The study of option prices allows insight into the price of an asset's risk.



risky securities should compensate their holders iwth expected returns that are in accordance with the systematic risks they require their holder to bear. options which deliver payoff in bad states of the world will earn lower returns than those that deliver their payoffs in good states.

separate option risk into the leverage effect and the sensitivity to higher order factors


The Black-Scholes model implies that this implicit leverage, which is reflected in option betas, should be priced. We show that this leverage should be priced under much more general conditions than the Black-Scholes assumptions. In particular, call options written on securities with expected returns above the risk-free rate should earn expected returns which exceed those of the underlying security. Put options should earn expected returns below that of the underlying security. We present strong empirical support for the pricing of the leverage effect.
The second component of option risk is related to the curvature of option payoffs. Since option values are nonlinear functions of underlying asset values, option returns are sensitive to the higher moments of the underlying asset’s returns. The Black-Scholes model assumes that any risks associated with higher moments of the underlying assets returns are not priced because asset returns follow geometric Brownian motion, a two-parameter process. This is equivalent to assuming that options are redundant assets. One testable implication of the redundancy assumption is that, net of the leverage effect, options portfolios should earn no risk premium. In other words, options positions that are delta-neutral (have a market beta of zero) should earn the risk-free rate on average. We test this hypothesis, and find quite robust evidence to reject it.


Test average Returns of Calls and Puts
First testable hypotheses is call returns > put returns

Moreover, the expected returns of both calls and puts are increasing in the strike price. To test whether option returns are consistent with these implications of the leverage effect, we examine payoffs associated with options on the S&P 500 and S&P 100 indices. We find that call options earn returns which are significantly larger than those of the underlying index, averaging roughly two percent per week. On the other hand, put options receive returns which are significantly below the risk-free rate, losing between eight and nine percent per week. We also find, consistent with the theory, that returns to both calls and puts are increasing

