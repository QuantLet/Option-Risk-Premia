import pdb

def delta_hedge(delta_vec, S, r, tau, N = 1):
    """
    delta_vec is vector of BS delta values
    S is vector of stock prices
    r is interest rate (const)
    tau is time to maturity, float
    N is amount of Calls

    Source: https://github.com/732jhy/Delta-Hedging/blob/master/Delta%20Hedging%20Sim.ipynb
    """

    #@Todo: Ensure that Delta is 0 at the end, we need to unravel the position
    delta_vec.append(0)
    S.append(S[-1]) # Underlying stays constant while we unravel


    dt = tau / len(delta_vec)

    # Initial Position
    S0 = S[0]
    n_shares = [delta_vec[0]*N]

    cost_of_shares = [n_shares[0]*S0]
    cumulative_cost = [n_shares[0]*S0]
    interest_cost = [cumulative_cost[0]*r*tau]

    # Rebalancing
    for i in range(1, len(delta_vec)):
        diff = delta_vec[i] - delta_vec[i-1]
        n_shares.append(diff*N)
        cost_of_shares.append(n_shares[i]*S[i])
        cumulative_cost.append(cost_of_shares[i] + cumulative_cost[i-1] + interest_cost[i-1])
        interest_cost.append(cumulative_cost[i]*r*dt)

    
    #if delta_vec[-2] > 0.01:
    #    pdb.set_trace()
    #raise ValueError('Delta must be 0 at last')

    return n_shares, cost_of_shares, cumulative_cost, interest_cost