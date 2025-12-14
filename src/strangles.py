import numpy as np
from scipy.stats import norm

def black_scholes_greeks_single(S, K, T, r, sigma, option_type='call'):
    """
    Calculates Price and Delta for a single option type using the BSM model.
    """
    # Guard against zero time-to-maturity
    if T < 1e-10:
        if option_type == 'call':
            return max(0, S - K), (1.0 if S > K else 0.0)
        else:
            return max(0, K - S), (-1.0 if S < K else 0.0)
            
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Price and Delta
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        # Delta = N(d1) - 1
        delta = norm.cdf(d1) - 1
    else:
        # Should not be reached
        return 0, 0
        
    return price, delta

def prove_beta_neutrality_implies_delta_neutrality_strangle():
    # --- BSM and Market Constants ---
    r = 0.05      # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)
    T = 0.25      # Time-to-maturity (3 months)
    BETA_S = 1.0  # Index Beta (S&P 500)
    S = 100    # Fixed Spot Price
    
    # --- Scenarios: Testing various strikes (Straddles and Strangles) ---
    scenarios = [
        {"Name": "ATM Straddle (Baseline)",   "KC": 100, "KP": 100},
        {"Name": "OTM Strangle (Typical)",    "KC": 120, "KP": 80},
        {"Name": "ITM Strangle",              "KC": 90, "KP": 110},
        {"Name": "Wide Asymmetric Strangle",  "KC": 120, "KP": 90},
    ]

    # --- Output Header ---
    print(f"--- Proof: Beta-Neutrality -> Delta-Neutrality for STRANGLES ---")
    print(f"Spot (S): {S}, Rate (r): {r}, Vol (σ): {sigma}, Time (T): {T}, Beta_S: {BETA_S}")
    print(f"{'Scenario':<30} | {'KC/KP':<10} | {'Call Beta':<10} | {'Put Beta':<10} | {'Weight (θ)':<10} | {'Port Delta':<10}")
    print("-" * 110)

    for scen in scenarios:
        KC = scen["KC"]
        KP = scen["KP"]

        # 1. Calculate BSM Prices and Deltas for Call (KC) and Put (KP)
        C, Delta_C = black_scholes_greeks_single(S, KC, T, r, sigma, 'call')
        P, Delta_P = black_scholes_greeks_single(S, KP, T, r, sigma, 'put')
        
        # 2. Calculate Option Betas (BSM-CAPM relationship)
        # Beta = Delta * (S / Option_Price) * Beta_Stock
        beta_c = Delta_C * (S / C) * BETA_S
        beta_p = Delta_P * (S / P) * BETA_S

        # 3. Calculate Beta-Neutral Weight (theta)
        # We solve: theta * beta_c + (1 - theta) * beta_p = 0
        if (beta_c - beta_p) == 0:
            theta = 0
        else:
            theta = -beta_p / (beta_c - beta_p)

        # 4. Calculate Final Portfolio Delta
        # Normalized to a $1 portfolio: n_c = theta / C, n_p = (1 - theta) / P
        n_c = theta / C
        n_p = (1 - theta) / P
        
        port_delta = (n_c * Delta_C) + (n_p * Delta_P)

        # Print Results
        print(f"{scen['Name']:<30} | {KC}/{KP:<6.0f} | {beta_c:<10.2f} | {beta_p:<10.2f} | {theta:<10.4f} | {port_delta:<10.10f}")

    print("-" * 110)
    
if __name__ == "__main__":
    prove_beta_neutrality_implies_delta_neutrality_strangle()