import numpy as np
from scipy.stats import norm
import pandas as pd

def black_scholes_greeks(S, K, T, r, sigma):
    """Calculates Call/Put Price and Delta using the BSM model."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Call Price and Delta
    C = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    Delta_C = norm.cdf(d1)
    
    # Put Price and Delta
    P = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    Delta_P = norm.cdf(d1) - 1
    
    return C, Delta_C, P, Delta_P

def prove_beta_neutrality_implies_delta_neutrality_dynamic():
    # --- Market Constants (BSM Inputs) ---
    r = 0.05      # Risk-free rate (5%)
    sigma = 0.8  # Volatility (20%)
    T = 0.25      # Time-to-maturity (3 months or 0.25 years)
    BETA_S = 1.0  # Index Beta (S&P 500)

    # --- Straddle Scenarios (Dynamic Inputs) ---
    # We test with varying moneyness (S vs. K)
    scenarios = [
        {"Name": "ATM Straddle (S=K)",        "S": 100, "K": 100},
        {"Name": "ITM Call Straddle (S>K)",   "S": 120, "K": 100},
        {"Name": "OTM Call Straddle (S<K)",   "S": 80, "K": 100},
        {"Name": "High Leverage/Short T",     "S": 100, "K": 100, "T": 0.05}, # 2.5 Weeks
        {"Name": "Low Leverage/Long T",       "S": 100, "K": 100, "T": 1.00}, # 1 Year
    ]

    results = []

    print(f"--- Proof: Beta-Neutrality -> Delta-Neutrality (BSM-Consistent Parameters) ---")
    print(f"{'Scenario':<25} | {'Call Beta':<10} | {'Put Beta':<10} | {'Weight (θ)':<10} | {'Port Delta':<10}")
    print("-" * 88)

    for scen in scenarios:
        S = scen["S"]
        K = scen["K"]
        # Use scenario-specific T if provided, otherwise default T
        T_scen = scen.get("T", T)

        # 1. Calculate BSM Prices and Deltas
        C, Delta_C, P, Delta_P = black_scholes_greeks(S, K, T_scen, r, sigma)
        
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
        # Normalize to a $1 portfolio: theta dollars in Calls, (1-theta) in Puts
        n_c = theta / C
        n_p = (1 - theta) / P
        
        port_delta = (n_c * Delta_C) + (n_p * Delta_P)

        # Store and Print
        results.append({
            "Scenario": scen["Name"],
            "Call Beta": beta_c,
            "Put Beta": beta_p,
            "Call Weight (θ)": theta,
            "Portfolio Delta": port_delta
        })

        print(f"{scen['Name']:<25} | {beta_c:<10.2f} | {beta_p:<10.2f} | {theta:<10.4f} | {port_delta:<10.10f}")

    print("-" * 88)

if __name__ == "__main__":
    prove_beta_neutrality_implies_delta_neutrality_dynamic()