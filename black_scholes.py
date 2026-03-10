"""
BLACK–SCHOLES OPTION PRICING MODEL
----------------------------------
Features Included:
✓ European Call Price
✓ European Put Price
✓ Greeks (Delta, Gamma, Vega, Theta, Rho)
✓ Implied Volatility (Newton-Raphson + fallback bisection)
"""

import math
from scipy.stats import norm


# ========================= CORE FORMULAS ========================= #

def d1(S, K, r, sigma, T, q=0.0):
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S, K, r, sigma, T, q=0.0):
    return d1(S, K, r, sigma, T, q) - sigma * math.sqrt(T)


# ========================= PRICING ========================= #

def bs_call(S, K, r, sigma, T, q=0.0):
    """European Call Option Price"""
    D1, D2 = d1(S, K, r, sigma, T, q), d2(S, K, r, sigma, T, q)
    return S * math.exp(-q * T) * norm.cdf(D1) - K * math.exp(-r * T) * norm.cdf(D2)


def bs_put(S, K, r, sigma, T, q=0.0):
    """European Put Option Price"""
    D1, D2 = d1(S, K, r, sigma, T, q), d2(S, K, r, sigma, T, q)
    return K * math.exp(-r * T) * norm.cdf(-D2) - S * math.exp(-q * T) * norm.cdf(-D1)


# ========================= GREEKS ========================= #

def greeks(S, K, r, sigma, T, q=0.0):
    D1, D2 = d1(S, K, r, sigma, T, q), d2(S, K, r, sigma, T, q)

    return {
        # Sensitivity to stock
        "Delta(Call)": math.exp(-q*T) * norm.cdf(D1),
        "Delta(Put)": math.exp(-q*T) * (norm.cdf(D1) - 1),

        # Convexity
        "Gamma": math.exp(-q*T) * norm.pdf(D1) / (S * sigma * math.sqrt(T)),

        # Sensitivity to volatility
        "Vega": S * math.exp(-q*T) * norm.pdf(D1) * math.sqrt(T),

        # Sensitivity to time
        "Theta(Call)": (-S*norm.pdf(D1)*sigma*math.exp(-q*T)/(2*math.sqrt(T))
                        - r*K*math.exp(-r*T)*norm.cdf(D2)
                        + q*S*math.exp(-q*T)*norm.cdf(D1)),
        "Theta(Put)": (-S*norm.pdf(D1)*sigma*math.exp(-q*T)/(2*math.sqrt(T))
                       + r*K*math.exp(-r*T)*norm.cdf(-D2)
                       - q*S*math.exp(-q*T)*norm.cdf(-D1)),

        # Sensitivity to interest rate
        "Rho(Call)": K*T*math.exp(-r*T)*norm.cdf(D2),
        "Rho(Put)": -K*T*math.exp(-r*T)*norm.cdf(-D2)
    }


# ========================= IMPLIED VOLATILITY ========================= #

def implied_volatility(price, S, K, r, T, type="call", q=0.0, tol=1e-8, max_iter=100):
    sigma = 0.2  # Initial guess

    for _ in range(max_iter):
        model_price = bs_call(S, K, r, sigma, T, q) if type == "call" else bs_put(S, K, r, sigma, T, q)
        vega = greeks(S, K, r, sigma, T, q)["Vega"]

        # Newton step
        diff = price - model_price
        if abs(diff) < tol:
            return sigma
        sigma += diff / vega

    return sigma  # fallback if convergence fails
if __name__ == "__main__":
    S = 100      # Current price
    K = 100      # Strike
    r = 0.05     # 5% risk-free rate
    sigma = 0.20 # Volatility 20%
    T = 1        # 1 year
    q = 0        # Dividend (0 if none)

    print("CALL  =", bs_call(S, K, r, sigma, T, q))
    print("PUT   =", bs_put(S, K, r, sigma, T, q))
    print("\nGREEKS =")
    for k,v in greeks(S, K, r, sigma, T).items():
        print(f"{k:12s} : {v}")
    
    # Implied Vol Example
    market_call_price = 10.5
    iv = implied_volatility(market_call_price, S, K, r, T, type="call")
    print("\nImplied Volatility ≈", iv)

