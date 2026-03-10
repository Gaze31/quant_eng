"""
Black-Scholes Greeks for European Options
-----------------------------------------
Computes:
  ✔ Delta
  ✔ Gamma
  ✔ Vega
  ✔ Theta
  ✔ Rho
"""

import math
from scipy.stats import norm


# ================= Black-Scholes d1 & d2 ================= #

def d1(S, K, r, sigma, T, q=0):
    return (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def d2(S, K, r, sigma, T, q=0):
    return d1(S, K, r, sigma, T, q) - sigma*math.sqrt(T)


# ================= Greeks ================= #

def greeks_black_scholes(S, K, r, sigma, T, q=0):
    D1, D2 = d1(S, K, r, sigma, T, q), d2(S, K, r, sigma, T, q)

    # Core Greeks
    delta_call = math.exp(-q*T) * norm.cdf(D1)
    delta_put  = math.exp(-q*T) * (norm.cdf(D1) - 1)

    gamma = math.exp(-q*T) * norm.pdf(D1) / (S * sigma * math.sqrt(T))

    vega  = S * math.exp(-q*T) * norm.pdf(D1) * math.sqrt(T)

    theta_call = (-S*math.exp(-q*T)*norm.pdf(D1)*sigma/(2*math.sqrt(T))
                  - r*K*math.exp(-r*T)*norm.cdf(D2)
                  + q*S*math.exp(-q*T)*norm.cdf(D1))

    theta_put = (-S*math.exp(-q*T)*norm.pdf(D1)*sigma/(2*math.sqrt(T))
                 + r*K*math.exp(-r*T)*norm.cdf(-D2)
                 - q*S*math.exp(-q*T)*norm.cdf(-D1))

    rho_call = K*T*math.exp(-r*T)*norm.cdf(D2)
    rho_put  = -K*T*math.exp(-r*T)*norm.cdf(-D2)

    return {
        "Delta (Call)": delta_call,  "Delta (Put)": delta_put,
        "Gamma": gamma,
        "Vega": vega,
        "Theta (Call)": theta_call, "Theta (Put)": theta_put,
        "Rho (Call)": rho_call,     "Rho (Put)": rho_put
    }
# Sample parameters
S = 100     # spot price
K = 100     # strike
r = 0.05    # interest rate
sigma = 0.20 # volatility
T = 1       # 1 year
q = 0       # dividend yield

greeks = greeks_black_scholes(S, K, r, sigma, T, q)

print("\n---- Greeks Output ----")
for g, val in greeks.items():
    print(f"{g:15s}: {val:.6f}")
