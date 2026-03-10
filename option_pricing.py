import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fetch historical data
ticker = "AAPL"
data = yf.download(ticker, period="1y")
data['Returns'] = data['Close'].pct_change()

sigma = data['Returns'].std() * np.sqrt(252)   # Annualized volatility
S0 = float(data['Close'].iloc[-1])
  # last closing price
                      # Current price

print("Spot Price:", S0)
print("Annual Volatility:", sigma)

def black_scholes(S, K, r, sigma, T, type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if type == "call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return price

def greeks(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    Delta = norm.cdf(d1)
    Gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    Vega  = S*norm.pdf(d1)*np.sqrt(T)
    Theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    Rho   = K*T*np.exp(-r*T)*norm.cdf(d2)

    return Delta, Gamma, Vega, Theta, Rho

# Example
price = black_scholes(S0, 150, 0.04, sigma, 0.5)
d,g,v,t,rh = greeks(S0,150,0.04,sigma,0.5)

print("BS Price:",price)
print("Greeks:",d,g,v,t,rh)

def monte_carlo(S0,K,r,sigma,T,n=50000,antithetic=True):
    Z = np.random.normal(size=n)
    if antithetic:
        Z = np.concatenate([Z, -Z])  # Variance reduction

    ST = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST-K, 0)
    return np.exp(-r*T)*np.mean(payoff)

price_mc = monte_carlo(S0,150,0.04,sigma,0.5)
print("Monte-Carlo Price:",price_mc)

def binomial_tree(S0,K,r,sigma,T,steps=200):
    dt = T/steps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)

    prices = [S0*u**i*d**(steps-i) for i in range(steps+1)]
    values = [max(p-K,0) for p in prices]

    for _ in range(steps):
        values = [np.exp(-r*dt)*(p*values[i+1] + (1-p)*values[i]) for i in range(len(values)-1)]

    return values[0]

print("Binomial Price:",binomial_tree(S0,150,0.04,sigma,0.5))
K_range = np.arange(80,180,2)
prices = [black_scholes(S0,K,0.04,sigma,0.5) for K in K_range]

plt.plot(K_range, prices)
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.title("Option Price vs Strike")
plt.grid()
plt.show()
