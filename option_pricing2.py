# ===============================================
# Options Pricing Mini-Project
# ===============================================

# 1️⃣ Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------
# Black-Scholes Formula
# -----------------------
def black_scholes(S, K, r, sigma, T, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type.lower()=="call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

# -----------------------
# Greeks Calculation
# -----------------------
def greeks(S,K,r,sigma,T,option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Delta = norm.cdf(d1) if option_type=="call" else norm.cdf(d1)-1
    Gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    Vega = S*norm.pdf(d1)*np.sqrt(T)
    Theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*(norm.cdf(d2) if option_type=="call" else norm.cdf(-d2))
    Rho = K*T*np.exp(-r*T)*(norm.cdf(d2) if option_type=="call" else -norm.cdf(-d2))
    return Delta, Gamma, Vega, Theta, Rho

# -----------------------
# Binomial Tree (American Options)
# -----------------------
def binomial_tree(S0, K, r, sigma, T, steps=200, option_type="call", american=True):
    dt = T/steps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    ST = np.zeros((steps+1, steps+1))
    ST[0,0] = S0
    for i in range(1, steps+1):
        ST[i,0] = ST[i-1,0]*u
        for j in range(1,i+1):
            ST[i,j] = ST[i-1,j-1]*d
    V = np.maximum(ST[-1,:]-K,0) if option_type=="call" else np.maximum(K-ST[-1,:],0)
    for i in range(steps-1,-1,-1):
        for j in range(i+1):
            V[j] = np.exp(-r*dt)*(p*V[j] + (1-p)*V[j+1])
            if american:
                if option_type=="call":
                    V[j] = max(V[j], ST[i,j]-K)
                else:
                    V[j] = max(V[j], K-ST[i,j])
    return V[0]

# -----------------------
# Asian Option Monte Carlo
# -----------------------
def asian_option_mc(S0,K,r,sigma,T,steps=252,sims=100000,option_type="call"):
    dt = T/steps
    Z = np.random.standard_normal((sims,steps))
    S = np.zeros_like(Z)
    S[:,0] = S0
    for t in range(1,steps):
        S[:,t] = S[:,t-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t])
    S_avg = np.mean(S, axis=1)
    payoff = np.maximum(S_avg-K,0) if option_type=="call" else np.maximum(K-S_avg,0)
    return np.exp(-r*T)*np.mean(payoff)

# -----------------------
# Monte Carlo European Option
# -----------------------
def monte_carlo_european(S0,K,r,sigma,T,sims=100000,option_type="call"):
    Z = np.random.standard_normal(sims)
    ST = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
    return np.exp(-r*T)*np.mean(payoff)

# ===============================================
# 2️⃣ Example Usage
# ===============================================
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0

# European Call (Black-Scholes)
bs_call = black_scholes(S0,K,r,sigma,T,"call")
bs_put  = black_scholes(S0,K,r,sigma,T,"put")
Delta,Gamma,Vega,Theta,Rho = greeks(S0,K,r,sigma,T,"call")

# American Put (Binomial)
am_put = binomial_tree(S0,K,r,sigma,T,steps=200,option_type="put",american=True)

# Asian Call
asian_call = asian_option_mc(S0,K,r,sigma,T,steps=252,sims=100000,option_type="call")

# Monte Carlo European
mc_call = monte_carlo_european(S0,K,r,sigma,T,sims=100000,option_type="call")

# ===============================================
# 3️⃣ Results Output
# ===============================================
print(f"Black-Scholes Call: {bs_call:.4f}")
print(f"Black-Scholes Put : {bs_put:.4f}")
print(f"Greeks (Delta,Gamma,Vega,Theta,Rho): {Delta:.4f},{Gamma:.4f},{Vega:.4f},{Theta:.4f},{Rho:.4f}")
print(f"American Put (Binomial Tree): {am_put:.4f}")
print(f"Asian Call (Monte Carlo): {asian_call:.4f}")
print(f"European Call (Monte Carlo): {mc_call:.4f}")

# ===============================================
# 4️⃣ Plot Example: Option Price vs Strike
# ===============================================
K_range = np.arange(80,120,2)
prices_bs = [black_scholes(S0,K,r,sigma,T,"call") for K in K_range]

import matplotlib.pyplot as plt
plt.plot(K_range,prices_bs,label="BS Call Price")
plt.xlabel("Strike Price")
plt.ylabel("Call Option Price")
plt.title("Black-Scholes Call Price vs Strike")
plt.grid(True)
plt.legend()
plt.show()
