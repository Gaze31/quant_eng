import numpy as np

def asian_option_mc(S0, K, r, sigma, T, steps=252, sims=100000,
                    option_type="call", avg_type="arithmetic"):
    """
    Monte Carlo pricing of Asian Options (Arithmetic or Geometric average)

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual)
    T : float
        Time to maturity (years)
    steps : int
        Number of time steps in averaging
    sims : int
        Number of Monte Carlo simulations
    option_type : str
        "call" or "put"
    avg_type : str
        "arithmetic" or "geometric" average

    Returns
    -------
    price : float
        Option price
    """

    dt = T / steps
    # Simulate paths: GBM
    Z = np.random.standard_normal((sims, steps))
    S_paths = np.zeros_like(Z)
    S_paths[:,0] = S0

    for t in range(1, steps):
        S_paths[:,t] = S_paths[:,t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t])

    # Compute average
    if avg_type.lower() == "arithmetic":
        S_avg = np.mean(S_paths, axis=1)
    elif avg_type.lower() == "geometric":
        S_avg = np.exp(np.mean(np.log(S_paths), axis=1))
    else:
        raise ValueError("avg_type must be 'arithmetic' or 'geometric'")

    # Payoff
    if option_type.lower() == "call":
        payoff = np.maximum(S_avg - K, 0)
    else:
        payoff = np.maximum(K - S_avg, 0)

    # Discount
    price = np.exp(-r * T) * np.mean(payoff)
    return price


# ================= Example Usage =================
if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    steps = 252
    sims = 100000

    price_call = asian_option_mc(S0, K, r, sigma, T, steps, sims, option_type="call", avg_type="arithmetic")
    price_put  = asian_option_mc(S0, K, r, sigma, T, steps, sims, option_type="put", avg_type="arithmetic")

    print(f"Asian Call (arithmetic avg) price: {price_call:.4f}")
    print(f"Asian Put  (arithmetic avg) price: {price_put:.4f}")
