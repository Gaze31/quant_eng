import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Monte Carlo Option Pricing
# ===============================

def monte_carlo_option_price(S0, K, r, sigma, T, sims=50000, option_type="call", plot_paths=False):
    """
    Monte Carlo pricer for European Call & Put options
    using Geometric Brownian Motion (GBM).
    """

    # Step 1: Simulate stock price at maturity
    Z = np.random.standard_normal(sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Step 2: Payoff calculation
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # Step 3: Discount to present value
    price = np.exp(-r * T) * np.mean(payoff)

    # Optional visualization
    if plot_paths:
        plt.figure(figsize=(7,4))
        for _ in range(10):
            path_Z = np.random.standard_normal(252)
            path = [S0]
            for i in range(252):
                path.append(path[-1] * np.exp((r - 0.5*sigma**2)*(T/252) + sigma*np.sqrt(T/252)*path_Z[i]))
            plt.plot(path, alpha=0.7)
        plt.title("Sample Simulated Price Paths")
        plt.xlabel("Time Steps (Daily)")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()

    return price


# ===============================
# Run Example
# ===============================

if __name__ == "__main__":
    S0 = 100      # Current stock price
    K = 110       # Strike price
    r = 0.05      # Risk-free interest rate
    sigma = 0.20  # Annual volatility
    T = 1.0       # Time to maturity (years)

    call_price = monte_carlo_option_price(S0, K, r, sigma, T, sims=100000, option_type="call", plot_paths=True)
    put_price  = monte_carlo_option_price(S0, K, r, sigma, T, sims=100000, option_type="put")

    print(f"Monte-Carlo European Call Price: {call_price:.4f}")
    print(f"Monte-Carlo European Put Price : {put_price:.4f}")
