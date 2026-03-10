import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Brownian Motion (Wiener Process)
# ==============================

def simulate_brownian_motion(T=1, N=1000, paths=5):
    """
    Simulates Standard Brownian Motion (W_t)
    dW = sqrt(dt) * N(0,1)

    T = total time horizon (years)
    N = steps
    paths = number of simulated trajectories
    """

    dt = T / N                      # step size
    t = np.linspace(0, T, N+1)      # timeline

    W = np.zeros((paths, N+1))      # to store all paths

    for i in range(paths):
        dW = np.sqrt(dt) * np.random.randn(N)  # increments
        W[i,1:] = np.cumsum(dW)                # build BM

    return t, W


# ==============================
# Run & Plot
# ==============================

if __name__ == "__main__":
    t, W = simulate_brownian_motion(T=1, N=1000, paths=10)

    plt.figure(figsize=(8,5))
    for i in range(W.shape[0]):
        plt.plot(t, W[i])

    plt.title("Standard Brownian Motion Simulation")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()
