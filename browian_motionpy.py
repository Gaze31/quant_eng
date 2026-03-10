import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=1.0, N=1000, paths=5):
    """
    Simulate Standard Brownian Motion (W_t)

    Parameters:
        T     = Total time (years)
        N     = Number of time steps
        paths = Number of Brownian trajectories to simulate
    """

    dt = T / N                     # step size
    t = np.linspace(0, T, N+1)     # time axis

    # Storage for paths
    W = np.zeros((paths, N+1))

    for i in range(paths):
        dW = np.sqrt(dt) * np.random.randn(N)   # increments
        W[i,1:] = np.cumsum(dW)                 # cumulative sum => BM

    return t, W


# ====================== RUN & PLOT ========================

if __name__ == "__main__":
    t, W = brownian_motion(T=1, N=1000, paths=10)

    plt.figure(figsize=(8,5))
    for i in range(W.shape[0]):
        plt.plot(t, W[i], linewidth=1)

    plt.title("Brownian Motion (Wiener Process) Simulation")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()
