import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Black–Scholes Parameters
# ------------------------
S_max = 200
K     = 100
T     = 1.0
r     = 0.05
sigma = 0.2

# Grid size
M = 200        # price steps
N = 2000       # time steps
dS = S_max / M
dt = T / N

# Stock price grid + payoff
S = np.linspace(0, S_max, M+1)
V = np.maximum(S - K, 0)  # call payoff

# Coefficients for Crank–Nicolson
i = np.arange(1, M)     # interior nodes only

alpha = 0.25 * dt * (sigma**2 * (i**2) - r*i)
beta  = -0.5 * dt * (sigma**2 * (i**2) + r)
gamma = 0.25 * dt * (sigma**2 * (i**2) + r*i)

# Time stepping backwards
for n in range(N):

    # RHS vector (all sizes = M-1)
    RHS = alpha * V[i-1] + (1 + beta)*V[i] + gamma * V[i+1]

    # Boundary conditions
    RHS[0]  += alpha[0] * 0                         # S=0
    RHS[-1] += gamma[-1] * (S_max - K*np.exp(-r*n*dt))  # S→∞

    # Tridiagonal coefficients
    A = -alpha
    B = 1 - beta
    C = -gamma

    # Forward elimination
    for j in range(1, M-1):
        w = A[j] / B[j-1]
        B[j] -= w*C[j-1]
        RHS[j] -= w*RHS[j-1]

    # Back substitution
    V_new = V.copy()
    V_new[M]   = S_max - K*np.exp(-r*n*dt)
    V_new[0]   = 0

    V_new[M-1] = RHS[-1] / B[-1]
    for j in range(M-3, -1, -1):
        V_new[j+1] = (RHS[j] - C[j]*V_new[j+2]) / B[j]

    V = V_new.copy()

# ----------- Result -----------
plt.plot(S, V, label="PDE Price")
plt.axvline(100, linestyle='--', label="S0")
plt.title("European Call Pricing via Black–Scholes PDE (Crank–Nicolson)")
plt.xlabel("Stock Price")
plt.ylabel("Option Value")
plt.legend()
plt.grid(True)
plt.show()

S0 = 100
print(f"\nCall Price using PDE = {np.interp(S0,S,V):.4f}")

