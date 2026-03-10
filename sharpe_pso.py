import numpy as np

# -----------------------
# Data
# -----------------------
np.random.seed(42)

n_assets = 5
n_days = 252

returns = np.random.randn(n_days, n_assets) * 0.01 + 0.0003
rf = 0.02 / 252


# -----------------------
# Sharpe Objective
# -----------------------
def sharpe_objective(w, returns, rf):
    portfolio_returns = returns @ w
    excess_returns = portfolio_returns - rf

    rp = np.mean(excess_returns)
    sigma = np.std(portfolio_returns)

    if sigma == 0:
        return 1e6

    sharpe = rp / sigma * np.sqrt(252)
    return -sharpe


# -----------------------
# PSO Parameters
# -----------------------
n_particles = 30
n_iterations = 200
n_dims = 5

w_inertia = 0.7
c1 = 1.5
c2 = 1.5

positions = np.random.rand(n_particles, n_dims)
velocities = np.zeros((n_particles, n_dims))

pbest_positions = positions.copy()
pbest_scores = np.array([sharpe_objective(p, returns, rf) for p in positions])

gbest_idx = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_idx].copy()
gbest_score = pbest_scores[gbest_idx]


# -----------------------
# PSO Loop
# -----------------------
for _ in range(n_iterations):
    for i in range(n_particles):
        r1 = np.random.rand(n_dims)
        r2 = np.random.rand(n_dims)

        velocities[i] = (
            w_inertia * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        positions[i] = positions[i] + velocities[i]

        # --- Projection step ---
        positions[i] = np.clip(positions[i], 0, 1)
        if np.sum(positions[i]) == 0:
            positions[i] = np.ones(n_dims) / n_dims
        else:
            positions[i] = positions[i] / np.sum(positions[i])

        score = sharpe_objective(positions[i], returns, rf)

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i].copy()

    gbest_idx = np.argmin(pbest_scores)
    if pbest_scores[gbest_idx] < gbest_score:
        gbest_score = pbest_scores[gbest_idx]
        gbest_position = pbest_positions[gbest_idx].copy()


# Final result
best_sharpe = -gbest_score
best_weights = gbest_position

print("Best Sharpe Ratio:", best_sharpe)
print("Optimal Weights:", best_weights)