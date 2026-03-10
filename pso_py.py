import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Rastrigin Function
# -----------------------------
def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# -----------------------------
# Swarm Initialization
# -----------------------------
def init_swarm(n_particles, n_dims, bounds):
    low, high = bounds
    positions = np.random.uniform(low, high, (n_particles, n_dims))
    velocities = np.random.uniform(-(high - low), (high - low), (n_particles, n_dims))
    return positions, velocities


# -----------------------------
# Parameters
# -----------------------------
n_particles = 100
n_dims = 2
bounds = (-5.12, 5.12)
w, c1, c2 = 0.9, 1.8, 1.2
iterations = 300

Vmax = abs(bounds[1] - bounds[0])

positions, velocities = init_swarm(n_particles, n_dims, bounds)

pbest_positions = positions.copy()
pbest_scores = np.array([rastrigin(p) for p in positions])

gbest_position = pbest_positions[np.argmin(pbest_scores)].copy()
gbest_score = np.min(pbest_scores)

history = []
gbest_history = []

# -----------------------------
# Optimization Loop
# -----------------------------
for _ in range(iterations):
    for i in range(n_particles):

        r1, r2 = np.random.rand(n_dims), np.random.rand(n_dims)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        velocities[i] = np.clip(velocities[i], -Vmax, Vmax)

        positions[i] = positions[i] + velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        score = rastrigin(positions[i])

        if score < pbest_scores[i]:
            pbest_positions[i] = positions[i].copy()
            pbest_scores[i] = score

        if score < gbest_score:
            gbest_position = positions[i].copy()
            gbest_score = score

    history.append(positions.copy())
    gbest_history.append(gbest_position.copy())

print("Final Best Score:", gbest_score)

# -----------------------------
# Visualization
# -----------------------------
x_range = np.linspace(bounds[0], bounds[1], 200)
y_range = np.linspace(bounds[0], bounds[1], 200)
X, Y = np.meshgrid(x_range, y_range)

Z = 10 * 2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))

fig, ax = plt.subplots()
ax.contourf(X, Y, Z, levels=50, cmap='viridis')

scatter = ax.scatter([], [], c='red', s=25)
gbest_dot = ax.scatter([], [], c='yellow', s=120, marker='*')

ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_title("PSO on Rastrigin Function")

def animate(frame):
    scatter.set_offsets(history[frame])
    gbest_dot.set_offsets([gbest_history[frame]])
    return scatter, gbest_dot

ani = animation.FuncAnimation(fig, animate, frames=iterations, interval=100)
plt.show()