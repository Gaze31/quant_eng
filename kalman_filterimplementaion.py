import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x0):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self, u=None):
        # 🔥 Fix implemented here
        if u is None:
            self.x = self.A @ self.x
        else:
            self.x = self.A @ self.x + self.B @ u
            
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x


# ============ Simulated Data =================
np.random.seed(42)
steps = 60
true = np.linspace(0, 30, steps)
measure = true + np.random.normal(0, 2, steps)

# ============ Filter Setup ===================
A = np.array([[1]])
H = np.array([[1]])
B = np.array([[0]])

Q = np.array([[1e-3]])
R = np.array([[4]])
P = np.array([[1]])

kf = KalmanFilter(A, B, H, Q, R, P, x0=np.array([[0]]))

# ============ Run Filter =====================
filtered = []
for z in measure:
    kf.predict()
    filtered.append(kf.update(np.array([[z]]))[0])

# ============ Plot ===========================
plt.plot(true, label="True Signal", linewidth=2)
plt.scatter(range(steps), measure, label="Noisy Measure", alpha=0.5)
plt.plot(filtered, label="Kalman Estimate", linewidth=2, color="green")
plt.legend(); plt.grid(); plt.show()
plt.grid(True)
plt.show()
