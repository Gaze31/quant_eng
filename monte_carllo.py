import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# MONTE CARLO STRESS TESTING — Robust Quant Engine
# ============================================================

np.random.seed(42)

# ============================================================
# 1. PORTFOLIO SETUP
# ============================================================

assets = ['SPY', 'TLT', 'GLD', 'OIL', 'HYG']
weights = np.array([0.35, 0.25, 0.15, 0.15, 0.10])
portfolio_value = 10_000_000  # $10M

# Annual parameters
annual_returns = np.array([0.10, 0.03, 0.05, 0.08, 0.06])
annual_vols    = np.array([0.18, 0.12, 0.15, 0.35, 0.14])

daily_returns = annual_returns / 252
daily_vols    = annual_vols / np.sqrt(252)

# ============================================================
# 2. CORRELATION MATRICES
# ============================================================

corr_normal = np.array([
    [1.00, -0.30,  0.05,  0.20,  0.60],
    [-0.30, 1.00,  0.20, -0.15,  0.10],
    [0.05,  0.20,  1.00,  0.10, -0.05],
    [0.20, -0.15,  0.10,  1.00,  0.15],
    [0.60,  0.10, -0.05,  0.15,  1.00]
])

corr_stressed = np.array([
    [1.00, -0.60,  0.30,  0.55,  0.90],
    [-0.60, 1.00,  0.40, -0.40,  0.20],
    [0.30,  0.40,  1.00,  0.25,  0.10],
    [0.55, -0.40,  0.25,  1.00,  0.50],
    [0.90,  0.20,  0.10,  0.50,  1.00]
])

# ============================================================
# 3. MATRIX UTILITIES (ROBUST)
# ============================================================

def build_cov(corr, vols):
    D = np.diag(vols)
    return D @ corr @ D

def make_psd(matrix, epsilon=1e-8):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

def safe_cholesky(matrix):
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        print("⚠ Covariance not PSD. Auto-correcting...")
        matrix_psd = make_psd(matrix)
        return np.linalg.cholesky(matrix_psd)

def diagnostics(matrix, name):
    eigvals = np.linalg.eigvalsh(matrix)
    print(f"\n{name} Covariance Diagnostics")
    print("-" * 40)
    print("Min eigenvalue :", eigvals.min())
    print("Condition num  :", np.linalg.cond(matrix))
    print("PSD valid      :", np.all(eigvals > -1e-10))

# ============================================================
# 4. BUILD COVARIANCE MATRICES
# ============================================================

cov_normal   = build_cov(corr_normal,  daily_vols)
cov_stressed = build_cov(corr_stressed, daily_vols * 2.5)

diagnostics(cov_normal, "Normal")
diagnostics(cov_stressed, "Stressed")

# ============================================================
# 5. MONTE CARLO ENGINE
# ============================================================

def run_monte_carlo(cov_matrix, daily_mu, weights, port_value,
                    n_simulations=10_000, n_days=252, shock=None):

    n_assets = len(weights)
    L = safe_cholesky(cov_matrix)

    final_pnl = []
    all_paths = []

    for _ in range(n_simulations):

        Z = np.random.standard_normal((n_days, n_assets))
        correlated_returns = Z @ L.T + daily_mu

        if shock:
            for asset_idx, shock_return in shock.items():
                correlated_returns[0, asset_idx] = shock_return

        port_returns = correlated_returns @ weights
        cumulative = port_value * np.cumprod(1 + port_returns)

        all_paths.append(cumulative)
        final_pnl.append(cumulative[-1] - port_value)

    return np.array(all_paths), np.array(final_pnl)

# ============================================================
# 6. RUN SCENARIOS
# ============================================================

print("\nRunning Normal Market Monte Carlo...")
paths_normal, pnl_normal = run_monte_carlo(
    cov_normal, daily_returns, weights, portfolio_value
)

print("Running Stressed Market Monte Carlo...")
paths_stressed, pnl_stressed = run_monte_carlo(
    cov_stressed, daily_returns * 0.3, weights, portfolio_value
)

scenario_shock = {0: -0.15, 4: -0.10, 3: -0.20}

print("Running 2008-Style Shock Scenario...")
paths_scenario, pnl_scenario = run_monte_carlo(
    cov_stressed, daily_returns * 0.3,
    weights, portfolio_value, shock=scenario_shock
)

# ============================================================
# 7. RISK METRICS
# ============================================================

def risk_metrics(pnl, label):
    var_99 = np.percentile(pnl, 1)
    cvar_99 = pnl[pnl <= var_99].mean()

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"Mean P&L      : ${np.mean(pnl):,.0f}")
    print(f"Std Dev       : ${np.std(pnl):,.0f}")
    print(f"VaR (99%)     : ${var_99:,.0f}")
    print(f"CVaR (99%)    : ${cvar_99:,.0f}")
    print(f"Worst Case    : ${np.min(pnl):,.0f}")
    print(f"P(Loss >10%)  : {(pnl < -portfolio_value*0.10).mean()*100:.2f}%")
    print(f"P(Loss >20%)  : {(pnl < -portfolio_value*0.20).mean()*100:.2f}%")

risk_metrics(pnl_normal, "NORMAL MARKET")
risk_metrics(pnl_stressed, "STRESSED MARKET")
risk_metrics(pnl_scenario, "2008 SHOCK")

# ============================================================
# 8. VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
for i in np.random.choice(len(paths_normal), 300, replace=False):
    ax1.plot(paths_normal[i], alpha=0.03)
ax1.set_title("Normal Market Paths")

ax2 = fig.add_subplot(gs[0, 1])
for i in np.random.choice(len(paths_stressed), 300, replace=False):
    ax2.plot(paths_stressed[i], alpha=0.03)
ax2.set_title("Stressed Market Paths")

ax3 = fig.add_subplot(gs[1, :])
ax3.hist(pnl_normal, bins=150, alpha=0.4, label="Normal", density=True)
ax3.hist(pnl_stressed, bins=150, alpha=0.4, label="Stressed", density=True)
ax3.hist(pnl_scenario, bins=150, alpha=0.4, label="2008 Shock", density=True)
ax3.legend()
ax3.set_title("P&L Distribution Comparison")

plt.tight_layout()

# ============================================================
# 9. SAFE SAVE
# ============================================================

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, "mc_stress_test.png")
plt.savefig(file_path, dpi=150)
plt.show()

print(f"\nPlot saved to: {file_path}")