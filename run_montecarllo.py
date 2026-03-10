import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MONTE CARLO STRESS TESTING — Quant Finance
# ============================================================

np.random.seed(42)

# --- 1. Portfolio Setup ---
assets = ['SPY', 'TLT', 'GLD', 'OIL', 'HYG']
weights = np.array([0.35, 0.25, 0.15, 0.15, 0.10])
portfolio_value = 10_000_000  # $10M portfolio

# --- 2. Historical Return Parameters (annualized → daily) ---
annual_returns = np.array([0.10, 0.03, 0.05, 0.08, 0.06])
annual_vols    = np.array([0.18, 0.12, 0.15, 0.35, 0.14])

daily_returns = annual_returns / 252
daily_vols    = annual_vols / np.sqrt(252)

# --- 3. Correlation Matrix (normal market) ---
corr_normal = np.array([
    [1.00, -0.30,  0.05,  0.20,  0.60],
    [-0.30, 1.00,  0.20, -0.15,  0.10],
    [0.05,  0.20,  1.00,  0.10, -0.05],
    [0.20, -0.15,  0.10,  1.00,  0.15],
    [0.60,  0.10, -0.05,  0.15,  1.00]
])

# --- 4. STRESSED Correlation Matrix (crisis — correlations increase) ---
corr_stressed = np.array([
    [1.00,  0.60,  0.50,  0.55,  0.80],  # Made correlations more realistic
    [0.60,  1.00,  0.40,  0.35,  0.50],
    [0.50,  0.40,  1.00,  0.45,  0.30],
    [0.55,  0.35,  0.45,  1.00,  0.50],
    [0.80,  0.50,  0.30,  0.50,  1.00]
])

# --- 5. Build Covariance Matrices ---
def build_cov(corr, vols):
    """Build covariance matrix from correlation and volatilities"""
    D = np.diag(vols)
    return D @ corr @ D

cov_normal = build_cov(corr_normal, daily_vols)

# Stress scenario: higher vols and different correlations
stressed_vols = daily_vols * 2.0  # Volatility doubles in stress
cov_stressed = build_cov(corr_stressed, stressed_vols)

# ============================================================
# MONTE CARLO ENGINE
# ============================================================

def run_monte_carlo(cov_matrix, daily_mu, weights, port_value,
                    n_simulations=10_000, n_days=252, shock=None):
    """
    Simulates portfolio paths using Cholesky decomposition.
    shock: optional dict with asset_idx → forced return shock (scenario overlay)
    """
    n_assets = len(weights)
    
    # Ensure covariance matrix is positive definite
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
    
    try:
        L = np.linalg.cholesky(cov_matrix)  # Cholesky for correlated normals
    except np.linalg.LinAlgError:
        # If not positive definite, use nearest positive definite matrix
        from scipy.linalg import sqrtm
        L = sqrtm(cov_matrix).real
    
    final_pnl = []
    all_paths = []

    for sim in range(n_simulations):
        # Generate correlated random returns
        Z = np.random.standard_normal((n_days, n_assets))
        correlated_returns = Z @ L.T  # shape: (n_days, n_assets)
        
        # Add mean returns
        correlated_returns = correlated_returns + daily_mu

        # Apply scenario shock on day 1 if specified
        if shock:
            for asset_idx, shock_return in shock.items():
                correlated_returns[0, asset_idx] = shock_return

        # Portfolio daily returns
        port_returns = correlated_returns @ weights
        
        # Calculate cumulative portfolio value
        cumulative = port_value * np.cumprod(1 + port_returns)
        
        all_paths.append(cumulative)
        final_pnl.append(cumulative[-1] - port_value)

    return np.array(all_paths), np.array(final_pnl)

# ============================================================
# RUN SCENARIOS
# ============================================================

print("Running Normal Market Monte Carlo...")
paths_normal, pnl_normal = run_monte_carlo(
    cov_normal, daily_returns, weights, portfolio_value
)

print("Running Stressed Market Monte Carlo...")
paths_stressed, pnl_stressed = run_monte_carlo(
    cov_stressed, daily_returns * 0.5,  # Lower expected returns in stress
    weights, portfolio_value
)

# Scenario: 2008-style shock - initial crash then recovery pattern
scenario_shock = {0: -0.15,  # SPY -15% day 1
                  3: -0.20,  # OIL -20% day 1
                  4: -0.10}  # HYG -10% day 1

print("Running 2008-Style Scenario Shock Monte Carlo...")
paths_scenario, pnl_scenario = run_monte_carlo(
    cov_stressed, daily_returns * 0.3,  # Severely reduced returns
    weights, portfolio_value, shock=scenario_shock
)

# ============================================================
# RISK METRICS
# ============================================================

def risk_metrics(pnl, label):
    """Calculate comprehensive risk metrics"""
    confidence = 0.99
    var_99 = np.percentile(pnl, (1-confidence)*100)
    cvar_99 = pnl[pnl <= var_99].mean()
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Mean P&L        : ${np.mean(pnl):>15,.0f}")
    print(f"  Std Dev         : ${np.std(pnl):>15,.0f}")
    print(f"  VaR (99%)       : ${var_99:>15,.0f}")
    print(f"  CVaR (99%)      : ${cvar_99:>15,.0f}")
    print(f"  Worst Case      : ${np.min(pnl):>15,.0f}")
    print(f"  Best Case       : ${np.max(pnl):>15,.0f}")
    print(f"  Sharpe Ratio    : {np.mean(pnl)/np.std(pnl)/np.sqrt(252):>14.3f}")
    print(f"  Max Drawdown    : {(np.min(pnl)/portfolio_value)*100:>13.2f}%")
    print(f"  P(Loss > 10%)   : {(pnl < -portfolio_value*0.10).mean()*100:.2f}%")
    print(f"  P(Loss > 20%)   : {(pnl < -portfolio_value*0.20).mean()*100:.2f}%")
    
    return var_99, cvar_99

print("\n" + "="*50)
print("RISK METRICS SUMMARY")
print("="*50)

var_n, cvar_n = risk_metrics(pnl_normal, "NORMAL MARKET")
var_s, cvar_s = risk_metrics(pnl_stressed, "STRESSED MARKET")
var_sc, cvar_sc = risk_metrics(pnl_scenario, "2008-STYLE SHOCK")

# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

text_color = '#e6edf3'
grid_color = '#21262d'
accent_cols = {'normal': '#3fb950', 'stressed': '#f85149', 'scenario': '#d29922'}

def style_ax(ax, title):
    """Apply consistent styling to axes"""
    ax.set_facecolor('#161b22')
    ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.grid(color=grid_color, linewidth=0.5, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)

# --- Plot 1: Simulated Portfolio Paths (Normal) ---
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Normal Market — 500 Sample Paths')

# Plot a subset of paths for clarity
n_plot_paths = min(500, len(paths_normal))
sample_idx = np.random.choice(len(paths_normal), n_plot_paths, replace=False)

for i in sample_idx:
    ax1.plot(paths_normal[i] / 1e6, alpha=0.03, color=accent_cols['normal'], linewidth=0.5)

# Add median path
median_path = np.median(paths_normal, axis=0)
ax1.plot(median_path / 1e6, color='white', linewidth=2, label='Median', alpha=0.9)

# Add confidence bands
p5 = np.percentile(paths_normal, 5, axis=0)
p95 = np.percentile(paths_normal, 95, axis=0)
ax1.fill_between(range(len(median_path)), p5/1e6, p95/1e6, 
                 color=accent_cols['normal'], alpha=0.15, label='90% CI')

ax1.axhline(portfolio_value / 1e6, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Trading Days', color=text_color, fontsize=10)
ax1.set_ylabel('Portfolio Value ($M)', color=text_color, fontsize=10)
ax1.legend(fontsize=9, labelcolor=text_color, facecolor='#161b22', loc='upper left')

# --- Plot 2: Simulated Portfolio Paths (Stressed) ---
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Stressed Market — 500 Sample Paths')

for i in sample_idx:
    ax2.plot(paths_stressed[i] / 1e6, alpha=0.03, color=accent_cols['stressed'], linewidth=0.5)

median_path_stressed = np.median(paths_stressed, axis=0)
ax2.plot(median_path_stressed / 1e6, color='white', linewidth=2, label='Median', alpha=0.9)

p5_stressed = np.percentile(paths_stressed, 5, axis=0)
p95_stressed = np.percentile(paths_stressed, 95, axis=0)
ax2.fill_between(range(len(median_path_stressed)), p5_stressed/1e6, p95_stressed/1e6, 
                 color=accent_cols['stressed'], alpha=0.15, label='90% CI')

ax2.axhline(portfolio_value / 1e6, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Trading Days', color=text_color, fontsize=10)
ax2.set_ylabel('Portfolio Value ($M)', color=text_color, fontsize=10)
ax2.legend(fontsize=9, labelcolor=text_color, facecolor='#161b22', loc='upper left')

# --- Plot 3: Simulated Portfolio Paths (2008 Scenario) ---
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, '2008-Style Shock — 500 Sample Paths')

for i in sample_idx:
    ax3.plot(paths_scenario[i] / 1e6, alpha=0.03, color=accent_cols['scenario'], linewidth=0.5)

median_path_scenario = np.median(paths_scenario, axis=0)
ax3.plot(median_path_scenario / 1e6, color='white', linewidth=2, label='Median', alpha=0.9)

p5_scenario = np.percentile(paths_scenario, 5, axis=0)
p95_scenario = np.percentile(paths_scenario, 95, axis=0)
ax3.fill_between(range(len(median_path_scenario)), p5_scenario/1e6, p95_scenario/1e6, 
                 color=accent_cols['scenario'], alpha=0.15, label='90% CI')

ax3.axhline(portfolio_value / 1e6, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Trading Days', color=text_color, fontsize=10)
ax3.set_ylabel('Portfolio Value ($M)', color=text_color, fontsize=10)
ax3.legend(fontsize=9, labelcolor=text_color, facecolor='#161b22', loc='upper left')

# --- Plot 4: P&L Distribution Comparison ---
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, 'P&L Distribution — Normal vs Stressed vs 2008 Shock')

bins = 100
for pnl, col, label in [
    (pnl_normal, accent_cols['normal'], 'Normal Market'),
    (pnl_stressed, accent_cols['stressed'], 'Stressed Market'),
    (pnl_scenario, accent_cols['scenario'], '2008 Shock')
]:
    ax4.hist(pnl / 1e6, bins=bins, alpha=0.4, color=col, label=label, density=True, 
             edgecolor='none')

# VaR lines
for var, col, label in [
    (var_n, accent_cols['normal'], 'VaR Normal'),
    (var_s, accent_cols['stressed'], 'VaR Stressed'),
    (var_sc, accent_cols['scenario'], 'VaR 2008')
]:
    ax4.axvline(var / 1e6, color=col, linestyle='--', linewidth=2,
                label=f'{label}: ${abs(var)/1e6:.2f}M', alpha=0.8)

ax4.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
ax4.set_xlabel('Annual P&L ($M)', color=text_color, fontsize=10)
ax4.set_ylabel('Probability Density', color=text_color, fontsize=10)
ax4.legend(fontsize=9, labelcolor=text_color, facecolor='#161b22', loc='upper left')

# --- Plot 5: Risk Metrics Bar Chart ---
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, 'Tail Risk Comparison')

metrics = ['VaR 99%', 'CVaR 99%', 'Worst Case']
vals_normal = [abs(var_n), abs(cvar_n), abs(pnl_normal.min())]
vals_stress = [abs(var_s), abs(cvar_s), abs(pnl_stressed.min())]
vals_scene = [abs(var_sc), abs(cvar_sc), abs(pnl_scenario.min())]

x = np.arange(len(metrics))
width = 0.25

bars1 = ax5.bar(x - width, [v/1e6 for v in vals_normal], width, 
                color=accent_cols['normal'], label='Normal', alpha=0.8, edgecolor='white', linewidth=0.5)
bars2 = ax5.bar(x, [v/1e6 for v in vals_stress], width, 
                color=accent_cols['stressed'], label='Stressed', alpha=0.8, edgecolor='white', linewidth=0.5)
bars3 = ax5.bar(x + width, [v/1e6 for v in vals_scene], width, 
                color=accent_cols['scenario'], label='2008', alpha=0.8, edgecolor='white', linewidth=0.5)

ax5.set_xticks(x)
ax5.set_xticklabels(metrics, color=text_color, fontsize=9)
ax5.set_ylabel('Expected Loss ($M)', color=text_color, fontsize=10)
ax5.legend(fontsize=9, labelcolor=text_color, facecolor='#161b22', loc='upper left')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.1f}M', ha='center', va='bottom', 
                color=text_color, fontsize=8, fontweight='bold')

fig.suptitle('Monte Carlo Stress Testing — $10M Multi-Asset Portfolio\nSPY(35%) TLT(25%) GLD(15%) OIL(15%) HYG(10%)',
             color=text_color, fontsize=14, fontweight='bold', y=0.98)

# Add summary text
summary_text = f"Normal VaR: ${abs(var_n)/1e6:.2f}M | Stressed VaR: ${abs(var_s)/1e6:.2f}M | 2008 VaR: ${abs(var_sc)/1e6:.2f}M"
fig.text(0.5, 0.92, summary_text, ha='center', color=text_color, fontsize=11, 
         bbox=dict(facecolor='#161b22', edgecolor=grid_color, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('mc_stress_test.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n✅ Plot saved as 'mc_stress_test.png'")

# ============================================================
# ADDITIONAL STATISTICAL ANALYSIS
# ============================================================

print("\n" + "="*50)
print("DETAILED STATISTICAL COMPARISON")
print("="*50)

# Create comparison DataFrame
comparison_data = {
    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 
               'Max Drawdown', 'VaR 99%', 'CVaR 99%'],
    'Normal': [
        f"{np.mean(pnl_normal)/portfolio_value*100:.2f}%",
        f"{np.std(pnl_normal)/portfolio_value*100:.2f}%",
        f"{np.mean(pnl_normal)/np.std(pnl_normal)/np.sqrt(252):.3f}",
        f"{abs(pnl_normal.min())/portfolio_value*100:.2f}%",
        f"${abs(var_n)/1e6:.2f}M",
        f"${abs(cvar_n)/1e6:.2f}M"
    ],
    'Stressed': [
        f"{np.mean(pnl_stressed)/portfolio_value*100:.2f}%",
        f"{np.std(pnl_stressed)/portfolio_value*100:.2f}%",
        f"{np.mean(pnl_stressed)/np.std(pnl_stressed)/np.sqrt(252):.3f}",
        f"{abs(pnl_stressed.min())/portfolio_value*100:.2f}%",
        f"${abs(var_s)/1e6:.2f}M",
        f"${abs(cvar_s)/1e6:.2f}M"
    ],
    '2008 Shock': [
        f"{np.mean(pnl_scenario)/portfolio_value*100:.2f}%",
        f"{np.std(pnl_scenario)/portfolio_value*100:.2f}%",
        f"{np.mean(pnl_scenario)/np.std(pnl_scenario)/np.sqrt(252):.3f}",
        f"{abs(pnl_scenario.min())/portfolio_value*100:.2f}%",
        f"${abs(var_sc)/1e6:.2f}M",
        f"${abs(cvar_sc)/1e6:.2f}M"
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n", df_comparison.to_string(index=False))