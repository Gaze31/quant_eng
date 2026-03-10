import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from py_port import Portfolio


class FrontierPlotter:
    def __init__(self, portfolio):
        """
        portfolio: Portfolio object
        """
        self.portfolio = portfolio
        sns.set_style("whitegrid")

    def plot_efficient_frontier_basic(self, figsize=(12, 7)):
        """Basic efficient frontier with assets and optimal portfolios."""
        frontier = self.portfolio.efficient_frontier(n_points=100)
        min_var_w, (min_ret, min_vol, _) = self.portfolio.min_variance_portfolio()
        max_sharpe_w, (max_ret, max_vol, max_sharpe) = self.portfolio.max_sharpe_portfolio()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # individual assets
        ax.scatter(self.portfolio.std_devs, self.portfolio.mean_returns, 
                  marker='o', s=200, c='blue', alpha=0.6, label='Individual Assets', zorder=3)
        for i, name in enumerate(self.portfolio.asset_names):
            ax.annotate(name, (self.portfolio.std_devs[i], self.portfolio.mean_returns[i]), 
                       fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        # efficient frontier
        if frontier.size > 0:
            ax.plot(frontier[:, 1], frontier[:, 0], 'g-', linewidth=3, 
                   label='Efficient Frontier', zorder=2)
        
        # optimal portfolios
        ax.scatter(min_vol, min_ret, marker='*', s=800, c='gold', edgecolor='black', 
                  linewidth=2, label=f'Min Variance (Sharpe={0:.3f})', zorder=5)
        ax.scatter(max_vol, max_ret, marker='*', s=800, c='red', edgecolor='black', 
                  linewidth=2, label=f'Max Sharpe (Sharpe={max_sharpe:.3f})', zorder=5)
        
        # capital market line (CML)
        if max_vol > 0:
            cml_x = np.linspace(0, max_vol * 1.2, 100)
            cml_y = self.portfolio.risk_free_rate + (max_ret - self.portfolio.risk_free_rate) / max_vol * cml_x
            ax.plot(cml_x, cml_y, 'r--', linewidth=2, label='Capital Market Line (CML)', alpha=0.7)
        
        # risk-free rate
        ax.scatter(0, self.portfolio.risk_free_rate, marker='X', s=300, c='black', 
                  label=f'Risk-Free Rate ({self.portfolio.risk_free_rate:.2%})', zorder=5)
        
        ax.set_xlabel('Volatility (Annualized Std Dev)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Expected Return (Annualized)', fontsize=11, fontweight='bold')
        ax.set_title('Efficient Frontier with Capital Market Line', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_efficient_frontier_heatmap(self, figsize=(12, 7), n_portfolios=5000):
        """Scatter plot with color-coded Sharpe ratios."""
        np.random.seed(42)
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.random(self.portfolio.n_assets)
            weights /= np.sum(weights)
            port_ret, port_vol, sharpe = self.portfolio.portfolio_stats(weights)
            results[0, i] = port_ret
            results[1, i] = port_vol
            results[2, i] = sharpe
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='none')
        
        # add optimal portfolios
        min_var_w, (min_ret, min_vol, _) = self.portfolio.min_variance_portfolio()
        max_sharpe_w, (max_ret, max_vol, max_sharpe) = self.portfolio.max_sharpe_portfolio()
        
        ax.scatter(min_vol, min_ret, marker='*', s=800, c='gold', edgecolor='black', 
                  linewidth=2, label='Min Variance', zorder=5)
        ax.scatter(max_vol, max_ret, marker='*', s=800, c='red', edgecolor='black', 
                  linewidth=2, label='Max Sharpe', zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Volatility', fontsize=11, fontweight='bold')
        ax.set_ylabel('Expected Return', fontsize=11, fontweight='bold')
        ax.set_title(f'Efficient Frontier ({n_portfolios} Random Portfolios)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_asset_correlation_heatmap(self, figsize=(8, 6)):
        """Correlation matrix heatmap of assets."""
        corr = self.portfolio.returns.corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Asset Correlation Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig, ax

    def plot_optimal_weights(self, figsize=(12, 5)):
        """Bar chart comparing weights of min variance vs max Sharpe portfolios."""
        min_var_w, _ = self.portfolio.min_variance_portfolio()
        max_sharpe_w, _ = self.portfolio.max_sharpe_portfolio()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # min variance
        ax1.bar(self.portfolio.asset_names, min_var_w, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax1.set_title('Minimum Variance Portfolio', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(min_var_w):
            ax1.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)
        
        # max sharpe
        ax2.bar(self.portfolio.asset_names, max_sharpe_w, color='coral', alpha=0.7)
        ax2.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax2.set_title('Maximum Sharpe Ratio Portfolio', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(max_sharpe_w):
            ax2.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)
        
        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_risk_return_comparison(self, figsize=(10, 6)):
        """Scatter comparing risk vs return for each asset."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.portfolio.n_assets))
        for i, name in enumerate(self.portfolio.asset_names):
            ax.scatter(self.portfolio.std_devs[i], self.portfolio.mean_returns[i], 
                      s=300, label=name, color=colors[i], alpha=0.7, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Risk (Volatility)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Return', fontsize=11, fontweight='bold')
        ax.set_title('Risk-Return Profile of Individual Assets', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    def plot_all(self):
        """Generate all frontier plots in subplots."""
        fig = plt.figure(figsize=(16, 12))
        
        # frontier basic
        ax1 = plt.subplot(2, 3, 1)
        self.portfolio.efficient_frontier(n_points=100)
        min_var_w, (min_ret, min_vol, _) = self.portfolio.min_variance_portfolio()
        max_sharpe_w, (max_ret, max_vol, max_sharpe) = self.portfolio.max_sharpe_portfolio()
        frontier = self.portfolio.efficient_frontier(n_points=100)
        ax1.scatter(self.portfolio.std_devs, self.portfolio.mean_returns, marker='o', s=100, alpha=0.6)
        if frontier.size > 0:
            ax1.plot(frontier[:, 1], frontier[:, 0], 'g-', linewidth=2)
        ax1.scatter(min_vol, min_ret, marker='*', s=600, c='gold', edgecolor='black')
        ax1.scatter(max_vol, max_ret, marker='*', s=600, c='red', edgecolor='black')
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Return')
        ax1.set_title('Efficient Frontier')
        ax1.grid(True, alpha=0.3)
        
        # correlation heatmap
        ax2 = plt.subplot(2, 3, 2)
        corr = self.portfolio.returns.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax2, square=True)
        ax2.set_title('Asset Correlation')
        
        # weights comparison
        ax3 = plt.subplot(2, 3, 3)
        x = np.arange(len(self.portfolio.asset_names))
        width = 0.35
        ax3.bar(x - width/2, min_var_w, width, label='Min Variance', alpha=0.7)
        ax3.bar(x + width/2, max_sharpe_w, width, label='Max Sharpe', alpha=0.7)
        ax3.set_ylabel('Weight')
        ax3.set_title('Portfolio Weights')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.portfolio.asset_names)
        ax3.legend()
        
        # risk-return scatter
        ax4 = plt.subplot(2, 3, 4)
        for i, name in enumerate(self.portfolio.asset_names):
            ax4.scatter(self.portfolio.std_devs[i], self.portfolio.mean_returns[i], s=200, label=name, alpha=0.7)
        ax4.set_xlabel('Risk (Volatility)')
        ax4.set_ylabel('Return')
        ax4.set_title('Asset Risk-Return Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
import numpy as np
import pandas as pd
from py_port import Portfolio
import sys, os
from src.frontier_plot import FrontierPlotter


def main():
    # synthetic returns for 4 assets over 1 year
    np.random.seed(42)
    n_days = 252
    
    returns_df = pd.DataFrame({
        'Stock_A': np.random.normal(0.0010, 0.018, n_days),
        'Stock_B': np.random.normal(0.0008, 0.015, n_days),
        'Bond': np.random.normal(0.0003, 0.005, n_days),
        'Real_Estate': np.random.normal(0.0009, 0.012, n_days)
    })
    
    # create portfolio and plotter
    portfolio = Portfolio(returns_df, risk_free_rate=0.02)
    plotter = FrontierPlotter(portfolio)
    
    # generate plots
    print("Generating efficient frontier visualizations...\n")
    
    portfolio.summary()
    
    print("\n✓ Plot 1: Basic Efficient Frontier with CML")
    plotter.plot_efficient_frontier_basic()
    plt.show()
    
    print("✓ Plot 2: Heatmap (Sharpe Ratio Colored)")
    plotter.plot_efficient_frontier_heatmap(n_portfolios=3000)
    plt.show()
    
    print("✓ Plot 3: Asset Correlation Heatmap")
    plotter.plot_asset_correlation_heatmap()
    plt.show()
    
    print("✓ Plot 4: Optimal Portfolio Weights")
    plotter.plot_optimal_weights()
    plt.show()
    
    print("✓ Plot 5: Risk-Return Comparison")
    plotter.plot_risk_return_comparison()
    plt.show()
    
    print("✓ Plot 6: All Plots Combined")
    plotter.plot_all()
    plt.show()

if __name__ == "__main__":
    main()    




