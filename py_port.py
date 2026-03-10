import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Portfolio:
    def __init__(self, returns_df, risk_free_rate=0.02):
        """
        returns_df: pd.DataFrame with asset returns (columns = assets, rows = periods)
        risk_free_rate: annual risk-free rate (default 2%)
        """
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.asset_names = returns_df.columns.tolist()
        self.risk_free_rate = risk_free_rate
        
        # annualized stats
        self.mean_returns = returns_df.mean() * 252
        self.cov_matrix = returns_df.cov() * 252
        self.std_devs = returns_df.std() * np.sqrt(252)

    def portfolio_stats(self, weights):
        """Calculate portfolio return, volatility, Sharpe ratio."""
        port_return = np.sum(self.mean_returns * weights)
        port_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_std if port_std > 0 else 0
        return port_return, port_std, sharpe

    def neg_sharpe(self, weights):
        """Negative Sharpe for minimization."""
        return -self.portfolio_stats(weights)[2]

    def portfolio_volatility(self, weights):
        """Portfolio volatility (std dev)."""
        return np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))

    def min_variance_portfolio(self):
        """Find minimum variance (most conservative) portfolio."""
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        result = minimize(self.portfolio_volatility, init_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, self.portfolio_stats(result.x)

    def max_sharpe_portfolio(self):
        """Find maximum Sharpe ratio (optimal risk-adjusted) portfolio."""
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        result = minimize(self.neg_sharpe, init_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x, self.portfolio_stats(result.x)

    def efficient_frontier(self, n_points=100):
        """Generate efficient frontier (risk-return tradeoff curve)."""
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), n_points)
        frontiers = []
        
        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(self.mean_returns * w) - target_ret}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
            result = minimize(self.portfolio_volatility, init_weights, 
                             method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                frontiers.append([target_ret, self.portfolio_volatility(result.x)])
        
        return np.array(frontiers) if frontiers else np.array([])

    def plot_efficient_frontier(self):
        """Plot efficient frontier and optimal portfolios."""
        frontier = self.efficient_frontier(n_points=50)
        min_var_w, (min_ret, min_vol, _) = self.min_variance_portfolio()
        max_sharpe_w, (max_ret, max_vol, max_sharpe) = self.max_sharpe_portfolio()
        
        plt.figure(figsize=(10, 6))
        
        # individual assets
        plt.scatter(self.std_devs, self.mean_returns, marker='o', s=100, 
                   label='Individual Assets', alpha=0.7)
        for i, name in enumerate(self.asset_names):
            plt.annotate(name, (self.std_devs[i], self.mean_returns[i]), fontsize=8)
        
        # efficient frontier
        if frontier.size > 0:
            plt.plot(frontier[:, 1], frontier[:, 0], 'b-', linewidth=2, label='Efficient Frontier')
        
        # optimal portfolios
        plt.scatter(min_vol, min_ret, marker='*', s=500, c='green', label='Min Variance', zorder=5)
        plt.scatter(max_vol, max_ret, marker='*', s=500, c='red', label='Max Sharpe', zorder=5)
        
        plt.xlabel('Volatility (Std Dev)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print portfolio summary."""
        min_var_w, (min_ret, min_vol, min_sharpe) = self.min_variance_portfolio()
        max_sharpe_w, (max_ret, max_vol, max_sharpe) = self.max_sharpe_portfolio()
        
        print("\n=== Portfolio Optimization Summary ===\n")
        print("Individual Assets:")
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: Return={self.mean_returns[i]:.4f}, Vol={self.std_devs[i]:.4f}")
        
        print("\n--- Minimum Variance Portfolio ---")
        print(f"Expected Return: {min_ret:.4f}")
        print(f"Volatility: {min_vol:.4f}")
        print(f"Sharpe Ratio: {min_sharpe:.4f}")
        print("Weights:")
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: {min_var_w[i]:.4f}")
        
        print("\n--- Maximum Sharpe Ratio Portfolio ---")
        print(f"Expected Return: {max_ret:.4f}")
        print(f"Volatility: {max_vol:.4f}")
        print(f"Sharpe Ratio: {max_sharpe:.4f}")
        print("Weights:")
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: {max_sharpe_w[i]:.4f}")
import numpy as np
import pandas as pd
# from src.portfolio_opt import Portfolio

def main():
    # generate synthetic daily returns for 3 assets over 1 year
    np.random.seed(42)
    n_days = 252
    
    returns_df = pd.DataFrame({
        'Stock_A': np.random.normal(0.0008, 0.015, n_days),
        'Stock_B': np.random.normal(0.0010, 0.018, n_days),
        'Bond': np.random.normal(0.0003, 0.005, n_days)
    })
    
    # create portfolio and optimize
    portfolio = Portfolio(returns_df, risk_free_rate=0.02)
    portfolio.summary()
    portfolio.plot_efficient_frontier()

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

def test_portfolio_weights_sum_to_one():
    returns_df = pd.DataFrame({
        'A': np.random.normal(0.0008, 0.015, 252),
        'B': np.random.normal(0.0010, 0.018, 252)
    })
    portfolio = Portfolio(returns_df)
    min_var_w, _ = portfolio.min_variance_portfolio()
    assert np.isclose(min_var_w.sum(), 1.0)

def test_max_sharpe_exists():
    returns_df = pd.DataFrame({
        'A': np.random.normal(0.0008, 0.015, 252),
        'B': np.random.normal(0.0010, 0.018, 252),
        'C': np.random.normal(0.0005, 0.010, 252)
    })
    portfolio = Portfolio(returns_df)
    max_sharpe_w, (ret, vol, sharpe) = portfolio.max_sharpe_portfolio()
    assert len(max_sharpe_w) == 3
    assert ret > 0 and vol > 0 and sharpe > 0

def test_efficient_frontier_non_empty():
    returns_df = pd.DataFrame({
        'A': np.random.normal(0.0008, 0.015, 252),
        'B': np.random.normal(0.0010, 0.018, 252)
    })
    portfolio = Portfolio(returns_df)
    frontier = portfolio.efficient_frontier(n_points=20)
    assert frontier.shape[0] > 0                