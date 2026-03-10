import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    """
    Portfolio optimization class implementing various optimization strategies.
    """
    
    def __init__(self, returns, risk_free_rate=0.02):
        """
        Initialize optimizer with return data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns for each asset (columns = assets)
        risk_free_rate : float
            Annual risk-free rate
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def portfolio_performance(self, weights):
        """
        Calculate portfolio return and volatility.
        
        Returns:
        --------
        tuple: (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Negative Sharpe ratio for minimization."""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Portfolio volatility for minimization."""
        return self.portfolio_performance(weights)[1]
    
    def max_sharpe_portfolio(self, constraints=None):
        """
        Find portfolio with maximum Sharpe ratio.
        
        Returns:
        --------
        dict: Optimal weights and performance metrics
        """
        # Constraints: weights sum to 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)
        
        return {
            'weights': optimal_weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def min_volatility_portfolio(self):
        """
        Find minimum variance portfolio.
        
        Returns:
        --------
        dict: Optimal weights and performance metrics
        """
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)
        
        return {
            'weights': optimal_weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def efficient_frontier(self, n_portfolios=100):
        """
        Generate efficient frontier portfolios.
        
        Parameters:
        -----------
        n_portfolios : int
            Number of portfolios to generate
            
        Returns:
        --------
        pd.DataFrame: Returns, volatilities, and Sharpe ratios
        """
        # Find range of target returns
        min_vol_port = self.min_volatility_portfolio()
        max_sharpe_port = self.max_sharpe_portfolio()
        
        min_ret = min_vol_port['return']
        max_ret = self.mean_returns.max() * 252
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target in target_returns:
            cons = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target}
            )
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            init_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(
                self.portfolio_volatility,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                ret, vol, sharpe = self.portfolio_performance(result.x)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def equal_weight_portfolio(self):
        """
        Calculate equal-weighted portfolio (1/N strategy).
        """
        weights = np.array([1/self.n_assets] * self.n_assets)
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def risk_parity_portfolio(self):
        """
        Risk parity portfolio (equal risk contribution).
        """
        def risk_contribution(weights):
            portfolio_vol = self.portfolio_volatility(weights)
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            risk_contrib = risk_contribution(weights)
            # Each asset should contribute equally to total risk
            target_contrib = np.ones(self.n_assets) / self.n_assets
            return np.sum((risk_contrib - target_contrib)**2)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.001, 1) for _ in range(self.n_assets))
        init_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            risk_parity_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)
        portfolio_vol = vol
        
        return {
            'weights': optimal_weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }
    
    def black_litterman(self, views, view_confidences, market_caps=None):
        """
        Black-Litterman model for portfolio optimization.
        
        Parameters:
        -----------
        views : dict
            {'Asset': expected_return} - Your views on asset returns
        view_confidences : dict
            {'Asset': confidence} - Confidence in your views (0-1)
        market_caps : dict
            {'Asset': market_cap} - Market capitalizations (optional)
        """
        # If no market caps, use equal weights for market portfolio
        if market_caps is None:
            market_weights = np.array([1/self.n_assets] * self.n_assets)
        else:
            total_cap = sum(market_caps.values())
            market_weights = np.array([market_caps[col]/total_cap for col in self.returns.columns])
        
        # Implied equilibrium returns
        delta = 2.5  # Risk aversion coefficient
        pi = delta * np.dot(self.cov_matrix * 252, market_weights)
        
        # Views matrix (P) and views vector (Q)
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset, ret) in enumerate(views.items()):
            asset_idx = list(self.returns.columns).index(asset)
            P[i, asset_idx] = 1
            Q[i] = ret
        
        # View uncertainty (Omega)
        omega = np.diag([1/view_confidences[asset] for asset in views.keys()])
        
        # Black-Litterman formula
        tau = 0.025  # Scaling factor
        M_inv = np.linalg.inv(tau * self.cov_matrix * 252)
        P_omega_inv = np.dot(P.T, np.linalg.inv(omega))
        
        bl_returns = np.linalg.inv(M_inv + np.dot(P_omega_inv, P)).dot(
            np.dot(M_inv, pi) + np.dot(P_omega_inv, Q)
        )
        
        # Optimize with Black-Litterman returns
        self.mean_returns = pd.Series(bl_returns / 252, index=self.returns.columns)
        return self.max_sharpe_portfolio()
    
    def plot_efficient_frontier(self, show_assets=True):
        """
        Plot the efficient frontier with optimal portfolios.
        """
        # Generate efficient frontier
        frontier = self.efficient_frontier(n_portfolios=50)
        
        # Calculate key portfolios
        max_sharpe = self.max_sharpe_portfolio()
        min_vol = self.min_volatility_portfolio()
        equal_weight = self.equal_weight_portfolio()
        
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.plot(frontier['volatility'], frontier['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot individual assets
        if show_assets:
            asset_returns = self.mean_returns * 252
            asset_vols = np.sqrt(np.diag(self.cov_matrix * 252))
            plt.scatter(asset_vols, asset_returns, c='gray', 
                       marker='o', s=100, alpha=0.6, label='Individual Assets')
            
            for i, txt in enumerate(self.returns.columns):
                plt.annotate(txt, (asset_vols[i], asset_returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Plot optimal portfolios
        plt.scatter(max_sharpe['volatility'], max_sharpe['return'], 
                   c='red', marker='*', s=500, label='Max Sharpe', zorder=5)
        plt.scatter(min_vol['volatility'], min_vol['return'], 
                   c='green', marker='*', s=500, label='Min Volatility', zorder=5)
        plt.scatter(equal_weight['volatility'], equal_weight['return'], 
                   c='orange', marker='D', s=200, label='Equal Weight', zorder=5)
        
        plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier and Optimal Portfolios', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt


# Example usage
if __name__ == "__main__":
    # Generate sample data for 5 assets
    np.random.seed(42)
    n_days = 252 * 3  # 3 years of daily data
    n_assets = 5
    
    asset_names = ['Stock A', 'Stock B', 'Stock C', 'Bond', 'Commodity']
    
    # Generate correlated returns
    mean_returns = np.array([0.0008, 0.0006, 0.0010, 0.0003, 0.0005])
    volatilities = np.array([0.02, 0.018, 0.025, 0.008, 0.022])
    
    correlation_matrix = np.array([
        [1.0, 0.7, 0.6, 0.2, 0.3],
        [0.7, 1.0, 0.5, 0.1, 0.4],
        [0.6, 0.5, 1.0, 0.15, 0.35],
        [0.2, 0.1, 0.15, 1.0, 0.05],
        [0.3, 0.4, 0.35, 0.05, 1.0]
    ])
    
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns_data, columns=asset_names)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.02)
    
    # Calculate different portfolio strategies
    print("="*60)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*60)
    
    strategies = {
        'Max Sharpe Ratio': optimizer.max_sharpe_portfolio(),
        'Min Volatility': optimizer.min_volatility_portfolio(),
        'Equal Weight': optimizer.equal_weight_portfolio(),
        'Risk Parity': optimizer.risk_parity_portfolio()
    }
    
    for strategy_name, result in strategies.items():
        print(f"\n{strategy_name}:")
        print(f"  Expected Return: {result['return']:.2%}")
        print(f"  Volatility: {result['volatility']:.2%}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"  Weights:")
        for i, asset in enumerate(asset_names):
            print(f"    {asset}: {result['weights'][i]:.2%}")
    
    print("\n" + "="*60)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(show_assets=True)
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    print("\nEfficient frontier plot saved as 'efficient_frontier.png'")