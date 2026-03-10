"""
Portfolio Constraints Implementation in Python
Demonstrates various constraint types for portfolio optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class PortfolioOptimizer:
    """
    Portfolio optimization with various constraint types
    """
    
    def __init__(self, returns: np.ndarray, asset_names: Optional[List[str]] = None):
        """
        Initialize optimizer with historical returns
        
        Parameters:
        -----------
        returns : np.ndarray
            Matrix of asset returns (T x N) where T is time periods, N is assets
        asset_names : list, optional
            Names of assets
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns, rowvar=False)
        
        if asset_names is None:
            self.asset_names = [f"Asset_{i+1}" for i in range(self.n_assets)]
        else:
            self.asset_names = asset_names
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return np.dot(weights, self.mean_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)"""
        return np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - risk_free_rate) / vol if vol > 0 else 0
    
    def negative_sharpe(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Negative Sharpe for minimization"""
        return -self.portfolio_sharpe(weights, risk_free_rate)
    
    # ==================== CONSTRAINT TYPES ====================
    
    def get_basic_constraints(self) -> List[Dict]:
        """
        Basic constraint: weights sum to 1 (fully invested)
        """
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        return constraints
    
    def get_long_only_bounds(self) -> List[Tuple]:
        """
        Long-only constraint: no short selling (0 <= w_i <= 1)
        """
        return [(0, 1) for _ in range(self.n_assets)]
    
    def get_box_constraints(self, lower: float = 0.0, upper: float = 0.3) -> List[Tuple]:
        """
        Box constraints: minimum and maximum weight for each asset
        """
        return [(lower, upper) for _ in range(self.n_assets)]
    
    def get_individual_limits(self, limits: Dict[int, Tuple[float, float]]) -> List[Tuple]:
        """
        Individual asset weight limits
        
        Parameters:
        -----------
        limits : dict
            Dictionary mapping asset index to (min_weight, max_weight)
        """
        bounds = [(0, 1) for _ in range(self.n_assets)]
        for asset_idx, (min_w, max_w) in limits.items():
            bounds[asset_idx] = (min_w, max_w)
        return bounds
    
    def get_group_constraints(self, groups: Dict[str, List[int]], 
                             group_limits: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """
        Group constraints: limit total weight for asset groups
        
        Parameters:
        -----------
        groups : dict
            Dictionary mapping group name to list of asset indices
        group_limits : dict
            Dictionary mapping group name to (min_weight, max_weight)
        """
        constraints = []
        
        for group_name, asset_indices in groups.items():
            min_weight, max_weight = group_limits[group_name]
            
            # Minimum group weight constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=asset_indices, min_w=min_weight: 
                       np.sum(w[idx]) - min_w
            })
            
            # Maximum group weight constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=asset_indices, max_w=max_weight: 
                       max_w - np.sum(w[idx])
            })
        
        return constraints
    
    def get_turnover_constraint(self, current_weights: np.ndarray, 
                               max_turnover: float = 0.2) -> List[Dict]:
        """
        Turnover constraint: limit trading from current portfolio
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights
        max_turnover : float
            Maximum allowed turnover (sum of absolute changes)
        """
        return [{
            'type': 'ineq',
            'fun': lambda w, curr=current_weights, max_t=max_turnover:
                   max_t - np.sum(np.abs(w - curr))
        }]
    
    def get_tracking_error_constraint(self, benchmark_weights: np.ndarray,
                                     max_tracking_error: float = 0.05) -> List[Dict]:
        """
        Tracking error constraint: limit deviation from benchmark
        
        Parameters:
        -----------
        benchmark_weights : np.ndarray
            Benchmark portfolio weights
        max_tracking_error : float
            Maximum allowed tracking error (volatility)
        """
        def tracking_error(w):
            diff = w - benchmark_weights
            te = np.sqrt(np.dot(diff, np.dot(self.cov_matrix, diff)))
            return max_tracking_error - te
        
        return [{'type': 'ineq', 'fun': tracking_error}]
    
    def get_risk_budget_constraints(self, risk_budgets: np.ndarray) -> List[Dict]:
        """
        Risk budget constraints: each asset contributes specified % to risk
        
        Parameters:
        -----------
        risk_budgets : np.ndarray
            Target risk contribution for each asset (should sum to 1)
        """
        def risk_contribution_constraint(w):
            # Marginal contribution to risk
            port_vol = self.portfolio_volatility(w)
            if port_vol < 1e-10:
                return 0
            
            marginal_contrib = np.dot(self.cov_matrix, w) / port_vol
            risk_contrib = w * marginal_contrib / port_vol
            
            # Minimize squared difference from target
            return -np.sum((risk_contrib - risk_budgets) ** 2)
        
        return [{'type': 'ineq', 'fun': risk_contribution_constraint}]
    
    def get_cardinality_constraint(self, max_assets: int) -> List[Dict]:
        """
        Cardinality constraint: limit number of non-zero positions
        Note: This is a mixed-integer problem, approximate solution
        
        Parameters:
        -----------
        max_assets : int
            Maximum number of assets to hold
        """
        # Approximate cardinality by penalizing small weights
        def cardinality_penalty(w, threshold=0.01):
            n_active = np.sum(w > threshold)
            return max_assets - n_active
        
        return [{'type': 'ineq', 'fun': cardinality_penalty}]
    
    def get_sector_neutral_constraints(self, sectors: Dict[str, List[int]],
                                      benchmark_weights: np.ndarray) -> List[Dict]:
        """
        Sector neutral constraints: match benchmark sector weights
        
        Parameters:
        -----------
        sectors : dict
            Dictionary mapping sector name to asset indices
        benchmark_weights : np.ndarray
            Benchmark weights
        """
        constraints = []
        
        for sector_name, asset_indices in sectors.items():
            bench_sector_weight = np.sum(benchmark_weights[asset_indices])
            
            constraints.append({
                'type': 'eq',
                'fun': lambda w, idx=asset_indices, target=bench_sector_weight:
                       np.sum(w[idx]) - target
            })
        
        return constraints
    
    # ==================== OPTIMIZATION METHODS ====================
    
    def optimize_max_sharpe(self, constraints: List[Dict], 
                           bounds: List[Tuple],
                           risk_free_rate: float = 0.02) -> Dict:
        """
        Maximize Sharpe ratio with given constraints
        """
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            fun=lambda w: self.negative_sharpe(w, risk_free_rate),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            return {
                'weights': weights,
                'return': self.portfolio_return(weights),
                'volatility': self.portfolio_volatility(weights),
                'sharpe': self.portfolio_sharpe(weights, risk_free_rate),
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def optimize_min_variance(self, constraints: List[Dict],
                             bounds: List[Tuple]) -> Dict:
        """
        Minimize portfolio variance with given constraints
        """
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            fun=lambda w: self.portfolio_volatility(w) ** 2,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            return {
                'weights': weights,
                'return': self.portfolio_return(weights),
                'volatility': self.portfolio_volatility(weights),
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def optimize_target_return(self, target_return: float,
                              constraints: List[Dict],
                              bounds: List[Tuple]) -> Dict:
        """
        Minimize variance subject to target return
        """
        # Add return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda w: self.portfolio_return(w) - target_return
        }
        all_constraints = constraints + [return_constraint]
        
        return self.optimize_min_variance(all_constraints, bounds)
    
    def efficient_frontier(self, n_points: int = 50,
                          constraints: List[Dict] = None,
                          bounds: List[Tuple] = None) -> pd.DataFrame:
        """
        Generate efficient frontier
        """
        if constraints is None:
            constraints = self.get_basic_constraints()
        if bounds is None:
            bounds = self.get_long_only_bounds()
        
        # Find min and max return portfolios
        min_var = self.optimize_min_variance(constraints, bounds)
        max_sharpe = self.optimize_max_sharpe(constraints, bounds)
        
        min_return = min_var['return']
        max_return = max_sharpe['return']
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        for target in target_returns:
            result = self.optimize_target_return(target, constraints, bounds)
            if result['success']:
                results.append({
                    'return': result['return'],
                    'volatility': result['volatility'],
                    'sharpe': result['return'] / result['volatility']
                })
        
        return pd.DataFrame(results)
    
    def display_results(self, result: Dict, title: str = "Portfolio Optimization"):
        """
        Display optimization results
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
        if not result['success']:
            print(f"Optimization failed: {result.get('message', 'Unknown error')}")
            return
        
        print(f"\nExpected Return: {result['return']:.4f} ({result['return']*100:.2f}%)")
        print(f"Volatility: {result['volatility']:.4f} ({result['volatility']*100:.2f}%)")
        if 'sharpe' in result:
            print(f"Sharpe Ratio: {result['sharpe']:.4f}")
        
        print(f"\n{'Asset':<20} {'Weight':<15}")
        print("-" * 35)
        
        weights = result['weights']
        for i, (name, weight) in enumerate(zip(self.asset_names, weights)):
            if weight > 0.001:  # Only show significant weights
                print(f"{name:<20} {weight:>6.2%}")
        
        print("-" * 35)
        print(f"{'Total':<20} {np.sum(weights):>6.2%}")


# ==================== EXAMPLE USAGE ====================

def example_basic_optimization():
    """Example: Basic portfolio optimization with different constraints"""
    
    # Generate synthetic return data
    np.random.seed(42)
    n_periods = 252
    n_assets = 5
    
    # Simulate correlated returns
    mean_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09]) / 252
    volatilities = np.array([0.15, 0.20, 0.12, 0.25, 0.18]) / np.sqrt(252)
    correlation = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.4],
        [0.3, 1.0, 0.4, 0.2, 0.3],
        [0.2, 0.4, 1.0, 0.3, 0.2],
        [0.1, 0.2, 0.3, 1.0, 0.1],
        [0.4, 0.3, 0.2, 0.1, 1.0]
    ])
    
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    asset_names = ['US Equities', 'Intl Equities', 'Bonds', 'Commodities', 'Real Estate']
    
    # Create optimizer
    optimizer = PortfolioOptimizer(returns, asset_names)
    
    # Example 1: Long-only, fully invested
    print("\n" + "="*60)
    print("EXAMPLE 1: Long-Only Portfolio (Max Sharpe)")
    print("="*60)
    
    constraints = optimizer.get_basic_constraints()
    bounds = optimizer.get_long_only_bounds()
    result = optimizer.optimize_max_sharpe(constraints, bounds)
    optimizer.display_results(result, "Long-Only Max Sharpe Portfolio")
    
    # Example 2: Box constraints
    print("\n" + "="*60)
    print("EXAMPLE 2: Box Constraints (5-30% per asset)")
    print("="*60)
    
    constraints = optimizer.get_basic_constraints()
    bounds = optimizer.get_box_constraints(lower=0.05, upper=0.30)
    result = optimizer.optimize_max_sharpe(constraints, bounds)
    optimizer.display_results(result, "Box Constrained Portfolio")
    
    # Example 3: Group constraints
    print("\n" + "="*60)
    print("EXAMPLE 3: Group Constraints")
    print("="*60)
    
    groups = {
        'Equities': [0, 1],      # US and Intl Equities
        'Fixed Income': [2],      # Bonds
        'Alternatives': [3, 4]    # Commodities and Real Estate
    }
    
    group_limits = {
        'Equities': (0.40, 0.70),       # 40-70% in equities
        'Fixed Income': (0.15, 0.40),   # 15-40% in bonds
        'Alternatives': (0.10, 0.30)    # 10-30% in alternatives
    }
    
    constraints = optimizer.get_basic_constraints()
    constraints.extend(optimizer.get_group_constraints(groups, group_limits))
    bounds = optimizer.get_long_only_bounds()
    
    result = optimizer.optimize_max_sharpe(constraints, bounds)
    optimizer.display_results(result, "Group Constrained Portfolio")
    
    # Example 4: Turnover constraint
    print("\n" + "="*60)
    print("EXAMPLE 4: Turnover Constraint (Max 20% turnover)")
    print("="*60)
    
    current_weights = np.array([0.25, 0.20, 0.30, 0.15, 0.10])
    
    constraints = optimizer.get_basic_constraints()
    constraints.extend(optimizer.get_turnover_constraint(current_weights, max_turnover=0.20))
    bounds = optimizer.get_long_only_bounds()
    
    result = optimizer.optimize_max_sharpe(constraints, bounds)
    optimizer.display_results(result, "Turnover Constrained Portfolio")
    
    print("\nCurrent Weights vs. Optimized Weights:")
    print(f"{'Asset':<20} {'Current':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 65)
    for name, curr, opt in zip(asset_names, current_weights, result['weights']):
        change = opt - curr
        print(f"{name:<20} {curr:>6.2%}         {opt:>6.2%}         {change:>+6.2%}")
    print(f"\nTotal Turnover: {np.sum(np.abs(result['weights'] - current_weights)):.2%}")


def example_efficient_frontier():
    """Example: Generate and plot efficient frontier"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_periods = 252
    n_assets = 4
    
    mean_returns = np.array([0.08, 0.12, 0.10, 0.15]) / 252
    volatilities = np.array([0.12, 0.20, 0.15, 0.25]) / np.sqrt(252)
    correlation = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0]
    ])
    
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    asset_names = ['Conservative', 'Moderate', 'Growth', 'Aggressive']
    optimizer = PortfolioOptimizer(returns, asset_names)
    
    # Generate efficient frontier
    constraints = optimizer.get_basic_constraints()
    bounds = optimizer.get_long_only_bounds()
    
    ef_data = optimizer.efficient_frontier(n_points=30, constraints=constraints, bounds=bounds)
    
    print("\n" + "="*60)
    print("EFFICIENT FRONTIER ANALYSIS")
    print("="*60)
    print(f"\nGenerated {len(ef_data)} efficient portfolios")
    print(f"\nReturn Range: {ef_data['return'].min():.2%} to {ef_data['return'].max():.2%}")
    print(f"Risk Range: {ef_data['volatility'].min():.2%} to {ef_data['volatility'].max():.2%}")
    print(f"Max Sharpe: {ef_data['sharpe'].max():.4f}")
    
    return ef_data, optimizer


if __name__ == "__main__":
    # Run examples
    example_basic_optimization()
    
    print("\n" + "="*60)
    print("Generating Efficient Frontier...")
    print("="*60)
    ef_data, optimizer = example_efficient_frontier()
    
    print("\nExamples completed successfully!")