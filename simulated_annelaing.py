import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Any

class SimulatedAnnealing:
    """
    Generic Simulated Annealing optimizer for continuous and discrete problems.
    """
    
    def __init__(
        self,
        objective_function: Callable,
        initial_solution: np.ndarray,
        neighbor_function: Callable,
        T0: float = 100.0,
        Tf: float = 0.01,
        alpha: float = 0.95,
        iterations_per_temp: int = 100,
        max_iterations: int = 10000
    ):
        """
        Parameters:
        -----------
        objective_function : function to minimize
        initial_solution : starting point
        neighbor_function : function to generate neighboring solutions
        T0 : initial temperature
        Tf : final temperature
        alpha : cooling rate (geometric cooling)
        iterations_per_temp : iterations at each temperature
        max_iterations : maximum total iterations
        """
        self.objective = objective_function
        self.neighbor_func = neighbor_function
        self.T0 = T0
        self.Tf = Tf
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp
        self.max_iterations = max_iterations
        
        # Tracking
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.current_energy = objective_function(initial_solution)
        self.best_energy = self.current_energy
        
        # History for analysis
        self.energy_history = []
        self.best_energy_history = []
        self.temperature_history = []
        self.acceptance_history = []
        
    def acceptance_probability(self, delta_E: float, T: float) -> float:
        """Metropolis criterion"""
        if delta_E < 0:
            return 1.0
        return np.exp(-delta_E / T)
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the optimization
        
        Returns:
        --------
        best_solution : optimal solution found
        best_energy : objective value at best solution
        """
        T = self.T0
        total_iterations = 0
        
        while T > self.Tf and total_iterations < self.max_iterations:
            accepted_moves = 0
            
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                new_solution = self.neighbor_func(self.current_solution)
                new_energy = self.objective(new_solution)
                
                # Calculate energy difference
                delta_E = new_energy - self.current_energy
                
                # Acceptance decision
                if np.random.rand() < self.acceptance_probability(delta_E, T):
                    self.current_solution = new_solution
                    self.current_energy = new_energy
                    accepted_moves += 1
                    
                    # Update best solution if improved
                    if new_energy < self.best_energy:
                        self.best_solution = new_solution.copy()
                        self.best_energy = new_energy
                
                # Track history
                self.energy_history.append(self.current_energy)
                self.best_energy_history.append(self.best_energy)
                self.temperature_history.append(T)
                
                total_iterations += 1
            
            # Track acceptance rate
            acceptance_rate = accepted_moves / self.iterations_per_temp
            self.acceptance_history.append(acceptance_rate)
            
            if verbose and total_iterations % 1000 == 0:
                print(f"Iter: {total_iterations}, T: {T:.4f}, "
                      f"Current: {self.current_energy:.6f}, "
                      f"Best: {self.best_energy:.6f}, "
                      f"Accept rate: {acceptance_rate:.2%}")
            
            # Cool down
            T *= self.alpha
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best objective value: {self.best_energy:.6f}")
            print(f"Total iterations: {total_iterations}")
        
        return self.best_solution, self.best_energy
    
    def plot_convergence(self):
        """Visualize optimization progress"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy over iterations
        axes[0, 0].plot(self.energy_history, alpha=0.6, label='Current Energy')
        axes[0, 0].plot(self.best_energy_history, 'r-', linewidth=2, label='Best Energy')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Energy Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temperature schedule
        axes[0, 1].plot(self.temperature_history)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Temperature')
        axes[0, 1].set_title('Cooling Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceptance rate
        axes[1, 0].plot(self.acceptance_history)
        axes[1, 0].set_xlabel('Temperature Step')
        axes[1, 0].set_ylabel('Acceptance Rate')
        axes[1, 0].set_title('Acceptance Rate Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        # Log-scale energy
        axes[1, 1].semilogy(self.best_energy_history)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Best Objective (log scale)')
        axes[1, 1].set_title('Best Energy (Log Scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def rastrigin_function(x: np.ndarray, A: float = 10.0) -> float:
    """
    Rastrigin function: a common test function with many local minima
    Global minimum at x = [0, 0, ...] with f(x) = 0
    """
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_neighbor(x: np.ndarray, step_size: float = 0.5) -> np.ndarray:
    """Generate neighbor by adding random noise"""
    return x + np.random.uniform(-step_size, step_size, size=x.shape)

# Test the basic implementation
if __name__ == "__main__":
    # Initial solution
    dim = 5
    x0 = np.random.uniform(-5, 5, dim)
    
    # Create optimizer
    sa = SimulatedAnnealing(
        objective_function=rastrigin_function,
        initial_solution=x0,
        neighbor_function=rastrigin_neighbor,
        T0=100.0,
        Tf=0.01,
        alpha=0.95,
        iterations_per_temp=100
    )
    
    # Optimize
    best_x, best_f = sa.optimize(verbose=True)
    
    print(f"\nBest solution found: {best_x}")
    print(f"Best objective value: {best_f:.6f}")
    print(f"Global optimum is at x=[0,0,...] with f(x)=0")
    
    # Plot convergence
    sa.plot_convergence()
import pandas as pd
from scipy.stats import norm

class PortfolioOptimizer:
    """
    Portfolio optimization with cardinality constraints using SA
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        max_assets: int = 20,
        min_weight: float = 0.01,
        max_weight: float = 0.20,
        risk_aversion: float = 1.0
    ):
        """
        Parameters:
        -----------
        returns : historical returns matrix (T x N)
        max_assets : maximum number of assets in portfolio
        min_weight : minimum weight for active positions
        max_weight : maximum weight per asset
        risk_aversion : risk aversion parameter (higher = more risk averse)
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.max_assets = max_assets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion
        
        # Calculate statistics
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns.T)
        
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Minimize: risk_aversion * variance - expected_return
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Objective: maximize Sharpe-like ratio
        return self.risk_aversion * portfolio_variance - portfolio_return
    
    def generate_neighbor(self, weights: np.ndarray) -> np.ndarray:
        """
        Generate neighboring portfolio with controlled changes
        """
        new_weights = weights.copy()
        active_positions = np.where(weights > 0)[0]
        inactive_positions = np.where(weights == 0)[0]
        
        # Choose a random operation
        operation = np.random.choice(['adjust', 'swap', 'add_remove'])
        
        if operation == 'adjust' and len(active_positions) > 0:
            # Adjust weights of two active positions
            if len(active_positions) >= 2:
                i, j = np.random.choice(active_positions, 2, replace=False)
                change = np.random.uniform(0.01, 0.05)
                new_weights[i] += change
                new_weights[j] -= change
                
        elif operation == 'swap' and len(active_positions) > 0 and len(inactive_positions) > 0:
            # Remove one asset and add another
            remove_idx = np.random.choice(active_positions)
            add_idx = np.random.choice(inactive_positions)
            transfer = new_weights[remove_idx]
            new_weights[remove_idx] = 0
            new_weights[add_idx] = transfer
            
        elif operation == 'add_remove':
            if len(active_positions) < self.max_assets and len(inactive_positions) > 0:
                # Add a new asset
                add_idx = np.random.choice(inactive_positions)
                new_weights[add_idx] = self.min_weight
                # Proportionally reduce other weights
                if np.sum(new_weights) > 0:
                    new_weights *= (1 - self.min_weight) / (np.sum(new_weights) - self.min_weight)
            elif len(active_positions) > 1:
                # Remove an asset
                remove_idx = np.random.choice(active_positions)
                transfer = new_weights[remove_idx]
                new_weights[remove_idx] = 0
                # Redistribute to others
                if np.sum(new_weights) > 0:
                    new_weights[active_positions] *= (1 + transfer) / np.sum(new_weights[active_positions])
        
        # Ensure constraints
        new_weights = self._enforce_constraints(new_weights)
        
        return new_weights
    
    def _enforce_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Enforce portfolio constraints"""
        weights = np.maximum(weights, 0)  # Non-negative
        
        # Enforce max weight constraint
        weights = np.minimum(weights, self.max_weight)
        
        # Remove weights below minimum threshold
        weights[weights < self.min_weight] = 0
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            # If all zeros, create equal-weight portfolio with max_assets
            n_active = min(self.max_assets, self.n_assets)
            active_idx = np.random.choice(self.n_assets, n_active, replace=False)
            weights[active_idx] = 1.0 / n_active
        
        # Enforce cardinality constraint
        active_count = np.sum(weights > 0)
        if active_count > self.max_assets:
            # Keep only top max_assets by weight
            sorted_idx = np.argsort(weights)[::-1]
            keep_idx = sorted_idx[:self.max_assets]
            new_weights = np.zeros_like(weights)
            new_weights[keep_idx] = weights[keep_idx]
            weights = new_weights / np.sum(new_weights)
        
        return weights
    
    def optimize_portfolio(self, verbose: bool = True) -> dict:
        """Run SA optimization and return results"""
        
        # Initial solution: equal-weight portfolio with random assets
        initial_weights = np.zeros(self.n_assets)
        n_initial = min(self.max_assets, self.n_assets)
        initial_active = np.random.choice(self.n_assets, n_initial, replace=False)
        initial_weights[initial_active] = 1.0 / n_initial
        
        # Run SA
        sa = SimulatedAnnealing(
            objective_function=self.objective_function,
            initial_solution=initial_weights,
            neighbor_function=self.generate_neighbor,
            T0=10.0,
            Tf=0.001,
            alpha=0.95,
            iterations_per_temp=100,
            max_iterations=50000
        )
        
        best_weights, best_obj = sa.optimize(verbose=verbose)
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(best_weights, self.mean_returns)
        portfolio_volatility = np.sqrt(np.dot(best_weights, np.dot(self.cov_matrix, best_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        results = {
            'weights': best_weights,
            'active_assets': np.sum(best_weights > 0),
            'expected_return': portfolio_return * 252,  # Annualized
            'volatility': portfolio_volatility * np.sqrt(252),  # Annualized
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
            'objective_value': best_obj,
            'optimizer': sa
        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Generate synthetic returns data
    np.random.seed(42)
    n_periods = 252  # 1 year of daily data
    n_assets = 50
    
    # Simulate returns
    mean_returns = np.random.uniform(0.0001, 0.001, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    returns = np.random.randn(n_periods, n_assets) * volatilities + mean_returns
    
    # Create asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(
        returns=returns,
        max_assets=15,
        min_weight=0.02,
        max_weight=0.15,
        risk_aversion=2.0
    )
    
    results = optimizer.optimize_portfolio(verbose=True)
    
    # Display results
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Number of active assets: {results['active_assets']}")
    print(f"Expected annual return: {results['expected_return']:.2%}")
    print(f"Annual volatility: {results['volatility']:.2%}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.4f}")
    print("\nPortfolio Weights:")
    print("-" * 40)
    
    active_weights = [(asset_names[i], w) for i, w in enumerate(results['weights']) if w > 0]
    active_weights.sort(key=lambda x: x[1], reverse=True)
    
    for asset, weight in active_weights:
        print(f"{asset:15s}: {weight:6.2%}")
    
    # Plot convergence
    results['optimizer'].plot_convergence()   
class HestonCalibration:
    """
    Calibrate Heston stochastic volatility model to market option prices
    """
    
    def __init__(
        self,
        market_prices: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        spot_price: float,
        risk_free_rate: float
    ):
        """
        Parameters:
        -----------
        market_prices : observed option prices
        strikes : strike prices
        maturities : time to maturity
        spot_price : current stock price
        risk_free_rate : risk-free interest rate
        """
        self.market_prices = market_prices
        self.strikes = strikes
        self.maturities = maturities
        self.S0 = spot_price
        self.r = risk_free_rate
        
    def heston_price(self, params: np.ndarray) -> np.ndarray:
        """
        Price options using Heston model (simplified/approximation)
        
        params: [v0, kappa, theta, sigma, rho]
        v0: initial variance
        kappa: mean reversion speed
        theta: long-term variance
        sigma: volatility of volatility
        rho: correlation
        """
        v0, kappa, theta, sigma, rho = params
        
        # Simple approximation using Black-Scholes with adjusted volatility
        # In practice, use proper Heston pricing (Fourier transform)
        prices = np.zeros_like(self.market_prices)
        
        for i in range(len(self.strikes)):
            K = self.strikes[i]
            T = self.maturities[i]
            
            # Average variance over time period
            avg_var = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
            vol = np.sqrt(avg_var)
            
            # Black-Scholes formula
            d1 = (np.log(self.S0 / K) + (self.r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            prices[i] = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        
        return prices
    
    def objective_function(self, params: np.ndarray) -> float:
        """Mean squared pricing error"""
        model_prices = self.heston_price(params)
        return np.mean((model_prices - self.market_prices)**2)
    
    def generate_neighbor(self, params: np.ndarray) -> np.ndarray:
        """Generate neighboring parameter set"""
        new_params = params.copy()
        
        # Perturb one or two parameters
        n_perturb = np.random.choice([1, 2])
        perturb_idx = np.random.choice(5, n_perturb, replace=False)
        
        for idx in perturb_idx:
            # Adaptive step size based on parameter
            if idx == 0:  # v0
                step = np.random.uniform(-0.01, 0.01)
            elif idx == 1:  # kappa
                step = np.random.uniform(-0.1, 0.1)
            elif idx == 2:  # theta
                step = np.random.uniform(-0.01, 0.01)
            elif idx == 3:  # sigma
                step = np.random.uniform(-0.05, 0.05)
            else:  # rho
                step = np.random.uniform(-0.1, 0.1)
            
            new_params[idx] += step
        
        # Enforce constraints
        new_params[0] = np.clip(new_params[0], 0.001, 1.0)  # v0 > 0
        new_params[1] = np.clip(new_params[1], 0.01, 10.0)  # kappa > 0
        new_params[2] = np.clip(new_params[2], 0.001, 1.0)  # theta > 0
        new_params[3] = np.clip(new_params[3], 0.01, 2.0)   # sigma > 0
        new_params[4] = np.clip(new_params[4], -0.99, 0.99) # -1 < rho < 1
        
        # Feller condition: 2*kappa*theta > sigma^2
        if 2 * new_params[1] * new_params[2] <= new_params[3]**2:
            new_params[3] = np.sqrt(2 * new_params[1] * new_params[2]) * 0.9
        
        return new_params
    
    def calibrate(self, verbose: bool = True) -> dict:
        """Run calibration"""
        
        # Initial guess
        initial_params = np.array([0.04, 2.0, 0.04, 0.3, -0.5])
        
        # Run SA
        sa = SimulatedAnnealing(
            objective_function=self.objective_function,
            initial_solution=initial_params,
            neighbor_function=self.generate_neighbor,
            T0=1.0,
            Tf=0.0001,
            alpha=0.95,
            iterations_per_temp=50,
            max_iterations=20000
        )
        
        best_params, best_error = sa.optimize(verbose=verbose)
        
        # Calculate final prices
        calibrated_prices = self.heston_price(best_params)
        
        results = {
            'v0': best_params[0],
            'kappa': best_params[1],
            'theta': best_params[2],
            'sigma': best_params[3],
            'rho': best_params[4],
            'mse': best_error,
            'rmse': np.sqrt(best_error),
            'market_prices': self.market_prices,
            'model_prices': calibrated_prices,
            'optimizer': sa
        }
        
        return results

# Example usage
if __name__ == "__main__":
    # Generate synthetic market data
    np.random.seed(42)
    
    # True Heston parameters
    true_params = np.array([0.04, 2.5, 0.04, 0.3, -0.7])
    
    # Market setup
    S0 = 100
    r = 0.05
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
    
    # Create calibrator with true prices (synthetic market)
    temp_calibrator = HestonCalibration(
        market_prices=np.zeros(5),
        strikes=strikes,
        maturities=maturities,
        spot_price=S0,
        risk_free_rate=r
    )
    market_prices = temp_calibrator.heston_price(true_params)
    
    # Add some noise
    market_prices += np.random.normal(0, 0.1, size=market_prices.shape)
    
    # Now calibrate
    calibrator = HestonCalibration(
        market_prices=market_prices,
        strikes=strikes,
        maturities=maturities,
        spot_price=S0,
        risk_free_rate=r
    )
    
    results = calibrator.calibrate(verbose=True)
    
    # Display results
    print("\n" + "="*60)
    print("HESTON MODEL CALIBRATION RESULTS")
    print("="*60)
    print(f"\nTrue vs Calibrated Parameters:")
    param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    print(f"{'Parameter':<10} {'True':>10} {'Calibrated':>12} {'Error':>10}")
    print("-" * 45)
    for i, name in enumerate(param_names):
        print(f"{name:<10} {true_params[i]:>10.4f} {results[name]:>12.4f} "
              f"{abs(true_params[i] - results[name]):>10.4f}")
    
    print(f"\nRoot Mean Squared Error: {results['rmse']:.6f}")
    
    print(f"\n{'Strike':<10} {'Market':>10} {'Model':>10} {'Error':>10}")
    print("-" * 42)
    for i in range(len(strikes)):
        error = results['model_prices'][i] - results['market_prices'][i]
        print(f"{strikes[i]:<10.2f} {results['market_prices'][i]:>10.4f} "
              f"{results['model_prices'][i]:>10.4f} {error:>10.4f}")
    
    # Plot convergence
    results['optimizer'].plot_convergence()     