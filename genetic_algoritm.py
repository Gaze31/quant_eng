"""
Genetic Algorithms - Practical Guide for Quantitative Finance
Complete examples, exercises, and best practices
FIXED VERSION - Works on any operating system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

# Get current directory for saving files
SAVE_DIR = os.getcwd()

# ============================================================================
# PRACTICAL TIPS AND BEST PRACTICES
# ============================================================================

class GABestPractices:
    """
    Collection of best practices for implementing GAs in finance
    """
    
    @staticmethod
    def parameter_selection_guide():
        """Guide for selecting GA parameters"""
        guide = """
        ================================================================
        GENETIC ALGORITHM PARAMETER SELECTION GUIDE
        ================================================================
        
        1. POPULATION SIZE
        ------------------
        - Small problems (< 10 variables): 30-50
        - Medium problems (10-50 variables): 50-100
        - Large problems (> 50 variables): 100-200
        - Rule of thumb: 10 × number of variables
        
        2. CROSSOVER RATE
        ------------------
        - Typical range: 0.6 - 0.9
        - High values (0.8-0.9): More exploration, slower convergence
        - Low values (0.6-0.7): More exploitation, faster convergence
        - Recommendation for finance: 0.75 - 0.85
        
        3. MUTATION RATE
        ------------------
        - Typical range: 0.01 - 0.1
        - Rule: 1 / gene_length is a good starting point
        - High values: Prevent premature convergence, more random
        - Low values: Fine-tuning, local search
        - Recommendation for finance: 0.02 - 0.05
        
        4. ELITISM
        ------------------
        - Number of best individuals preserved: 2-10
        - Typically 2-5% of population size
        - Always use elitism to preserve best solutions
        
        5. NUMBER OF GENERATIONS
        ------------------
        - Until convergence (no improvement for N generations)
        - Typical: 100-500 generations
        - Early stopping if fitness plateaus for 20-50 generations
        
        6. SELECTION METHOD
        ------------------
        - Tournament (size 2-5): Good balance, most common
        - Roulette Wheel: Can cause premature convergence
        - Rank-based: Good for varying fitness ranges
        - Recommendation: Tournament with size 3
        
        7. CROSSOVER TYPE
        ------------------
        - Real-valued problems: Arithmetic or BLX-α
        - Binary problems: Single-point or uniform
        - Portfolio optimization: Arithmetic crossover
        
        8. MUTATION TYPE
        ------------------
        - Gaussian: Good for fine-tuning
        - Uniform: Good for exploration
        - Adaptive: Adjusts during evolution
        - Recommendation: Gaussian with decreasing variance
        
        ================================================================
        """
        print(guide)
    
    @staticmethod
    def overfitting_prevention():
        """Strategies to prevent overfitting in financial GAs"""
        strategies = """
        ================================================================
        PREVENTING OVERFITTING IN FINANCIAL GAs
        ================================================================
        
        1. WALK-FORWARD ANALYSIS
        ------------------
        - Train on in-sample period
        - Test on out-of-sample period
        - Roll forward and repeat
        
        2. CROSS-VALIDATION
        ------------------
        - K-fold cross-validation on time series
        - Purging and embargo for financial data
        
        3. COMPLEXITY PENALTIES
        ------------------
        - Add penalty for number of parameters
        - Penalize extreme parameter values
        - Use AIC/BIC-type criteria
        
        4. TRANSACTION COSTS
        ------------------
        - Always include realistic trading costs
        - Model slippage and market impact
        - Penalize excessive trading
        
        5. ROBUSTNESS CHECKS
        ------------------
        - Test on multiple market regimes
        - Monte Carlo simulation of parameters
        - Stress testing
        
        6. REGULARIZATION
        ------------------
        - Limit parameter ranges
        - Prefer simpler strategies
        - Use ensemble methods
        
        ================================================================
        """
        print(strategies)


# ============================================================================
# EXERCISE 1: MEAN-VARIANCE PORTFOLIO OPTIMIZATION
# ============================================================================

class Exercise1_MeanVariancePortfolio:
    """
    Exercise: Implement mean-variance portfolio optimization
    Task: Find portfolio that maximizes Sharpe ratio with constraints
    """
    
    @staticmethod
    def problem_statement():
        print("""
        ================================================================
        EXERCISE 1: Mean-Variance Portfolio Optimization
        ================================================================
        
        OBJECTIVE:
        Maximize Sharpe Ratio subject to constraints
        
        CONSTRAINTS:
        1. Weights sum to 1
        2. No short selling (weights >= 0)
        3. Maximum position size: 40%
        4. Minimum allocation: 5% if invested
        
        DATA:
        5 assets with historical returns
        
        TASKS:
        1. Implement fitness function with constraints
        2. Create custom mutation operator respecting constraints
        3. Compare with equal-weight and maximum Sharpe portfolios
        4. Analyze results and interpret allocations
        ================================================================
        """)
    
    @staticmethod
    def solution():
        """Complete solution to Exercise 1"""
        
        # Generate sample data
        np.random.seed(42)
        n_periods = 252 * 3  # 3 years of daily data
        n_assets = 5
        
        # Asset characteristics
        annual_returns = np.array([0.08, 0.12, 0.10, 0.15, 0.07])
        annual_vols = np.array([0.15, 0.25, 0.18, 0.30, 0.12])
        
        # Correlation matrix
        correlation = np.array([
            [1.0, 0.3, 0.2, 0.1, 0.4],
            [0.3, 1.0, 0.5, 0.4, 0.2],
            [0.2, 0.5, 1.0, 0.3, 0.3],
            [0.1, 0.4, 0.3, 1.0, 0.2],
            [0.4, 0.2, 0.3, 0.2, 1.0]
        ])
        
        # Generate returns
        daily_returns = annual_returns / 252
        daily_vols = annual_vols / np.sqrt(252)
        cov_daily = np.outer(daily_vols, daily_vols) * correlation
        
        returns = np.random.multivariate_normal(daily_returns, cov_daily, n_periods)
        
        # Calculate statistics
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        
        print("\nAsset Statistics:")
        print("-" * 60)
        for i in range(n_assets):
            print(f"Asset {i+1}: Return={mean_returns[i]:.2%}, "
                  f"Volatility={np.sqrt(cov_matrix[i,i]):.2%}")
        
        # Define fitness function with constraints
        def constrained_sharpe_ratio(weights, risk_free_rate=0.02):
            """
            Sharpe ratio with penalty for constraint violations
            Constraints:
            1. Weights sum to 1
            2. No short selling
            3. Max position 40%
            4. Min position 5% if invested
            """
            # Normalize weights
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Calculate Sharpe ratio
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_var)
            
            if portfolio_std == 0:
                sharpe = -1000
            else:
                sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            
            # Apply penalties for constraint violations
            penalty = 0
            
            # Constraint 1: weights sum to 1 (already normalized)
            
            # Constraint 2: no short selling
            if np.any(weights < 0):
                penalty += 10 * np.sum(np.abs(weights[weights < 0]))
            
            # Constraint 3: max position 40%
            excess = weights[weights > 0.40] - 0.40
            if len(excess) > 0:
                penalty += 10 * np.sum(excess)
            
            # Constraint 4: min position 5% if invested
            small_positions = (weights > 0) & (weights < 0.05)
            if np.any(small_positions):
                penalty += 5 * np.sum(small_positions)
            
            return sharpe - penalty
        
        # Custom GA implementation
        population_size = 100
        num_generations = 200
        
        # Initialize population
        population = np.random.dirichlet(np.ones(n_assets), population_size)
        
        best_fitness_history = []
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(num_generations):
            # Evaluate fitness
            fitness = np.array([constrained_sharpe_ratio(ind) for ind in population])
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Sharpe = {best_fitness:.4f}")
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            elite_count = 5
            elite_idx = np.argsort(fitness)[-elite_count:]
            new_population.extend([population[i].copy() for i in elite_idx])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                idx = np.random.choice(population_size, tournament_size, replace=False)
                parent1 = population[idx[np.argmax(fitness[idx])]].copy()
                
                idx = np.random.choice(population_size, tournament_size, replace=False)
                parent2 = population[idx[np.argmax(fitness[idx])]].copy()
                
                # Arithmetic crossover
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Normalize to sum to 1
                child = child / np.sum(child)
                
                # Mutation - Dirichlet noise
                if np.random.random() < 0.1:
                    noise = np.random.dirichlet(np.ones(n_assets) * 10)
                    child = 0.9 * child + 0.1 * noise
                    child = child / np.sum(child)
                
                new_population.append(child)
            
            population = np.array(new_population[:population_size])
        
        # Results
        print("\n" + "="*60)
        print("OPTIMAL PORTFOLIO FOUND:")
        print("="*60)
        
        optimal_weights = best_solution / np.sum(best_solution)
        optimal_return = np.dot(optimal_weights, mean_returns)
        optimal_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        optimal_sharpe = (optimal_return - 0.02) / optimal_vol
        
        print(f"\nSharpe Ratio: {optimal_sharpe:.4f}")
        print(f"Expected Return: {optimal_return:.2%}")
        print(f"Volatility: {optimal_vol:.2%}")
        print("\nOptimal Weights:")
        for i in range(n_assets):
            if optimal_weights[i] > 0.01:
                print(f"  Asset {i+1}: {optimal_weights[i]:.2%}")
        
        # Compare with benchmarks
        equal_weight = np.ones(n_assets) / n_assets
        eq_return = np.dot(equal_weight, mean_returns)
        eq_vol = np.sqrt(np.dot(equal_weight.T, np.dot(cov_matrix, equal_weight)))
        eq_sharpe = (eq_return - 0.02) / eq_vol
        
        print("\n" + "-"*60)
        print("COMPARISON WITH EQUAL-WEIGHT PORTFOLIO:")
        print("-"*60)
        print(f"Equal-Weight Sharpe: {eq_sharpe:.4f}")
        print(f"Equal-Weight Return: {eq_return:.2%}")
        print(f"Equal-Weight Volatility: {eq_vol:.2%}")
        print(f"\nImprovement in Sharpe: {((optimal_sharpe - eq_sharpe) / eq_sharpe * 100):.2f}%")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Evolution
        axes[0, 0].plot(best_fitness_history, linewidth=2)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Allocation comparison
        x = np.arange(n_assets)
        width = 0.35
        axes[0, 1].bar(x - width/2, optimal_weights, width, label='GA Optimized', alpha=0.8)
        axes[0, 1].bar(x + width/2, equal_weight, width, label='Equal Weight', alpha=0.8)
        axes[0, 1].set_xlabel('Asset')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].set_title('Portfolio Allocation Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'A{i+1}' for i in range(n_assets)])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Efficient frontier
        n_portfolios = 5000
        random_weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
        random_returns = np.array([np.dot(w, mean_returns) for w in random_weights])
        random_vols = np.array([np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) 
                               for w in random_weights])
        
        axes[1, 0].scatter(random_vols, random_returns, alpha=0.3, s=5, label='Random Portfolios')
        axes[1, 0].scatter(optimal_vol, optimal_return, color='red', s=300, 
                          marker='*', label='GA Optimized', zorder=5)
        axes[1, 0].scatter(eq_vol, eq_return, color='green', s=200, 
                          marker='s', label='Equal Weight', zorder=5)
        axes[1, 0].set_xlabel('Volatility')
        axes[1, 0].set_ylabel('Expected Return')
        axes[1, 0].set_title('Risk-Return Space')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        optimal_portfolio_returns = np.dot(returns, optimal_weights)
        equal_portfolio_returns = np.dot(returns, equal_weight)
        
        cum_optimal = np.cumprod(1 + optimal_portfolio_returns)
        cum_equal = np.cumprod(1 + equal_portfolio_returns)
        
        axes[1, 1].plot(cum_optimal, label='GA Optimized', linewidth=2)
        axes[1, 1].plot(cum_equal, label='Equal Weight', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Trading Days')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].set_title('Portfolio Performance Backtest')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, 'exercise1_portfolio_optimization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()
        
        return optimal_weights, best_fitness_history


# ============================================================================
# EXERCISE 2: OPTIONS PRICING MODEL CALIBRATION
# ============================================================================

class Exercise2_OptionCalibration:
    """
    Exercise: Calibrate Black-Scholes implied volatility surface
    """
    
    @staticmethod
    def problem_statement():
        print("""
        ================================================================
        EXERCISE 2: Options Pricing Model Calibration
        ================================================================
        
        OBJECTIVE:
        Calibrate a volatility surface model to market option prices
        
        MODEL:
        Implied volatility as function of strike and maturity:
        σ(K, T) = a + b*T + c*(K - S0)/S0 + d*((K - S0)/S0)^2
        
        PARAMETERS TO OPTIMIZE:
        a, b, c, d (4 parameters)
        
        FITNESS FUNCTION:
        Minimize mean squared error between model and market prices
        
        TASKS:
        1. Generate synthetic market option prices
        2. Implement fitness function (pricing error)
        3. Optimize parameters using GA
        4. Visualize fitted volatility surface
        ================================================================
        """)
    
    @staticmethod
    def solution():
        """Complete solution to Exercise 2"""
        from scipy.stats import norm
        
        # Black-Scholes formula
        def black_scholes_call(S, K, T, r, sigma):
            """Black-Scholes call option price"""
            if T <= 0:
                return max(S - K, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call_price
        
        # Generate market data
        S0 = 100  # Current stock price
        r = 0.05  # Risk-free rate
        
        # Strikes and maturities
        strikes = np.array([80, 90, 95, 100, 105, 110, 120])
        maturities = np.array([0.25, 0.5, 1.0, 2.0])  # 3M, 6M, 1Y, 2Y
        
        # True volatility surface parameters (to be discovered)
        true_params = [0.20, 0.05, -0.3, 0.5]
        
        def vol_surface(K, T, params):
            """Volatility surface model"""
            a, b, c, d = params
            moneyness = (K - S0) / S0
            sigma = a + b * T + c * moneyness + d * moneyness**2
            return max(sigma, 0.05)  # Floor at 5%
        
        # Generate market prices
        market_prices = {}
        for T in maturities:
            for K in strikes:
                sigma = vol_surface(K, T, true_params)
                price = black_scholes_call(S0, K, T, r, sigma)
                # Add noise
                noise = np.random.normal(0, 0.05 * price)
                market_prices[(K, T)] = price + noise
        
        print(f"\nGenerated {len(market_prices)} market option prices")
        
        # Fitness function
        def calibration_error(params):
            """Calculate mean squared pricing error"""
            total_error = 0
            count = 0
            
            for (K, T), market_price in market_prices.items():
                sigma = vol_surface(K, T, params)
                model_price = black_scholes_call(S0, K, T, r, sigma)
                error = (model_price - market_price)**2
                total_error += error
                count += 1
            
            mse = total_error / count
            return -np.sqrt(mse)  # Negative RMSE (maximize negative = minimize RMSE)
        
        # Simple GA implementation
        population_size = 80
        num_generations = 150
        
        # Initialize population
        population = np.random.uniform(0.0, 1.0, size=(population_size, 4))
        
        best_fitness_history = []
        best_params = None
        best_fitness = -np.inf
        
        print("\nCalibrating volatility surface...")
        for generation in range(num_generations):
            # Evaluate fitness
            fitness = np.array([calibration_error(ind) for ind in population])
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_params = population[gen_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best RMSE = {-best_fitness:.6f}")
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_count = 3
            elite_idx = np.argsort(fitness)[-elite_count:]
            new_population.extend([population[i].copy() for i in elite_idx])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                idx = np.random.choice(population_size, 3, replace=False)
                parent1 = population[idx[np.argmax(fitness[idx])]].copy()
                
                idx = np.random.choice(population_size, 3, replace=False)
                parent2 = population[idx[np.argmax(fitness[idx])]].copy()
                
                # Arithmetic crossover
                if np.random.random() < 0.8:
                    alpha = np.random.random()
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    child = parent1.copy()
                
                # Gaussian mutation
                if np.random.random() < 0.05:
                    child += np.random.normal(0, 0.1, 4)
                    child = np.clip(child, 0.0, 1.0)
                
                new_population.append(child)
            
            population = np.array(new_population[:population_size])
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS:")
        print("="*60)
        print(f"\nTrue Parameters: {true_params}")
        print(f"Estimated Parameters: {best_params}")
        print(f"RMSE: {-best_fitness:.6f}")
        
        # Calculate parameter errors
        param_errors = np.abs(np.array(best_params) - np.array(true_params))
        print(f"\nParameter Errors: {param_errors}")
        print(f"Average Error: {np.mean(param_errors):.4f}")
        
        # Visualize results
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D volatility surface
        ax1 = fig.add_subplot(221, projection='3d')
        
        K_grid = np.linspace(80, 120, 30)
        T_grid = np.linspace(0.25, 2.0, 30)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        
        # True surface
        vol_true = np.array([[vol_surface(k, t, true_params) 
                             for k in K_grid] for t in T_grid])
        ax1.plot_surface(K_mesh, T_mesh, vol_true, alpha=0.7, 
                        cmap='viridis', label='True')
        
        # Estimated surface
        vol_est = np.array([[vol_surface(k, t, best_params) 
                           for k in K_grid] for t in T_grid])
        ax1.plot_wireframe(K_mesh, T_mesh, vol_est, color='red', 
                          alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Volatility Surface (True vs Estimated)')
        
        # Calibration evolution
        ax2 = fig.add_subplot(222)
        ax2.plot([-f for f in best_fitness_history], linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Calibration Progress')
        ax2.grid(True, alpha=0.3)
        
        # Pricing errors
        ax3 = fig.add_subplot(223)
        
        actual_prices = []
        model_prices = []
        
        for (K, T), market_price in market_prices.items():
            sigma = vol_surface(K, T, best_params)
            model_price = black_scholes_call(S0, K, T, r, sigma)
            actual_prices.append(market_price)
            model_prices.append(model_price)
        
        ax3.scatter(actual_prices, model_prices, alpha=0.6, s=100, edgecolors='black')
        ax3.plot([min(actual_prices), max(actual_prices)], 
                [min(actual_prices), max(actual_prices)], 
                'r--', linewidth=2, label='Perfect Fit')
        ax3.set_xlabel('Market Price')
        ax3.set_ylabel('Model Price')
        ax3.set_title('Model vs Market Prices')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Implied volatility smile
        ax4 = fig.add_subplot(224)
        
        for T in maturities:
            vol_true_curve = [vol_surface(K, T, true_params) for K in strikes]
            vol_est_curve = [vol_surface(K, T, best_params) for K in strikes]
            
            ax4.plot(strikes, vol_true_curve, 'o-', label=f'True T={T}', alpha=0.7)
            ax4.plot(strikes, vol_est_curve, 's--', label=f'Est T={T}', alpha=0.7)
        
        ax4.set_xlabel('Strike Price')
        ax4.set_ylabel('Implied Volatility')
        ax4.set_title('Volatility Smile by Maturity')
        ax4.legend(ncol=2, fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, 'exercise2_option_calibration.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nCalibration visualization saved to: {save_path}")
        plt.close()
        
        return best_params, best_fitness_history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENETIC ALGORITHMS - PRACTICAL GUIDE")
    print("="*70)
    print(f"Files will be saved to: {SAVE_DIR}")
    
    # Display best practices
    GABestPractices.parameter_selection_guide()
    
    print("\n" + "="*70)
    print("EXERCISE 1: PORTFOLIO OPTIMIZATION")
    print("="*70)
    Exercise1_MeanVariancePortfolio.problem_statement()
    ex1_weights, ex1_history = Exercise1_MeanVariancePortfolio.solution()
    
    print("\n" + "="*70)
    print("EXERCISE 2: OPTIONS CALIBRATION")
    print("="*70)
    Exercise2_OptionCalibration.problem_statement()
    ex2_params, ex2_history = Exercise2_OptionCalibration.solution()
    
    print("\n" + "="*70)
    print("ALL EXERCISES COMPLETED!")
    print("="*70)
    print(f"\nGenerated files in: {SAVE_DIR}")
    print("  - exercise1_portfolio_optimization.png")
    print("  - exercise2_option_calibration.png") 


