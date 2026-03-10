"""
Quadratic Programming for Quantitative Finance
===============================================

Practical QP applications specifically for quantitative engineering
and financial modeling.

Topics covered:
1. Portfolio optimization (various formulations)
2. Risk management and hedging
3. Transaction costs and constraints
4. Factor models
5. Multi-period optimization
6. Risk parity
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. BASIC MARKOWITZ PORTFOLIO OPTIMIZATION
# ============================================================================

def markowitz_portfolio(mu, Sigma, target_return=None, risk_aversion=None):
    """
    Classic Markowitz mean-variance optimization
    
    Two formulations:
    1. Minimize risk for target return (if target_return is provided)
    2. Maximize return - risk_aversion * risk (if risk_aversion is provided)
    
    Parameters:
    -----------
    mu : array, expected returns
    Sigma : array, covariance matrix
    target_return : float, target portfolio return (formulation 1)
    risk_aversion : float, risk aversion parameter (formulation 2)
    
    Returns:
    --------
    weights : optimal portfolio weights
    """
    print("="*70)
    print("MARKOWITZ PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    n = len(mu)
    print(f"\nNumber of assets: {n}")
    print(f"Expected returns: {mu}")
    print(f"Volatilities: {np.sqrt(np.diag(Sigma))}")
    
    # We'll show both formulations
    
    # Formulation 1: Minimize variance for target return
    if target_return is not None:
        print(f"\nFormulation 1: Min variance for target return = {target_return}")
        print("\nQP Problem:")
        print("  minimize    w^T Σ w")
        print("  subject to  μ^T w = target_return")
        print("              1^T w = 1")
        print("              w >= 0")
        
        try:
            import cvxpy as cp
            
            w = cp.Variable(n)
            objective = cp.Minimize(cp.quad_form(w, Sigma))
            constraints = [
                cp.sum(w) == 1,
                mu @ w >= target_return,
                w >= 0
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                print(f"\nOptimal weights:")
                for i, weight in enumerate(w.value):
                    if weight > 0.001:  # Only show significant weights
                        print(f"  Asset {i+1}: {weight:6.2%}")
                
                port_return = mu @ w.value
                port_vol = np.sqrt(problem.value)
                sharpe = port_return / port_vol
                
                print(f"\nPortfolio metrics:")
                print(f"  Return:     {port_return:6.2%}")
                print(f"  Volatility: {port_vol:6.2%}")
                print(f"  Sharpe:     {sharpe:6.3f}")
                
                return w.value
            else:
                print(f"Optimization failed: {problem.status}")
                return None
                
        except ImportError:
            print("\nCVXPY not installed. Using scipy instead...")
            from scipy.optimize import minimize
            
            def objective(w):
                return w @ Sigma @ w
            
            def constraint_return(w):
                return mu @ w - target_return
            
            def constraint_budget(w):
                return np.sum(w) - 1
            
            constraints = [
                {'type': 'eq', 'fun': constraint_budget},
                {'type': 'ineq', 'fun': constraint_return}
            ]
            bounds = [(0, 1) for _ in range(n)]
            
            result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                print(f"\nOptimal weights: {result.x}")
                return result.x
            else:
                print(f"Optimization failed")
                return None
    
    # Formulation 2: Maximize utility (return - risk_aversion * risk)
    elif risk_aversion is not None:
        print(f"\nFormulation 2: Max utility with risk aversion = {risk_aversion}")
        print("\nQP Problem:")
        print("  maximize    μ^T w - λ * w^T Σ w")
        print("  subject to  1^T w = 1")
        print("              w >= 0")
        
        try:
            import cvxpy as cp
            
            w = cp.Variable(n)
            objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                print(f"\nOptimal weights:")
                for i, weight in enumerate(w.value):
                    if weight > 0.001:
                        print(f"  Asset {i+1}: {weight:6.2%}")
                
                port_return = mu @ w.value
                port_vol = np.sqrt(w.value @ Sigma @ w.value)
                
                print(f"\nPortfolio metrics:")
                print(f"  Return:     {port_return:6.2%}")
                print(f"  Volatility: {port_vol:6.2%}")
                print(f"  Utility:    {problem.value:6.4f}")
                
                return w.value
            else:
                print(f"Optimization failed: {problem.status}")
                return None
                
        except ImportError:
            print("\nCVXPY not installed")
            return None


# ============================================================================
# 2. PORTFOLIO WITH TRANSACTION COSTS
# ============================================================================

def portfolio_with_transaction_costs(mu, Sigma, current_weights, transaction_cost):
    """
    Portfolio optimization with proportional transaction costs
    
    minimize    w^T Σ w + κ * ||w - w_current||_1
    subject to  1^T w = 1
                w >= 0
    
    where κ is the transaction cost rate
    """
    print("\n" + "="*70)
    print("PORTFOLIO REBALANCING WITH TRANSACTION COSTS")
    print("="*70)
    
    try:
        import cvxpy as cp
        
        n = len(mu)
        w = cp.Variable(n)
        
        # Transaction cost term: κ * sum of absolute changes
        trades = w - current_weights
        
        # Objective: risk + transaction costs
        risk = cp.quad_form(w, Sigma)
        costs = transaction_cost * cp.norm(trades, 1)
        objective = cp.Minimize(risk + costs)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            print(f"\nCurrent weights: {current_weights}")
            print(f"\nNew optimal weights:")
            for i, weight in enumerate(w.value):
                change = w.value[i] - current_weights[i]
                if abs(change) > 0.001:
                    print(f"  Asset {i+1}: {weight:6.2%} (change: {change:+6.2%})")
            
            total_cost = transaction_cost * np.sum(np.abs(w.value - current_weights))
            turnover = np.sum(np.abs(w.value - current_weights)) / 2
            
            print(f"\nRebalancing metrics:")
            print(f"  Turnover:         {turnover:6.2%}")
            print(f"  Transaction cost: {total_cost:6.4f}")
            print(f"  New volatility:   {np.sqrt(w.value @ Sigma @ w.value):6.2%}")
            
            return w.value
        else:
            print(f"Optimization failed: {problem.status}")
            return None
            
    except ImportError:
        print("\nCVXPY not installed")
        return None


# ============================================================================
# 3. FACTOR MODEL PORTFOLIO
# ============================================================================

def factor_model_portfolio(factor_returns, factor_loadings, specific_var, 
                           factor_risk_budgets=None):
    """
    Portfolio optimization using factor model
    
    Returns: r_i = B_i^T f + ε_i
    Risk: w^T (B Σ_f B^T + D) w
    
    where B is factor loadings matrix, Σ_f is factor covariance, D is diagonal
    matrix of specific variances
    """
    print("\n" + "="*70)
    print("FACTOR MODEL PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    try:
        import cvxpy as cp
        
        n_assets = factor_loadings.shape[0]
        n_factors = factor_loadings.shape[1]
        
        print(f"\nAssets: {n_assets}, Factors: {n_factors}")
        
        # Compute factor covariance
        Sigma_f = np.cov(factor_returns.T)
        
        # Compute return covariance using factor model
        B = factor_loadings
        D = np.diag(specific_var)
        Sigma = B @ Sigma_f @ B.T + D
        
        # Expected returns (simple average for demo)
        mu = factor_returns @ B.T
        mu = mu.mean(axis=0)
        
        w = cp.Variable(n_assets)
        
        # Risk decomposition
        factor_risk = cp.quad_form(B.T @ w, Sigma_f)
        specific_risk = cp.quad_form(w, D)
        total_risk = factor_risk + specific_risk
        
        objective = cp.Minimize(total_risk)
        
        constraints = [
            cp.sum(w) == 1,
            mu @ w >= 0.10,  # Target 10% return
            w >= 0
        ]
        
        # Optional: Factor risk budgets
        if factor_risk_budgets is not None:
            for i, budget in enumerate(factor_risk_budgets):
                # Limit exposure to each factor
                constraints.append(cp.abs(B[:, i] @ w) <= budget)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            print("\nOptimal weights:")
            for i, weight in enumerate(w.value):
                if weight > 0.001:
                    print(f"  Asset {i+1}: {weight:6.2%}")
            
            # Risk decomposition
            factor_contrib = (B.T @ w.value) @ Sigma_f @ (B.T @ w.value)
            specific_contrib = w.value @ D @ w.value
            
            print(f"\nRisk decomposition:")
            print(f"  Factor risk:   {np.sqrt(factor_contrib):6.2%}")
            print(f"  Specific risk: {np.sqrt(specific_contrib):6.2%}")
            print(f"  Total risk:    {np.sqrt(problem.value):6.2%}")
            
            # Factor exposures
            print(f"\nFactor exposures:")
            exposures = B.T @ w.value
            for i, exp in enumerate(exposures):
                print(f"  Factor {i+1}: {exp:+6.3f}")
            
            return w.value
        else:
            print(f"Optimization failed: {problem.status}")
            return None
            
    except ImportError:
        print("\nCVXPY not installed")
        return None


# ============================================================================
# 4. RISK PARITY PORTFOLIO
# ============================================================================

def risk_parity_portfolio(Sigma):
    """
    Risk parity: each asset contributes equally to portfolio risk
    
    This is actually not a pure QP problem, but we can approximate it
    using an iterative QP approach or solve it as a non-convex optimization
    
    Risk contribution: RC_i = w_i * (Σw)_i
    Goal: RC_i = RC_j for all i, j
    """
    print("\n" + "="*70)
    print("RISK PARITY PORTFOLIO")
    print("="*70)
    
    n = Sigma.shape[0]
    
    # Simple iterative approach: equal risk contribution
    # Start with equal weights
    w = np.ones(n) / n
    
    print("\nIterative risk parity optimization...")
    
    for iteration in range(100):
        # Calculate risk contributions
        marginal_risk = Sigma @ w
        risk_contributions = w * marginal_risk
        total_risk = np.sqrt(w @ Sigma @ w)
        
        # Target: equal risk contribution
        target_rc = total_risk ** 2 / n
        
        # Update weights to equalize risk contributions
        # New weight proportional to target_rc / marginal_risk
        w_new = target_rc / marginal_risk
        w_new = w_new / np.sum(w_new)  # Normalize
        
        # Check convergence
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            print(f"Converged after {iteration + 1} iterations")
            break
        
        w = w_new
    
    print("\nRisk parity weights:")
    for i, weight in enumerate(w):
        print(f"  Asset {i+1}: {weight:6.2%}")
    
    # Verify equal risk contribution
    marginal_risk = Sigma @ w
    risk_contributions = w * marginal_risk
    total_risk_sq = w @ Sigma @ w
    
    print(f"\nRisk contributions (should be equal):")
    for i, rc in enumerate(risk_contributions):
        print(f"  Asset {i+1}: {rc/total_risk_sq:6.2%}")
    
    print(f"\nPortfolio volatility: {np.sqrt(total_risk_sq):6.2%}")
    
    return w


# ============================================================================
# 5. LONG-SHORT PORTFOLIO (MARKET NEUTRAL)
# ============================================================================

def long_short_portfolio(mu, Sigma, gross_exposure=2.0, net_exposure=0.0):
    """
    Long-short portfolio optimization (e.g., 130/30 strategy)
    
    minimize    w^T Σ w - λ * μ^T w
    subject to  sum(|w|) <= gross_exposure
                sum(w) = net_exposure
    """
    print("\n" + "="*70)
    print("LONG-SHORT PORTFOLIO (MARKET NEUTRAL)")
    print("="*70)
    
    try:
        import cvxpy as cp
        
        n = len(mu)
        w = cp.Variable(n)
        
        risk_aversion = 1.0
        objective = cp.Minimize(
            risk_aversion * cp.quad_form(w, Sigma) - mu @ w
        )
        
        constraints = [
            cp.sum(w) == net_exposure,           # Net exposure (0 = market neutral)
            cp.norm(w, 1) <= gross_exposure      # Gross exposure (sum of |w|)
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            print(f"\nOptimal weights:")
            long_weights = w.value[w.value > 0.001]
            short_weights = w.value[w.value < -0.001]
            
            print(f"\nLong positions:")
            for i, weight in enumerate(w.value):
                if weight > 0.001:
                    print(f"  Asset {i+1}: {weight:+6.2%}")
            
            print(f"\nShort positions:")
            for i, weight in enumerate(w.value):
                if weight < -0.001:
                    print(f"  Asset {i+1}: {weight:+6.2%}")
            
            long_exposure = np.sum(np.maximum(w.value, 0))
            short_exposure = -np.sum(np.minimum(w.value, 0))
            
            print(f"\nExposure metrics:")
            print(f"  Long exposure:  {long_exposure:6.2%}")
            print(f"  Short exposure: {short_exposure:6.2%}")
            print(f"  Net exposure:   {np.sum(w.value):6.2%}")
            print(f"  Gross exposure: {long_exposure + short_exposure:6.2%}")
            
            port_return = mu @ w.value
            port_vol = np.sqrt(w.value @ Sigma @ w.value)
            
            print(f"\nPortfolio metrics:")
            print(f"  Expected return: {port_return:6.2%}")
            print(f"  Volatility:      {port_vol:6.2%}")
            
            return w.value
        else:
            print(f"Optimization failed: {problem.status}")
            return None
            
    except ImportError:
        print("\nCVXPY not installed")
        return None


# ============================================================================
# 6. TRACKING ERROR MINIMIZATION (INDEX TRACKING)
# ============================================================================

def index_tracking_portfolio(returns, index_returns, max_assets=None):
    """
    Minimize tracking error to an index
    
    minimize    w^T Σ w - 2 * w^T Σ_index + σ²_index
    subject to  sum(w) = 1
                w >= 0
                # of non-zero weights <= max_assets (cardinality constraint)
    """
    print("\n" + "="*70)
    print("INDEX TRACKING PORTFOLIO")
    print("="*70)
    
    try:
        import cvxpy as cp
        
        n_assets = returns.shape[1]
        
        # Compute covariances
        combined = np.column_stack([returns, index_returns])
        cov_matrix = np.cov(combined.T)
        
        Sigma = cov_matrix[:-1, :-1]  # Asset covariance
        cov_with_index = cov_matrix[:-1, -1]  # Asset-index covariance
        
        w = cp.Variable(n_assets)
        
        # Tracking error = Var(w^T r - r_index)
        tracking_error = cp.quad_form(w, Sigma) - 2 * cov_with_index @ w
        objective = cp.Minimize(tracking_error)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        # Cardinality constraint (approximate with L1 regularization)
        if max_assets is not None:
            # Can't directly enforce cardinality in convex QP
            # Use L1 penalty to encourage sparsity
            lambda_sparse = 0.001
            objective = cp.Minimize(tracking_error + lambda_sparse * cp.norm(w, 1))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            print(f"\nTracking portfolio weights:")
            non_zero = np.sum(w.value > 0.001)
            print(f"Number of holdings: {non_zero}")
            
            for i, weight in enumerate(w.value):
                if weight > 0.001:
                    print(f"  Asset {i+1}: {weight:6.2%}")
            
            # Calculate tracking error
            portfolio_returns = returns @ w.value
            tracking_diff = portfolio_returns - index_returns
            te = np.std(tracking_diff)
            
            print(f"\nTracking metrics:")
            print(f"  Tracking error (realized): {te:6.4f}")
            print(f"  Correlation with index:    {np.corrcoef(portfolio_returns, index_returns)[0,1]:6.4f}")
            
            return w.value
        else:
            print(f"Optimization failed: {problem.status}")
            return None
            
    except ImportError:
        print("\nCVXPY not installed")
        return None


# ============================================================================
# MAIN EXECUTION WITH EXAMPLES
# ============================================================================

def main():
    """Run all quantitative finance examples"""
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_assets = 5
    n_periods = 252  # Daily data for 1 year
    
    # Generate returns
    mu = np.array([0.12, 0.10, 0.08, 0.15, 0.09]) / 252  # Daily returns
    
    # Generate covariance matrix
    A = np.random.randn(n_assets, n_assets)
    Sigma = (A @ A.T) / (252 * 100)  # Scale for daily data
    
    print("\n" + "="*70)
    print(" QUANTITATIVE FINANCE QP EXAMPLES")
    print("="*70)
    print(f"\nSynthetic data generated:")
    print(f"  Assets: {n_assets}")
    print(f"  Periods: {n_periods}")
    print(f"  Annual returns: {mu * 252}")
    print(f"  Annual volatilities: {np.sqrt(np.diag(Sigma) * 252)}")
    
    # 1. Basic Markowitz
    print("\n" + "-"*70)
    weights1 = markowitz_portfolio(mu * 252, Sigma * 252, target_return=0.10)
    
    # 2. With transaction costs
    if weights1 is not None:
        print("\n" + "-"*70)
        current_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        weights2 = portfolio_with_transaction_costs(
            mu * 252, Sigma * 252, current_weights, transaction_cost=0.01
        )
    
    # 3. Factor model
    print("\n" + "-"*70)
    n_factors = 3
    factor_returns = np.random.randn(n_periods, n_factors) * 0.01
    factor_loadings = np.random.randn(n_assets, n_factors)
    specific_var = np.random.rand(n_assets) * 0.001
    
    weights3 = factor_model_portfolio(
        factor_returns, factor_loadings, specific_var
    )
    
    # 4. Risk parity
    print("\n" + "-"*70)
    weights4 = risk_parity_portfolio(Sigma * 252)
    
    # 5. Long-short
    print("\n" + "-"*70)
    weights5 = long_short_portfolio(mu * 252, Sigma * 252, gross_exposure=1.6, net_exposure=0.0)
    
    # 6. Index tracking
    print("\n" + "-"*70)
    returns = np.random.multivariate_normal(mu, Sigma, n_periods)
    index_returns = returns @ np.ones(n_assets) / n_assets + np.random.randn(n_periods) * 0.001
    
    weights6 = index_tracking_portfolio(returns, index_returns)
    
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()