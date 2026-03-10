import numpy as np
import matplotlib.pyplot as plt

# Try to import cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("="*70)
    print("CVXPY not installed!")
    print("Install with: pip install cvxpy")
    print("="*70)
    print("\nShowing code examples without execution...\n")

def check_cvxpy():
    """Check if CVXPY is available"""
    if not CVXPY_AVAILABLE:
        print("⚠️  CVXPY is required to run these examples.")
        print("Install it using: pip install cvxpy")
        return False
    return True

# ============================================================================
# 1. SIMPLE LINEAR PROGRAMMING
# ============================================================================

def example_1_simple_lp():
    """
    Simple Linear Programming Example
    
    Maximize: x + 2y
    Subject to:
        x + y <= 3
        x >= 0
        y >= 0
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Linear Programming")
    print("="*70)
    
    if not check_cvxpy():
        print("\nProblem:")
        print("  Maximize: x + 2y")
        print("  Subject to: x + y <= 3, x >= 0, y >= 0")
        return
    
    # Define variables
    x = cp.Variable()
    y = cp.Variable()
    
    # Define objective function (CVXPY minimizes by default, so negate for maximize)
    objective = cp.Maximize(x + 2*y)
    
    # Define constraints
    constraints = [
        x + y <= 3,
        x >= 0,
        y >= 0
    ]
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    
    # Display results
    print("\nOptimization Status:", problem.status)
    print(f"Optimal Value: {result:.4f}")
    print(f"Optimal x: {x.value:.4f}")
    print(f"Optimal y: {y.value:.4f}")
    
    # Visualize
    visualize_lp(x.value, y.value)

def visualize_lp(opt_x, opt_y):
    """Visualize the linear programming problem"""
    x_vals = np.linspace(0, 4, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Plot constraint: x + y <= 3
    y_constraint = 3 - x_vals
    plt.fill_between(x_vals, 0, y_constraint, where=(y_constraint >= 0), 
                     alpha=0.3, label='Feasible Region')
    plt.plot(x_vals, y_constraint, 'b-', linewidth=2, label='x + y = 3')
    
    # Plot objective function contours
    X, Y = np.meshgrid(np.linspace(0, 4, 50), np.linspace(0, 4, 50))
    Z = X + 2*Y
    plt.contour(X, Y, Z, levels=10, alpha=0.5)
    
    # Plot optimal point
    plt.plot(opt_x, opt_y, 'r*', markersize=20, label=f'Optimal: ({opt_x:.2f}, {opt_y:.2f})')
    
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Linear Programming Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# 2. QUADRATIC PROGRAMMING
# ============================================================================

def example_2_quadratic_programming():
    """
    Quadratic Programming Example
    
    Minimize: (1/2)x'Px + q'x
    Subject to: Gx <= h
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Quadratic Programming")
    print("="*70)
    
    if not check_cvxpy():
        print("\nMinimize: (1/2)x'Px + q'x subject to constraints")
        return
    
    # Problem data
    P = np.array([[2, 0.5], [0.5, 1]])
    q = np.array([1, 1])
    G = np.array([[-1, 0], [0, -1], [1, 1]])
    h = np.array([0, 0, 1])
    
    # Define variable
    x = cp.Variable(2)
    
    # Define objective (quadratic form)
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
    
    # Define constraints
    constraints = [G @ x <= h]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    
    print("\nOptimization Status:", problem.status)
    print(f"Optimal Value: {result:.4f}")
    print(f"Optimal x: {x.value}")
    
    # Visualize
    visualize_qp(P, q, G, h, x.value)

def visualize_qp(P, q, G, h, opt_x):
    """Visualize quadratic programming problem"""
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute objective function
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i,j], X2[i,j]])
            Z[i,j] = 0.5 * x.T @ P @ x + q.T @ x
    
    plt.figure(figsize=(10, 6))
    
    # Plot objective contours
    plt.contour(X1, X2, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Objective Value')
    
    # Plot constraints
    x1_line = np.linspace(-0.5, 1.5, 100)
    plt.plot([0, 0], [-0.5, 1.5], 'r--', linewidth=2, label='x₁ >= 0')
    plt.plot([-0.5, 1.5], [0, 0], 'g--', linewidth=2, label='x₂ >= 0')
    x2_line = 1 - x1_line
    plt.plot(x1_line, x2_line, 'b--', linewidth=2, label='x₁ + x₂ <= 1')
    
    # Shade feasible region
    x1_fill = np.linspace(0, 1, 100)
    x2_fill = 1 - x1_fill
    plt.fill_between(x1_fill, 0, x2_fill, alpha=0.2)
    
    # Plot optimal point
    plt.plot(opt_x[0], opt_x[1], 'r*', markersize=20, 
             label=f'Optimal: ({opt_x[0]:.3f}, {opt_x[1]:.3f})')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Quadratic Programming Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# 3. PORTFOLIO OPTIMIZATION
# ============================================================================

def example_3_portfolio_optimization():
    """
    Portfolio Optimization (Markowitz Mean-Variance)
    
    Minimize: (1/2)w'Σw
    Subject to:
        μ'w >= target_return
        1'w = 1
        w >= 0
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Portfolio Optimization (Markowitz)")
    print("="*70)
    
    if not check_cvxpy():
        print("\nMinimize portfolio variance subject to return target")
        return
    
    # Sample data: 5 assets
    np.random.seed(42)
    n_assets = 5
    
    # Expected returns
    mu = np.array([0.12, 0.10, 0.08, 0.15, 0.09])
    
    # Covariance matrix (must be positive semidefinite)
    Sigma = np.array([
        [0.04, 0.01, 0.02, 0.01, 0.015],
        [0.01, 0.03, 0.01, 0.02, 0.010],
        [0.02, 0.01, 0.05, 0.01, 0.020],
        [0.01, 0.02, 0.01, 0.06, 0.015],
        [0.015, 0.010, 0.020, 0.015, 0.045]
    ])
    
    # Target return
    target_return = 0.10
    
    # Define variable (portfolio weights)
    w = cp.Variable(n_assets)
    
    # Objective: minimize variance (risk)
    risk = cp.quad_form(w, Sigma)
    objective = cp.Minimize(risk)
    
    # Constraints
    constraints = [
        mu @ w >= target_return,  # Minimum return
        cp.sum(w) == 1,           # Weights sum to 1
        w >= 0                     # No short selling
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    
    # Calculate portfolio metrics
    portfolio_return = mu @ w.value
    portfolio_risk = np.sqrt(w.value @ Sigma @ w.value)
    
    print("\nOptimization Status:", problem.status)
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
    print(f"  Portfolio Risk (Std Dev): {portfolio_risk:.4f} ({portfolio_risk*100:.2f}%)")
    print(f"  Sharpe Ratio (assuming rf=0): {portfolio_return/portfolio_risk:.4f}")
    
    print(f"\nOptimal Portfolio Weights:")
    asset_names = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
    for i, (name, weight) in enumerate(zip(asset_names, w.value)):
        print(f"  {name}: {weight:6.2%} (Return: {mu[i]:.2%})")
    
    # Visualize weights
    visualize_portfolio(asset_names, w.value, mu)
    
    # Compute efficient frontier
    compute_efficient_frontier(mu, Sigma)

def visualize_portfolio(asset_names, weights, returns):
    """Visualize portfolio allocation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart of weights
    colors = plt.cm.Set3(range(len(asset_names)))
    ax1.pie(weights, labels=asset_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
    
    # Bar chart comparing weights and returns
    x = np.arange(len(asset_names))
    width = 0.35
    
    ax2.bar(x - width/2, weights * 100, width, label='Weight (%)', color='skyblue')
    ax2.bar(x + width/2, returns * 100, width, label='Return (%)', color='lightcoral')
    
    ax2.set_xlabel('Assets', fontsize=12)
    ax2.set_ylabel('Percentage', fontsize=12)
    ax2.set_title('Weights vs Expected Returns', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(asset_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compute_efficient_frontier(mu, Sigma):
    """Compute and plot the efficient frontier"""
    n_assets = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    risks = []
    returns_achieved = []
    
    for target_return in target_returns:
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, Sigma)
        objective = cp.Minimize(risk)
        
        constraints = [
            mu @ w >= target_return,
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                risks.append(np.sqrt(w.value @ Sigma @ w.value))
                returns_achieved.append(mu @ w.value)
        except:
            pass
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(risks) * 100, np.array(returns_achieved) * 100, 
             'b-', linewidth=2, label='Efficient Frontier')
    plt.scatter(np.sqrt(np.diag(Sigma)) * 100, mu * 100, 
                c='red', s=100, marker='o', label='Individual Assets')
    
    plt.xlabel('Risk (Standard Deviation) %', fontsize=12)
    plt.ylabel('Expected Return %', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# 4. LEAST SQUARES REGRESSION
# ============================================================================

def example_4_least_squares():
    """
    Least Squares Regression using CVXPY
    
    Minimize: ||Ax - b||²
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Least Squares Regression")
    print("="*70)
    
    if not check_cvxpy():
        print("\nMinimize: ||Ax - b||²")
        return
    
    # Generate synthetic data
    np.random.seed(42)
    m, n = 100, 2
    A = np.random.randn(m, n)
    x_true = np.array([3, -2])
    b = A @ x_true + 0.5 * np.random.randn(m)
    
    # Define variable
    x = cp.Variable(n)
    
    # Objective: minimize squared error
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    
    # Solve
    problem = cp.Problem(objective)
    result = problem.solve()
    
    print("\nOptimization Status:", problem.status)
    print(f"Optimal Value (MSE): {result:.4f}")
    print(f"True parameters: {x_true}")
    print(f"Estimated parameters: {x.value}")
    print(f"Estimation error: {np.linalg.norm(x.value - x_true):.4f}")
    
    # Compare with NumPy solution
    x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"NumPy lstsq solution: {x_numpy}")

# ============================================================================
# 5. L1 REGULARIZATION (LASSO)
# ============================================================================

def example_5_lasso_regression():
    """
    LASSO Regression (L1 Regularization)
    
    Minimize: ||Ax - b||² + λ||x||₁
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: LASSO Regression (L1 Regularization)")
    print("="*70)
    
    if not check_cvxpy():
        print("\nMinimize: ||Ax - b||² + λ||x||₁")
        return
    
    # Generate sparse data
    np.random.seed(42)
    m, n = 50, 20
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[0:5] = [5, -3, 2, -1, 4]  # Only first 5 coefficients are non-zero
    b = A @ x_true + 0.3 * np.random.randn(m)
    
    # Regularization parameter
    lambda_reg = 0.5
    
    # Define variable
    x = cp.Variable(n)
    
    # Objective: squared error + L1 penalty
    objective = cp.Minimize(cp.sum_squares(A @ x - b) + lambda_reg * cp.norm(x, 1))
    
    # Solve
    problem = cp.Problem(objective)
    result = problem.solve()
    
    print("\nOptimization Status:", problem.status)
    print(f"Number of non-zero coefficients:")
    print(f"  True: {np.sum(x_true != 0)}")
    print(f"  Estimated: {np.sum(np.abs(x.value) > 0.1)}")
    
    # Visualize coefficients
    visualize_lasso(x_true, x.value)

def visualize_lasso(x_true, x_estimated):
    """Visualize LASSO results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.stem(x_true, basefmt=' ', label='True')
    plt.stem(x_estimated, basefmt=' ', markerfmt='ro', linefmt='r-', label='Estimated')
    plt.xlabel('Coefficient Index', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('Coefficient Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_true, x_estimated, s=100, alpha=0.6)
    min_val = min(x_true.min(), x_estimated.min())
    max_val = max(x_true.max(), x_estimated.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    plt.xlabel('True Coefficients', fontsize=12)
    plt.ylabel('Estimated Coefficients', fontsize=12)
    plt.title('True vs Estimated', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CVXPY BASICS - CONVEX OPTIMIZATION IN PYTHON")
    print("="*70)
    
    if not CVXPY_AVAILABLE:
        print("\n⚠️  CVXPY is not installed!")
        print("\nTo run these examples, install CVXPY:")
        print("  pip install cvxpy")
        print("\nOptional: For faster solving, install additional solvers:")
        print("  pip install cvxopt")
        print("  pip install scs")
        print("\n" + "="*70)
    
    # Run examples
    example_1_simple_lp()
    example_2_quadratic_programming()
    example_3_portfolio_optimization()
    example_4_least_squares()
    example_5_lasso_regression()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
    print("\nKey CVXPY Concepts:")
    print("  • Variables: cp.Variable()")
    print("  • Objectives: cp.Minimize() / cp.Maximize()")
    print("  • Constraints: List of inequalities/equalities")
    print("  • Problem: cp.Problem(objective, constraints)")
    print("  • Solve: problem.solve()")
    print("\nCommon Optimization Problems:")
    print("  • Linear Programming (LP)")
    print("  • Quadratic Programming (QP)")
    print("  • Portfolio Optimization")
    print("  • Regression (Least Squares, LASSO, Ridge)")
    print("  • Support Vector Machines")
    print("="*70)