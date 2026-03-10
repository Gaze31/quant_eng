import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML

class GradientDescent:
    """
    Implementation of various Gradient Descent algorithms
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize Gradient Descent optimizer
        
        Parameters:
        -----------
        learning_rate : float
            Step size for parameter updates
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = {'params': [], 'cost': []}
    
    def vanilla_gradient_descent(self, X, y, initial_params=None):
        """
        Standard Batch Gradient Descent
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix (m samples x n features)
        y : numpy array
            Target values (m samples)
        initial_params : numpy array, optional
            Initial parameter values
        
        Returns:
        --------
        params : numpy array
            Optimized parameters
        """
        m, n = X.shape
        
        # Initialize parameters
        if initial_params is None:
            params = np.zeros(n)
        else:
            params = initial_params.copy()
        
        self.history = {'params': [], 'cost': []}
        
        for iteration in range(self.max_iterations):
            # Compute predictions
            predictions = X.dot(params)
            
            # Compute cost (Mean Squared Error)
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            
            # Compute gradient
            gradient = (1 / m) * X.T.dot(predictions - y)
            
            # Update parameters
            params = params - self.learning_rate * gradient
            
            # Store history
            self.history['params'].append(params.copy())
            self.history['cost'].append(cost)
            
            # Check convergence
            if iteration > 0:
                cost_diff = abs(self.history['cost'][-2] - cost)
                if cost_diff < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return params
    
    def stochastic_gradient_descent(self, X, y, initial_params=None, batch_size=1):
        """
        Stochastic Gradient Descent (SGD)
        Updates parameters using one or mini-batch of samples at a time
        
        Parameters:
        -----------
        batch_size : int
            Number of samples per batch (1 for pure SGD)
        """
        m, n = X.shape
        
        if initial_params is None:
            params = np.zeros(n)
        else:
            params = initial_params.copy()
        
        self.history = {'params': [], 'cost': []}
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Compute predictions
                predictions = X_batch.dot(params)
                
                # Compute gradient
                gradient = (1 / len(X_batch)) * X_batch.T.dot(predictions - y_batch)
                
                # Update parameters
                params = params - self.learning_rate * gradient
            
            # Compute full cost for tracking
            full_predictions = X.dot(params)
            cost = (1 / (2 * m)) * np.sum((full_predictions - y) ** 2)
            
            self.history['params'].append(params.copy())
            self.history['cost'].append(cost)
            
            if iteration > 0:
                cost_diff = abs(self.history['cost'][-2] - cost)
                if cost_diff < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return params
    
    def momentum_gradient_descent(self, X, y, initial_params=None, momentum=0.9):
        """
        Gradient Descent with Momentum
        Accelerates convergence and helps escape local minima
        
        Parameters:
        -----------
        momentum : float
            Momentum coefficient (typically 0.9)
        """
        m, n = X.shape
        
        if initial_params is None:
            params = np.zeros(n)
        else:
            params = initial_params.copy()
        
        velocity = np.zeros(n)
        self.history = {'params': [], 'cost': []}
        
        for iteration in range(self.max_iterations):
            # Compute predictions
            predictions = X.dot(params)
            
            # Compute cost
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            
            # Compute gradient
            gradient = (1 / m) * X.T.dot(predictions - y)
            
            # Update velocity
            velocity = momentum * velocity - self.learning_rate * gradient
            
            # Update parameters
            params = params + velocity
            
            self.history['params'].append(params.copy())
            self.history['cost'].append(cost)
            
            if iteration > 0:
                cost_diff = abs(self.history['cost'][-2] - cost)
                if cost_diff < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return params
    
    def adam_optimizer(self, X, y, initial_params=None, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam Optimizer (Adaptive Moment Estimation)
        Combines momentum and adaptive learning rates
        
        Parameters:
        -----------
        beta1 : float
            Exponential decay rate for first moment
        beta2 : float
            Exponential decay rate for second moment
        epsilon : float
            Small constant for numerical stability
        """
        m, n = X.shape
        
        if initial_params is None:
            params = np.zeros(n)
        else:
            params = initial_params.copy()
        
        # Initialize moment estimates
        m_t = np.zeros(n)  # First moment (mean)
        v_t = np.zeros(n)  # Second moment (variance)
        
        self.history = {'params': [], 'cost': []}
        
        for t in range(1, self.max_iterations + 1):
            # Compute predictions
            predictions = X.dot(params)
            
            # Compute cost
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            
            # Compute gradient
            gradient = (1 / m) * X.T.dot(predictions - y)
            
            # Update biased first moment estimate
            m_t = beta1 * m_t + (1 - beta1) * gradient
            
            # Update biased second moment estimate
            v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected moment estimates
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)
            
            # Update parameters
            params = params - self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
            
            self.history['params'].append(params.copy())
            self.history['cost'].append(cost)
            
            if t > 1:
                cost_diff = abs(self.history['cost'][-2] - cost)
                if cost_diff < self.tolerance:
                    print(f"Converged at iteration {t}")
                    break
        
        return params


def visualize_gradient_descent_1d():
    """Visualize Gradient Descent on a simple 1D function"""
    
    # Define function: f(x) = x^2 - 4x + 5
    def f(x):
        return x**2 - 4*x + 5
    
    def gradient_f(x):
        return 2*x - 4
    
    # Gradient descent
    x = 0.0  # Starting point
    learning_rate = 0.1
    history_x = [x]
    history_y = [f(x)]
    
    for _ in range(20):
        grad = gradient_f(x)
        x = x - learning_rate * grad
        history_x.append(x)
        history_y.append(f(x))
    
    # Plot
    x_range = np.linspace(-1, 5, 100)
    y_range = f(x_range)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Function and path
    plt.subplot(1, 2, 1)
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x² - 4x + 5')
    plt.plot(history_x, history_y, 'ro-', markersize=8, label='Gradient Descent Path')
    plt.plot(history_x[0], history_y[0], 'go', markersize=12, label='Start')
    plt.plot(history_x[-1], history_y[-1], 'r*', markersize=15, label='End')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Gradient Descent on 1D Function', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost vs iterations
    plt.subplot(1, 2, 2)
    plt.plot(range(len(history_y)), history_y, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost vs Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Minimum found at x = {history_x[-1]:.4f}")
    print(f"Minimum value f(x) = {history_y[-1]:.4f}")
    print(f"True minimum at x = 2, f(x) = 1")


def compare_optimizers():
    """Compare different gradient descent variants"""
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100) * 0.5
    
    # Add bias term
    X_b = np.c_[np.ones((100, 1)), X]
    
    # Test different optimizers
    optimizers = {
        'Vanilla GD': GradientDescent(learning_rate=0.1, max_iterations=100),
        'SGD': GradientDescent(learning_rate=0.1, max_iterations=100),
        'Momentum': GradientDescent(learning_rate=0.1, max_iterations=100),
        'Adam': GradientDescent(learning_rate=0.1, max_iterations=100)
    }
    
    results = {}
    
    # Vanilla GD
    print("Running Vanilla Gradient Descent...")
    results['Vanilla GD'] = optimizers['Vanilla GD'].vanilla_gradient_descent(X_b, y)
    
    # SGD
    print("Running Stochastic Gradient Descent...")
    results['SGD'] = optimizers['SGD'].stochastic_gradient_descent(X_b, y, batch_size=10)
    
    # Momentum
    print("Running Momentum Gradient Descent...")
    results['Momentum'] = optimizers['Momentum'].momentum_gradient_descent(X_b, y)
    
    # Adam
    print("Running Adam Optimizer...")
    results['Adam'] = optimizers['Adam'].adam_optimizer(X_b, y)
    
    # Plot comparison
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Cost convergence
    plt.subplot(1, 2, 1)
    for name, optimizer in optimizers.items():
        plt.plot(optimizer.history['cost'], linewidth=2, label=name)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.title('Cost Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Final predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.5, label='Data')
    
    colors = ['red', 'green', 'blue', 'orange']
    for (name, params), color in zip(results.items(), colors):
        X_test = np.linspace(0, 2, 100).reshape(-1, 1)
        X_test_b = np.c_[np.ones((100, 1)), X_test]
        y_pred = X_test_b.dot(params)
        plt.plot(X_test, y_pred, color=color, linewidth=2, label=f'{name}')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Final Model Predictions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n" + "="*60)
    print("FINAL PARAMETERS:")
    print("="*60)
    for name, params in results.items():
        print(f"{name:15s}: θ₀={params[0]:.4f}, θ₁={params[1]:.4f}")
    print(f"{'True values':15s}: θ₀=4.0000, θ₁=3.0000")
    print("="*60)


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("GRADIENT DESCENT DEMONSTRATIONS")
    print("="*60)
    
    print("\n1. Simple 1D Gradient Descent")
    print("-" * 60)
    visualize_gradient_descent_1d()
    
    print("\n\n2. Comparing Different Optimizers")
    print("-" * 60)
    compare_optimizers()
    
    print("\n✅ Demonstrations complete!")
    print("\nKey Takeaways:")
    print("  • Vanilla GD: Stable but can be slow")
    print("  • SGD: Faster updates, more noise")
    print("  • Momentum: Accelerates convergence")
    print("  • Adam: Adaptive learning rates, often best choice")