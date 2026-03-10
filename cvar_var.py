import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CVaRAnalysis:
    """
    Comprehensive CVaR (Expected Shortfall) Analysis Class
    """
    
    def __init__(self, returns):
        """
        Initialize with returns data
        
        Parameters:
        returns (array-like): Historical returns data
        """
        self.returns = np.array(returns)
        self.n = len(self.returns)
        
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk
        
        Parameters:
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        
        Returns:
        float: Value at Risk
        """
        return np.percentile(self.returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Parameters:
        confidence_level (float): Confidence level
        
        Returns:
        float: CVaR
        """
        var = self.calculate_var(confidence_level)
        # CVaR is the average of returns below VaR
        cvar = self.returns[self.returns <= var].mean()
        return cvar
    
    def calculate_cvar_historical(self, confidence_level=0.95):
        """
        Historical CVaR calculation
        """
        return self.calculate_cvar(confidence_level)
    
    def calculate_cvar_gaussian(self, confidence_level=0.95):
        """
        Gaussian parametric CVaR calculation
        
        Parameters:
        confidence_level (float): Confidence level
        
        Returns:
        float: Parametric CVaR assuming normal distribution
        """
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        z = stats.norm.ppf(1 - confidence_level)
        # Formula for CVaR under normal distribution
        cvar = mu - sigma * stats.norm.pdf(z) / (1 - confidence_level)
        return cvar
    
    def calculate_cvar_tdist(self, confidence_level=0.95):
        """
        CVaR using Student's t-distribution
        
        Parameters:
        confidence_level (float): Confidence level
        
        Returns:
        float: CVaR assuming t-distribution
        """
        from scipy.stats import t
        
        # Fit t-distribution parameters
        params = t.fit(self.returns)
        df, loc, scale = params
        
        # Calculate CVaR using t-distribution
        t_val = t.ppf(1 - confidence_level, df, loc, scale)
        # PDF at t_val
        pdf_val = t.pdf(t_val, df, loc, scale)
        # CVaR formula for t-distribution
        cvar = loc - scale * (df + t_val**2) / ((df - 1) * (1 - confidence_level)) * pdf_val
        
        return cvar
    
    def cvar_ratio(self, confidence_level=0.95):
        """Calculate CVaR ratio (return / CVaR)"""
        return self.returns.mean() / abs(self.calculate_cvar(confidence_level))

def get_returns_data(tickers, start_date, end_date):
    """
    Safely get returns data for one or multiple tickers
    
    Parameters:
    tickers (str or list): Single ticker or list of tickers
    start_date (str): Start date
    end_date (str): End date
    
    Returns:
    pd.DataFrame or np.array: Returns data
    """
    # Convert single ticker to list for uniform handling
    if isinstance(tickers, str):
        tickers = [tickers]
        single_ticker = True
    else:
        single_ticker = False
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Handle the data based on number of tickers
    if len(tickers) == 1:
        ticker = tickers[0]
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            # Try to get Adj Close, then Close
            if ('Adj Close', ticker) in data.columns:
                prices = data[('Adj Close', ticker)]
            elif ('Close', ticker) in data.columns:
                prices = data[('Close', ticker)]
                print(f"Using 'Close' instead of 'Adj Close' for {ticker}")
            else:
                # Find any price column
                for col in data.columns:
                    if 'Close' in col[0] or 'Price' in col[0]:
                        prices = data[col]
                        print(f"Using {col} for {ticker}")
                        break
                else:
                    raise ValueError(f"No price data found for {ticker}")
        else:
            # Single level columns
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
                print(f"Using 'Close' instead of 'Adj Close' for {ticker}")
            else:
                price_cols = [col for col in data.columns if 'Close' in col or 'Price' in col]
                if price_cols:
                    prices = data[price_cols[0]]
                    print(f"Using {price_cols[0]} for {ticker}")
                else:
                    raise ValueError(f"No price data found for {ticker}")
        
        returns = prices.pct_change().dropna()
        if single_ticker:
            return returns.values  # Return array for single ticker
        else:
            return pd.DataFrame({ticker: returns})
    
    else:
        # Multiple tickers
        returns_dict = {}
        for ticker in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ('Adj Close', ticker) in data.columns:
                        prices = data[('Adj Close', ticker)]
                    elif ('Close', ticker) in data.columns:
                        prices = data[('Close', ticker)]
                        print(f"Using 'Close' instead of 'Adj Close' for {ticker}")
                    else:
                        continue
                else:
                    if 'Adj Close' in data.columns:
                        # For multiple tickers with single level columns, Adj Close will have all tickers
                        prices = data['Adj Close'][ticker] if ticker in data['Adj Close'].columns else None
                    elif 'Close' in data.columns:
                        prices = data['Close'][ticker] if ticker in data['Close'].columns else None
                    else:
                        continue
                
                if prices is not None:
                    returns = prices.pct_change().dropna()
                    returns_dict[ticker] = returns
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            return returns_df
        else:
            raise ValueError("No valid data found for any ticker")

class PortfolioCVaR:
    """
    Portfolio CVaR Optimization and Analysis
    """
    
    def __init__(self, returns_df):
        """
        Initialize with returns DataFrame
        
        Parameters:
        returns_df (pd.DataFrame): Returns for multiple assets
        """
        self.returns = returns_df
        self.assets = returns_df.columns.tolist()
        self.n_assets = len(self.assets)
        
    def portfolio_cvar(self, weights, confidence_level=0.95):
        """
        Calculate CVaR for a portfolio given weights
        
        Parameters:
        weights (array-like): Portfolio weights
        confidence_level (float): Confidence level
        
        Returns:
        float: Portfolio CVaR
        """
        portfolio_returns = self.returns @ weights
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar
    
    def equal_weight_cvar(self, confidence_level=0.95):
        """Calculate CVaR for equal-weight portfolio"""
        weights = np.ones(self.n_assets) / self.n_assets
        return self.portfolio_cvar(weights, confidence_level)
    
    def individual_cvar(self, confidence_level=0.95):
        """Calculate CVaR for each individual asset"""
        results = {}
        for asset in self.assets:
            asset_returns = self.returns[asset].values
            cvar_analyzer = CVaRAnalysis(asset_returns)
            results[asset] = {
                'cvar': cvar_analyzer.calculate_cvar(confidence_level),
                'var': cvar_analyzer.calculate_var(confidence_level),
                'mean': asset_returns.mean()
            }
        return results
    
    def minimize_cvar(self, confidence_level=0.95):
        """
        Find weights that minimize portfolio CVaR
        
        Parameters:
        confidence_level (float): Confidence level
        
        Returns:
        dict: Optimal weights and metrics
        """
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Objective function
        def objective(weights):
            portfolio_returns = self.returns @ weights
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            return portfolio_returns[portfolio_returns <= var].mean()
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-9, 'disp': False})
        
        if result.success:
            optimal_weights = result.x
            # Round very small weights to zero
            optimal_weights = np.where(optimal_weights < 1e-6, 0, optimal_weights)
            optimal_weights = optimal_weights / optimal_weights.sum()  # Renormalize
            
            portfolio_return = np.sum(self.returns.mean() * optimal_weights)
            portfolio_risk = self.portfolio_cvar(optimal_weights, confidence_level)
            
            return {
                'weights': dict(zip(self.assets, optimal_weights)),
                'expected_return': portfolio_return,
                'cvar': portfolio_risk,
                'sharpe_ratio': portfolio_return / abs(portfolio_risk) if portfolio_risk != 0 else 0
            }
        else:
            print(f"Optimization failed: {result.message}")
            return None
    
    def cvar_efficient_frontier(self, confidence_level=0.95, points=20):
        """
        Calculate CVaR efficient frontier
        
        Parameters:
        confidence_level (float): Confidence level
        points (int): Number of points on frontier
        
        Returns:
        pd.DataFrame: Efficient frontier points
        """
        # Calculate min and max returns
        min_return = self.returns.mean().min()
        max_return = self.returns.mean().max()
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, points)
        
        frontier = []
        for target in target_returns:
            # Custom optimization for target return
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.returns.mean() * x) - target}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            x0 = np.array([1/self.n_assets] * self.n_assets)
            
            def objective(weights):
                portfolio_returns = self.returns @ weights
                var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                return portfolio_returns[portfolio_returns <= var].mean()
            
            result = minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'ftol': 1e-9, 'disp': False})
            
            if result.success:
                weights = result.x
                weights = np.where(weights < 1e-6, 0, weights)
                weights = weights / weights.sum()
                
                frontier.append({
                    'return': np.sum(self.returns.mean() * weights),
                    'cvar': self.portfolio_cvar(weights, confidence_level),
                    'weights': weights
                })
        
        return pd.DataFrame(frontier)

def main():
    """Main function to run CVaR analysis"""
    
    print("=" * 70)
    print("COMPREHENSIVE CVaR (EXPECTED SHORTFALL) ANALYSIS")
    print("=" * 70)
    
    # Part 1: Single Asset Analysis
    print("\n" + "=" * 70)
    print("PART 1: SINGLE ASSET ANALYSIS")
    print("=" * 70)
    
    # Get returns for single asset
    ticker = "SPY"
    print(f"\nAnalyzing {ticker}...")
    spy_returns = get_returns_data(ticker, "2020-01-01", "2024-01-01")
    
    if spy_returns is not None:
        # Create CVaR analyzer
        cvar_analyzer = CVaRAnalysis(spy_returns)
        
        # Calculate various metrics
        print(f"\n{ticker} Risk Metrics:")
        print("-" * 40)
        for cl in [0.90, 0.95, 0.99]:
            var = cvar_analyzer.calculate_var(cl)
            cvar = cvar_analyzer.calculate_cvar(cl)
            cvar_gauss = cvar_analyzer.calculate_cvar_gaussian(cl)
            cvar_t = cvar_analyzer.calculate_cvar_tdist(cl)
            
            print(f"\n{cl:.0%} Confidence Level:")
            print(f"  VaR: {var:.6f} ({var*100:.2f}%)")
            print(f"  Historical CVaR: {cvar:.6f} ({cvar*100:.2f}%)")
            print(f"  Gaussian CVaR:   {cvar_gauss:.6f} ({cvar_gauss*100:.2f}%)")
            print(f"  T-Distribution:   {cvar_t:.6f} ({cvar_t*100:.2f}%)")
        
        print(f"\nCVaR Ratio (95%): {cvar_analyzer.cvar_ratio(0.95):.4f}")
    
    # Part 2: Portfolio Analysis
    print("\n" + "=" * 70)
    print("PART 2: PORTFOLIO ANALYSIS")
    print("=" * 70)
    
    # Analyze multiple assets
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    print(f"\nAnalyzing portfolio: {tickers}")
    
    # Get returns for multiple assets
    portfolio_returns = get_returns_data(tickers, "2020-01-01", "2024-01-01")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Create portfolio CVaR analyzer
        portfolio = PortfolioCVaR(portfolio_returns)
        
        # Individual asset analysis
        print("\nIndividual Asset 95% CVaR:")
        print("-" * 40)
        individual = portfolio.individual_cvar(0.95)
        for asset, metrics in individual.items():
            print(f"{asset}:")
            print(f"  Mean Return: {metrics['mean']*100:.4f}%")
            print(f"  VaR (95%): {metrics['var']*100:.2f}%")
            print(f"  CVaR (95%): {metrics['cvar']*100:.2f}%")
        
        # Equal weight portfolio
        equal_cvar = portfolio.equal_weight_cvar(0.95)
        print(f"\nEqual Weight Portfolio 95% CVaR: {equal_cvar*100:.2f}%")
        
        # Minimum CVaR portfolio
        print("\n" + "=" * 70)
        print("OPTIMIZATION: MINIMUM CVaR PORTFOLIO")
        print("=" * 70)
        
        optimal = portfolio.minimize_cvar(0.95)
        if optimal:
            print(f"\nExpected Return: {optimal['expected_return']*100:.4f}%")
            print(f"CVaR (95%): {optimal['cvar']*100:.2f}%")
            print(f"Sharpe Ratio (using CVaR): {optimal['sharpe_ratio']:.4f}")
            print("\nOptimal Weights:")
            for asset, weight in optimal['weights'].items():
                if weight > 0:
                    print(f"  {asset}: {weight*100:.2f}%")
        
        # Calculate efficient frontier
        print("\n" + "=" * 70)
        print("CALCULATING CVaR EFFICIENT FRONTIER...")
        print("=" * 70)
        
        frontier = portfolio.cvar_efficient_frontier(0.95, 15)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Returns distribution comparison
        axes[0, 0].hist(portfolio_returns, bins=50, alpha=0.5, label=portfolio_returns.columns)
        axes[0, 0].set_title('Returns Distribution by Asset')
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual CVaR comparison
        assets = list(individual.keys())
        cvar_values = [individual[a]['cvar'] * 100 for a in assets]
        colors = ['red' if x < 0 else 'green' for x in cvar_values]
        axes[0, 1].bar(assets, cvar_values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Individual Asset 95% CVaR')
        axes[0, 1].set_xlabel('Asset')
        axes[0, 1].set_ylabel('CVaR (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Efficient frontier
        if not frontier.empty:
            axes[1, 0].scatter(frontier['cvar'] * 100, frontier['return'] * 100, 
                              c='blue', s=50, alpha=0.6)
            axes[1, 0].set_xlabel('CVaR (95%) %')
            axes[1, 0].set_ylabel('Expected Return %')
            axes[1, 0].set_title('CVaR Efficient Frontier')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Mark equal weight and min CVaR portfolios
            axes[1, 0].scatter(equal_cvar * 100, portfolio_returns.mean().mean() * 100,
                              color='red', s=200, marker='*', label='Equal Weight', zorder=5)
            if optimal:
                axes[1, 0].scatter(optimal['cvar'] * 100, optimal['expected_return'] * 100,
                                  color='green', s=200, marker='*', label='Min CVaR', zorder=5)
            axes[1, 0].legend()
        
        # Plot 4: Optimal portfolio weights
        if optimal:
            weights = [optimal['weights'][a] for a in assets if optimal['weights'][a] > 0]
            labels = [a for a in assets if optimal['weights'][a] > 0]
            if weights:
                axes[1, 1].pie(weights, labels=labels, autopct='%1.1f%%')
                axes[1, 1].set_title('Minimum CVaR Portfolio Weights')
            else:
                axes[1, 1].text(0.5, 0.5, 'No positive weights', ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("PORTFOLIO SUMMARY STATISTICS")
        print("=" * 70)
        
        summary = pd.DataFrame({
            'Mean Return %': portfolio_returns.mean() * 100,
            'Std Dev %': portfolio_returns.std() * 100,
            'CVaR 95% %': [individual[a]['cvar'] * 100 for a in assets],
            'VaR 95% %': [individual[a]['var'] * 100 for a in assets]
        }, index=assets)
        
        print("\n", summary.round(4))
        
        # Correlation matrix
        print("\nCorrelation Matrix:")
        print(portfolio_returns.corr().round(4))

if __name__ == "__main__":
    main()                 