import numpy as np
import pandas as pd
from scipy import stats

class RiskMetrics:
    """
    A class to calculate various financial risk metrics for portfolio analysis.
    """
    
    def __init__(self, returns, confidence_level=0.95):
        """
        Initialize with return data.
        
        Parameters:
        -----------
        returns : array-like
            Historical returns (can be daily, monthly, etc.)
        confidence_level : float
            Confidence level for VaR and CVaR calculations (default: 0.95)
        """
        self.returns = np.array(returns)
        self.confidence_level = confidence_level
        
    def volatility(self, annualize=True, periods=252):
        """
        Calculate volatility (standard deviation of returns).
        
        Parameters:
        -----------
        annualize : bool
            Whether to annualize the volatility
        periods : int
            Number of periods per year (252 for daily, 12 for monthly)
        """
        vol = np.std(self.returns, ddof=1)
        if annualize:
            vol = vol * np.sqrt(periods)
        return vol
    
    def value_at_risk(self, method='historical'):
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        -----------
        method : str
            'historical' or 'parametric'
        """
        if method == 'historical':
            var = np.percentile(self.returns, (1 - self.confidence_level) * 100)
        elif method == 'parametric':
            mu = np.mean(self.returns)
            sigma = np.std(self.returns, ddof=1)
            var = stats.norm.ppf(1 - self.confidence_level, mu, sigma)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
        
        return var
    
    def conditional_var(self):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        Returns the expected loss given that VaR has been exceeded.
        """
        var = self.value_at_risk(method='historical')
        cvar = self.returns[self.returns <= var].mean()
        return cvar
    
    def sharpe_ratio(self, risk_free_rate=0.02, periods=252):
        """
        Calculate Sharpe Ratio.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
        periods : int
            Number of periods per year
        """
        excess_returns = self.returns - (risk_free_rate / periods)
        return np.sqrt(periods) * excess_returns.mean() / excess_returns.std(ddof=1)
    
    def sortino_ratio(self, risk_free_rate=0.02, periods=252):
        """
        Calculate Sortino Ratio (focuses only on downside volatility).
        """
        excess_returns = self.returns - (risk_free_rate / periods)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(periods) * excess_returns.mean() / downside_std
    
    def maximum_drawdown(self):
        """
        Calculate Maximum Drawdown.
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def beta(self, market_returns):
        """
        Calculate Beta relative to market returns.
        
        Parameters:
        -----------
        market_returns : array-like
            Market benchmark returns
        """
        covariance = np.cov(self.returns, market_returns)[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        return covariance / market_variance
    
    def tracking_error(self, benchmark_returns, annualize=True, periods=252):
        """
        Calculate Tracking Error relative to a benchmark.
        """
        diff = self.returns - np.array(benchmark_returns)
        te = np.std(diff, ddof=1)
        if annualize:
            te = te * np.sqrt(periods)
        return te
    
    def information_ratio(self, benchmark_returns, periods=252):
        """
        Calculate Information Ratio.
        """
        excess_returns = self.returns - np.array(benchmark_returns)
        return np.sqrt(periods) * excess_returns.mean() / excess_returns.std(ddof=1)
    
    def get_all_metrics(self, market_returns=None, benchmark_returns=None):
        """
        Calculate and return all risk metrics as a dictionary.
        """
        metrics = {
            'Volatility (Annual)': self.volatility(),
            'VaR (Historical)': self.value_at_risk('historical'),
            'VaR (Parametric)': self.value_at_risk('parametric'),
            'CVaR': self.conditional_var(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Maximum Drawdown': self.maximum_drawdown()
        }
        
        if market_returns is not None:
            metrics['Beta'] = self.beta(market_returns)
        
        if benchmark_returns is not None:
            metrics['Tracking Error'] = self.tracking_error(benchmark_returns)
            metrics['Information Ratio'] = self.information_ratio(benchmark_returns)
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Generate sample return data
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.0005, 0.01, 252)  # Daily returns
    market_returns = np.random.normal(0.0004, 0.012, 252)
    
    # Initialize risk metrics calculator
    risk = RiskMetrics(portfolio_returns, confidence_level=0.95)
    
    # Calculate individual metrics
    print("Individual Risk Metrics:")
    print(f"Annual Volatility: {risk.volatility():.4f}")
    print(f"VaR (95%, Historical): {risk.value_at_risk('historical'):.4f}")
    print(f"CVaR (95%): {risk.conditional_var():.4f}")
    print(f"Sharpe Ratio: {risk.sharpe_ratio():.4f}")
    print(f"Maximum Drawdown: {risk.maximum_drawdown():.4f}")
    print(f"Beta: {risk.beta(market_returns):.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Get all metrics at once
    all_metrics = risk.get_all_metrics(market_returns=market_returns)
    print("All Risk Metrics:")
    for metric, value in all_metrics.items():
        print(f"{metric}: {value:.4f}")