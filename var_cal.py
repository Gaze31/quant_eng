import numpy as np
import pandas as pd
from scipy import stats

def historical_var(returns, confidence=0.95):
    """
    Historical VaR: percentile of past returns.
    returns: pd.Series of daily/periodic returns
    confidence: confidence level (e.g., 0.95 for 95%)
    Returns: negative value (loss threshold)
    """
    return np.percentile(returns, (1 - confidence) * 100)

def parametric_var(returns, confidence=0.95):
    """
    Parametric VaR (assumes normal distribution).
    Returns: negative value (loss threshold)
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return mu + z * sigma

def monte_carlo_var(returns, confidence=0.95, n_sims=10000, steps=1):
    """
    Monte Carlo VaR: simulate future returns using drift & volatility.
    returns: pd.Series of historical returns
    steps: forecast horizon (e.g., 1 day, 10 days)
    Returns: negative value (loss threshold)
    """
    mu = returns.mean()
    sigma = returns.std()
    # simulate n_sims paths, each of length steps
    sims = np.random.normal(mu, sigma, size=(n_sims, steps))
    # cumulative returns over steps
    terminal_returns = sims.sum(axis=1)
    return np.percentile(terminal_returns, (1 - confidence) * 100)

def expected_shortfall(returns, confidence=0.95, method='historical'):
    """
    Expected Shortfall (CVaR): average loss beyond VaR.
    """
    if method == 'historical':
        var = historical_var(returns, confidence)
        return returns[returns <= var].mean()
    elif method == 'parametric':
        mu = returns.mean()
        sigma = returns.std()
        z = stats.norm.ppf(1 - confidence)
        return mu - sigma * stats.norm.pdf(z) / (1 - confidence)
    return None

def var_summary(returns, confidence=0.95, portfolio_value=1e6):
    """
    Print VaR summary in dollar terms.
    portfolio_value: notional value of portfolio
    """
    var_hist = historical_var(returns, confidence)
    var_param = parametric_var(returns, confidence)
    es_hist = expected_shortfall(returns, confidence, 'historical')
    
    dollar_loss_hist = abs(var_hist) * portfolio_value
    dollar_loss_param = abs(var_param) * portfolio_value
    dollar_es = abs(es_hist) * portfolio_value
    
    print(f"\n=== VaR Analysis (confidence={confidence*100}%) ===")
    print(f"Historical VaR:  {var_hist:.4f} ({-dollar_loss_hist:,.0f} USD loss)")
    print(f"Parametric VaR:  {var_param:.4f} ({-dollar_loss_param:,.0f} USD loss)")
    print(f"Expected Shortfall (ES/CVaR): {es_hist:.4f} ({-dollar_es:,.0f} USD loss)")
    
    return {
        'var_historical': var_hist,
        'var_parametric': var_param,
        'expected_shortfall': es_hist,
        'dollar_loss_hist': dollar_loss_hist,
        'dollar_loss_param': dollar_loss_param,
        'dollar_es': dollar_es
    }
import pandas as pd
import numpy as np
# Use local definitions of var_summary, historical_var, parametric_var, monte_carlo_var

def main():
    # generate synthetic daily returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, size=252))  # 1 year of daily returns
    
    print("Daily returns summary:")
    print(f"Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    print(f"Min: {returns.min():.4f}, Max: {returns.max():.4f}")
    
    # VaR at 95% and 99% confidence
    for conf in [0.95, 0.99]:
        var_summary(returns, confidence=conf, portfolio_value=1e6)
    
    # Monte Carlo
    mc_var = monte_carlo_var(returns, confidence=0.95, n_sims=5000, steps=10)
    print(f"\nMonte Carlo VaR (10-day, 95%): {mc_var:.4f}")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
# Use local definitions of historical_var, parametric_var, expected_shortfall

def test_var_historical():
    returns = pd.Series(np.random.normal(0, 0.01, size=1000))
    var_95 = historical_var(returns, confidence=0.95)
    assert var_95 < 0  # VaR should be negative (loss)
    assert var_95 > returns.min()  # should be less extreme than worst case

def test_var_parametric():
    returns = pd.Series(np.random.normal(0, 0.01, size=1000))
    var_95 = parametric_var(returns, confidence=0.95)
    assert var_95 < 0

def test_expected_shortfall():
    returns = pd.Series(np.random.normal(0, 0.01, size=1000))
    es = expected_shortfall(returns, confidence=0.95, method='historical')
    assert es < 0
    assert es < historical_var(returns, confidence=0.95)  # ES should exceed VaR   