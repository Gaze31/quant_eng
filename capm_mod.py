"""
capm.py

Simple CAPM utilities:
 - estimate_beta_ols: beta via OLS regression of asset excess returns on market excess returns
 - estimate_beta_cov: beta via covariance / variance
 - capm_expected_return: E[R] = Rf + beta * (E[Rm] - Rf)
 - jensen_alpha: alpha = actual_mean_excess - beta * market_mean_excess
 - plot_sml: plot security market line and asset point

Dependencies:
 - numpy
 - pandas
 - statsmodels (for OLS regression)
 - matplotlib
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def excess_returns(returns: pd.Series, risk_free_rate: float) -> pd.Series:
    """
    Convert a returns series (decimal, e.g. 0.01 = 1%) into excess returns over a constant risk-free rate.
    Assumes rf is per-period and aligned with returns' frequency.
    """
    return returns - risk_free_rate


def estimate_beta_ols(asset_excess: pd.Series, market_excess: pd.Series) -> Dict:
    """
    Estimate CAPM beta using OLS regression:
       asset_excess = alpha + beta * market_excess + epsilon

    Returns dict with beta, alpha, r2, stderr_beta, t_beta, p_beta, nobs, model (statsmodels result)
    """
    if len(asset_excess) != len(market_excess):
        raise ValueError("asset_excess and market_excess must be same length and aligned")

    X = sm.add_constant(market_excess.values)  # adds intercept
    y = asset_excess.values
    model = sm.OLS(y, X).fit()
    alpha, beta = model.params[0], model.params[1]
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "r2": float(model.rsquared),
        "stderr_beta": float(model.bse[1]),
        "t_beta": float(model.tvalues[1]),
        "p_beta": float(model.pvalues[1]),
        "nobs": int(model.nobs),
        "model": model
    }


def estimate_beta_cov(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Estimate beta by covariance method using raw returns (not excess), beta = cov(Ri, Rm) / var(Rm)
    The user should be consistent about returns (raw vs excess) when applying CAPM formula.
    """
    cov = np.cov(asset_returns, market_returns, ddof=1)[0, 1]
    var_m = np.var(market_returns, ddof=1)
    if var_m == 0:
        raise ValueError("Market variance is zero")
    return float(cov / var_m)


def capm_expected_return(risk_free_rate: float, beta: float, market_expected_return: float) -> float:
    """
    Compute expected return from CAPM:
       E[R_i] = R_f + beta * (E[R_m] - R_f)
    All inputs in decimal (e.g., 0.02 = 2%)
    """
    return risk_free_rate + beta * (market_expected_return - risk_free_rate)


def jensen_alpha(asset_mean_return: float, expected_return_capm: float) -> float:
    """
    Jensen's alpha = Actual mean return - CAPM expected return
    """
    return asset_mean_return - expected_return_capm


def annualize_return_from_periodic(mean_periodic_return: float, periods_per_year: int) -> float:
    """
    Convert mean periodic return (arithmetic mean) to approximate annual return via geometric approx:
      (1 + r_periodic) ** periods_per_year - 1
    For small returns arithmetic * periods_per_year is often used as approximation; geometric is safer.
    """
    return (1 + mean_periodic_return) ** periods_per_year - 1


def plot_sml(risk_free_rate: float, market_expected_return: float, betas: np.ndarray,
             asset_expected_returns: np.ndarray = None, asset_labels: list = None):
    """
    Plot Security Market Line (SML) and optionally plot asset(s) given their betas and expected returns.
    betas: array-like of betas to use for drawing markers
    asset_expected_returns: if provided, must be same length as betas (observed expected returns)
    """
    b_min, b_max = np.min(betas) - 0.2, np.max(betas) + 0.2
    x = np.linspace(b_min, b_max, 200)
    sml = risk_free_rate + x * (market_expected_return - risk_free_rate)

    plt.figure(figsize=(8, 6))
    plt.plot(x, sml, label="Security Market Line (SML)")
    plt.scatter(betas, asset_expected_returns if asset_expected_returns is not None else risk_free_rate + betas * (market_expected_return - risk_free_rate),
                marker='o', label='Assets')
    if asset_labels:
        for i, lab in enumerate(asset_labels):
            plt.annotate(lab, (betas[i], asset_expected_returns[i] if asset_expected_returns is not None else risk_free_rate + betas[i] * (market_expected_return - risk_free_rate)),
                         xytext=(5, -5), textcoords='offset points')
    plt.xlabel("Beta")
    plt.ylabel("Expected Return")
    plt.title("Security Market Line")
    plt.grid(True)
    plt.legend()
    plt.show()


# --------------------------
# Example / demo
# --------------------------
if __name__ == "__main__":
    # Example with synthetic monthly returns (decimal, e.g. 0.01 = 1%)
    np.random.seed(42)
    n = 60  # 60 months (~5 years)
    market_mean = 0.007  # monthly mean ~0.7%
    market_sigma = 0.04
    market_returns = pd.Series(np.random.normal(loc=market_mean, scale=market_sigma, size=n))

    # Asset with beta 1.3 plus some idiosyncratic noise
    true_beta = 1.3
    asset_alpha_monthly = 0.001  # monthly alpha
    asset_returns = pd.Series(asset_alpha_monthly + true_beta * market_returns + np.random.normal(0, 0.03, size=n))

    # Risk-free rate (monthly). If annual RF is 2% -> monthly ~ (1+0.02)**(1/12)-1
    rf_annual = 0.02
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1

    # Compute excess returns
    asset_exc = excess_returns(asset_returns, rf_monthly)
    market_exc = excess_returns(market_returns, rf_monthly)

    # Estimate beta by OLS on excess returns
    res = estimate_beta_ols(asset_exc, market_exc)
    print("OLS Regression results (on excess returns):")
    print(f"  alpha (monthly excess): {res['alpha']:.6f}")
    print(f"  beta: {res['beta']:.4f}")
    print(f"  R^2: {res['r2']:.4f}")
    print()

    # Beta via covariance method using raw returns
    beta_cov = estimate_beta_cov(asset_returns, market_returns)
    print(f"Beta (covariance method, raw returns): {beta_cov:.4f}")

    # Market expected return (use sample mean monthly return here)
    market_exp_monthly = market_returns.mean()
    asset_mean_monthly = asset_returns.mean()

    # CAPM expected return (monthly)
    expected_asset_monthly = capm_expected_return(rf_monthly, res['beta'], market_exp_monthly)
    print(f"CAPM expected return (monthly): {expected_asset_monthly:.6f}")
    print(f"Actual mean asset return (monthly): {asset_mean_monthly:.6f}")

    # Jensen alpha (monthly)
    alpha_jensen = jensen_alpha(asset_mean_monthly, expected_asset_monthly)
    print(f"Jensen's alpha (monthly): {alpha_jensen:.6f}")

    # Annualize (approx)
    periods_per_year = 12
    print()
    print("Annualized approximations:")
    print(f"  Annualized CAPM expected return: {annualize_return_from_periodic(expected_asset_monthly, periods_per_year):.4%}")
    print(f"  Annualized actual asset mean return: {annualize_return_from_periodic(asset_mean_monthly, periods_per_year):.4%}")

    # Plot SML with the estimated beta and actual asset mean return
    plot_sml(rf_monthly, market_exp_monthly, betas=np.array([res['beta']]), asset_expected_returns=np.array([asset_mean_monthly]),
             asset_labels=["MyAsset"])
