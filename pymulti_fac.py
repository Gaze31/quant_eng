"""
multi_factor_model.py

Models:
 - CAPM
 - Fama-French 3-Factor (MKT, SMB, HML)
 - Carhart 4-Factor (adds Momentum MOM)
 - Generalized multi-factor regression (N factors)

Dependencies:
    pandas, numpy, statsmodels, matplotlib (optional)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import Dict, List


def factor_model(asset_returns: pd.Series, factor_data: pd.DataFrame, rf: pd.Series = None) -> Dict:
    """
    Generic multi-factor regression model.

    asset_returns: pd.Series (daily/monthly log or pct returns)
    factor_data: DataFrame with columns = available factors
    rf (optional): risk-free rate series -> computes excess returns

    Returns:
        alpha, betas(each factor), t-stats, p-values, R², summary
    """

    # Convert to excess return if risk-free provided
    if rf is not None:
        y = asset_returns - rf
        X = factor_data.sub(rf.values, axis=0, fill_value=0) if "MKT" in factor_data.columns else factor_data
    else:
        y = asset_returns
        X = factor_data

    X = sm.add_constant(X)  # add intercept/alpha
    model = sm.OLS(y, X).fit()

    return {
        "alpha": float(model.params[0]),
        "betas": model.params[1:].to_dict(),
        "t_values": model.tvalues.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "r_squared": float(model.rsquared),
        "summary": model.summary()
    }


# --------------------------------------------------------------
# Helper Functions for Annualized Metrics
# --------------------------------------------------------------

def annualize_alpha(alpha_periodic: float, periods: int = 12) -> float:
    """Convert periodic alpha to annualized."""
    return (1 + alpha_periodic) ** periods - 1


def plot_factor_loadings(result: Dict, title="Factor Loadings"):
    """Visualize betas for factors."""
    betas = list(result["betas"].values())
    labels = list(result["betas"].keys())
    
    plt.bar(labels, betas)
    plt.title(title)
    plt.ylabel("Beta Loading")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------
# Example Usage with Fama-French & Carhart
# --------------------------------------------------------------
if __name__ == "__main__":
    # Example Synthetic Factor Data (replace later with FRED/WRDS datasets)
    np.random.seed(42)
    n = 60  # 60 months

    # Factors ~ Normal distributions
    MKT = np.random.normal(0.007, 0.04, n)  # market factor
    SMB = np.random.normal(0.002, 0.03, n)  # size premium
    HML = np.random.normal(0.001, 0.025, n) # value premium
    MOM = np.random.normal(0.003, 0.03, n)  # momentum factor

    factor_df_3F = pd.DataFrame({"MKT": MKT, "SMB": SMB, "HML": HML})
    factor_df_4F = pd.DataFrame({"MKT": MKT, "SMB": SMB, "HML": HML, "MOM": MOM})

    # Create Asset dependent on factors (true betas)
    beta_true = [1.2, -0.5, 0.8, 0.6]
    noise = np.random.normal(0, 0.03, n)
    asset_ret = 0.002 + (beta_true[0]*MKT + beta_true[1]*SMB + beta_true[2]*HML + beta_true[3]*MOM) + noise
    
    asset_returns = pd.Series(asset_ret)

    # Run Fama-French 3F
    ff3 = factor_model(asset_returns, factor_df_3F)
    print("\n===== Fama–French 3-Factor Output =====")
    print(ff3["summary"])

    plot_factor_loadings(ff3, "FF-3 Factor Loadings")

    # Run Carhart 4F
    carhart = factor_model(asset_returns, factor_df_4F)
    print("\n===== Carhart 4-Factor Output =====")
    print(carhart["summary"])

    plot_factor_loadings(carhart, "Carhart 4-Factor Loadings")
