import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: FINANCIAL METRICS
# ============================================================================

class FinancialMetrics:
    """
    Comprehensive financial metrics for investment analysis
    """
    
    @staticmethod
    def calculate_returns(prices, method='log'):
        """
        Calculate returns from price series
        
        Parameters:
        -----------
        prices : array-like
            Price series
        method : str
            'simple' or 'log' returns
            
        Returns:
        --------
        returns : array
            Return series
        """
        prices = np.array(prices)
        
        if method == 'simple':
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
        elif method == 'log':
            returns = np.log(prices[1:] / prices[:-1])
        else:
            raise ValueError("Method must be 'simple' or 'log'")
            
        return returns
    
    @staticmethod
    def annualize_return(period_returns, periods_per_year=252):
        """
        Annualize returns
        
        Parameters:
        -----------
        period_returns : float or array
            Period returns
        periods_per_year : int
            Number of periods in a year (252 for daily, 12 for monthly, 4 for quarterly)
            
        Returns:
        --------
        annual_return : float
            Annualized return
        """
        if isinstance(period_returns, (list, np.ndarray)):
            total_return = (1 + np.mean(period_returns)) ** periods_per_year - 1
        else:
            total_return = (1 + period_returns) ** periods_per_year - 1
            
        return total_return
    
    @staticmethod
    def calculate_volatility(returns, annualize=True, periods_per_year=252):
        """
        Calculate volatility (standard deviation of returns)
        
        Parameters:
        -----------
        returns : array-like
            Return series
        annualize : bool
            Whether to annualize volatility
        periods_per_year : int
            Number of periods in a year
            
        Returns:
        --------
        volatility : float
            Volatility
        """
        returns = np.array(returns)
        vol = np.std(returns, ddof=1)
        
        if annualize:
            vol = vol * np.sqrt(periods_per_year)
            
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate Sharpe ratio
        
        Parameters:
        -----------
        returns : array-like
            Return series
        risk_free_rate : float
            Annual risk-free rate
        periods_per_year : int
            Number of periods in a year
            
        Returns:
        --------
        sharpe : float
            Sharpe ratio
        """
        returns = np.array(returns)
        
        # Convert risk-free rate to period rate
        period_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        
        excess_returns = returns - period_rf
        sharpe = np.mean(excess_returns) / np.std(returns, ddof=1)
        sharpe_annualized = sharpe * np.sqrt(periods_per_year)
        
        return sharpe_annualized
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0, periods_per_year=252):
        """
        Calculate Sortino ratio (uses downside deviation)
        
        Parameters:
        -----------
        returns : array-like
            Return series
        risk_free_rate : float
            Annual risk-free rate
        target_return : float
            Minimum acceptable return
        periods_per_year : int
            
        Returns:
        --------
        sortino : float
            Sortino ratio
        """
        returns = np.array(returns)
        
        # Convert risk-free rate to period rate
        period_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        
        excess_returns = np.mean(returns) - period_rf
        
        # Calculate downside deviation
        downside = returns[returns < target_return]
        if len(downside) == 0:
            downside_deviation = 0
        else:
            downside_deviation = np.std(downside, ddof=1)
        
        if downside_deviation == 0:
            return np.inf if excess_returns > 0 else -np.inf
        
        sortino = excess_returns / downside_deviation
        sortino_annualized = sortino * np.sqrt(periods_per_year)
        
        return sortino_annualized
    
    @staticmethod
    def calculate_calmar_ratio(returns, periods_per_year=252):
        """
        Calculate Calmar ratio (return / max drawdown)
        
        Parameters:
        -----------
        returns : array-like
            Return series
        periods_per_year : int
            
        Returns:
        --------
        calmar : float
            Calmar ratio
        """
        returns = np.array(returns)
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate max drawdown
        max_drawdown = FinancialMetrics.calculate_max_drawdown(cum_returns)
        
        # Calculate annualized return
        total_return = cum_returns[-1] - 1
        years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        if max_drawdown == 0:
            return np.inf if annualized_return > 0 else -np.inf
            
        calmar = annualized_return / abs(max_drawdown)
        
        return calmar
    
    @staticmethod
    def calculate_max_drawdown(prices_or_returns, is_prices=True):
        """
        Calculate maximum drawdown
        
        Parameters:
        -----------
        prices_or_returns : array-like
            Price series or return series
        is_prices : bool
            True if input is prices, False if returns
            
        Returns:
        --------
        max_dd : float
            Maximum drawdown (positive number)
        """
        if not is_prices:
            # Convert returns to prices
            prices = (1 + np.array(prices_or_returns)).cumprod()
        else:
            prices = np.array(prices_or_returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        -----------
        returns : array-like
            Return series
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        method : str
            'historical', 'parametric', or 'cornish_fisher'
            
        Returns:
        --------
        var : float
            Value at Risk
        """
        returns = np.array(returns)
        
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            var = stats.norm.ppf(1 - confidence_level, mean, std)
            
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * (kurt - 3) / 24 - (2*z**3 - 5*z) * (skew**2) / 36
            var = mean + z_cf * std
            
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'cornish_fisher'")
            
        return var
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Parameters:
        -----------
        returns : array-like
            Return series
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        cvar : float
            Conditional Value at Risk
        """
        returns = np.array(returns)
        var = FinancialMetrics.calculate_var(returns, confidence_level, method='historical')
        cvar = np.mean(returns[returns <= var])
        
        return cvar
    
    @staticmethod
    def calculate_beta(asset_returns, benchmark_returns):
        """
        Calculate beta (systematic risk)
        
        Parameters:
        -----------
        asset_returns : array-like
            Asset return series
        benchmark_returns : array-like
            Benchmark return series
            
        Returns:
        --------
        beta : float
            Beta coefficient
        """
        asset_returns = np.array(asset_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns, ddof=1)
        
        beta = covariance / variance
        
        return beta
    
    @staticmethod
    def calculate_alpha(asset_returns, benchmark_returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate Jensen's Alpha
        
        Parameters:
        -----------
        asset_returns : array-like
            Asset return series
        benchmark_returns : array-like
            Benchmark return series
        risk_free_rate : float
            Annual risk-free rate
        periods_per_year : int
            
        Returns:
        --------
        alpha : float
            Jensen's Alpha
        """
        asset_returns = np.array(asset_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Convert risk-free rate to period rate
        period_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        
        # Calculate beta
        beta = FinancialMetrics.calculate_beta(asset_returns, benchmark_returns)
        
        # Expected return based on CAPM
        expected_return = period_rf + beta * (np.mean(benchmark_returns) - period_rf)
        
        # Alpha
        alpha = np.mean(asset_returns) - expected_return
        alpha_annualized = alpha * periods_per_year
        
        return alpha_annualized
    
    @staticmethod
    def calculate_treynor_ratio(asset_returns, benchmark_returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate Treynor ratio
        
        Parameters:
        -----------
        asset_returns : array-like
            Asset return series
        benchmark_returns : array-like
            Benchmark return series
        risk_free_rate : float
            Annual risk-free rate
        periods_per_year : int
            
        Returns:
        --------
        treynor : float
            Treynor ratio
        """
        asset_returns = np.array(asset_returns)
        
        # Convert risk-free rate to period rate
        period_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        
        # Calculate beta
        beta = FinancialMetrics.calculate_beta(asset_returns, benchmark_returns)
        
        # Excess return
        excess_return = np.mean(asset_returns) - period_rf
        
        if beta == 0:
            return np.inf if excess_return > 0 else -np.inf
            
        treynor = excess_return / beta
        treynor_annualized = treynor * periods_per_year
        
        return treynor_annualized
    
    @staticmethod
    def calculate_information_ratio(asset_returns, benchmark_returns):
        """
        Calculate Information ratio
        
        Parameters:
        -----------
        asset_returns : array-like
            Asset return series
        benchmark_returns : array-like
            Benchmark return series
            
        Returns:
        --------
        ir : float
            Information ratio
        """
        asset_returns = np.array(asset_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Active returns
        active_returns = asset_returns - benchmark_returns
        
        # Information ratio
        ir = np.mean(active_returns) / np.std(active_returns, ddof=1)
        
        return ir
    
    @staticmethod
    def calculate_omega_ratio(returns, threshold=0):
        """
        Calculate Omega ratio
        
        Parameters:
        -----------
        returns : array-like
            Return series
        threshold : float
            Return threshold
            
        Returns:
        --------
        omega : float
            Omega ratio
        """
        returns = np.array(returns)
        
        # Gains above threshold
        gains = returns[returns > threshold] - threshold
        
        # Losses below threshold
        losses = threshold - returns[returns < threshold]
        
        if len(losses) == 0:
            return np.inf
            
        omega = np.sum(gains) / np.sum(losses)
        
        return omega


# ============================================================================
# PART 2: RISK METRICS
# ============================================================================

class RiskMetrics:
    """
    Advanced risk metrics for portfolio and credit risk
    """
    
    @staticmethod
    def calculate_correlation_matrix(returns_df):
        """
        Calculate correlation matrix
        
        Parameters:
        -----------
        returns_df : DataFrame
            Returns for multiple assets
            
        Returns:
        --------
        corr_matrix : DataFrame
            Correlation matrix
        """
        return returns_df.corr()
    
    @staticmethod
    def calculate_covariance_matrix(returns_df):
        """
        Calculate covariance matrix
        
        Parameters:
        -----------
        returns_df : DataFrame
            Returns for multiple assets
            
        Returns:
        --------
        cov_matrix : DataFrame
            Covariance matrix
        """
        return returns_df.cov()
    
    @staticmethod
    def portfolio_variance(weights, cov_matrix):
        """
        Calculate portfolio variance
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        cov_matrix : array-like
            Covariance matrix
            
        Returns:
        --------
        variance : float
            Portfolio variance
        """
        weights = np.array(weights)
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        return portfolio_var
    
    @staticmethod
    def portfolio_volatility(weights, cov_matrix):
        """
        Calculate portfolio volatility
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        cov_matrix : array-like
            Covariance matrix
            
        Returns:
        --------
        volatility : float
            Portfolio volatility
        """
        return np.sqrt(RiskMetrics.portfolio_variance(weights, cov_matrix))
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns, benchmark_returns):
        """
        Calculate tracking error
        
        Parameters:
        -----------
        portfolio_returns : array-like
            Portfolio return series
        benchmark_returns : array-like
            Benchmark return series
            
        Returns:
        --------
        te : float
            Tracking error
        """
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        
        return tracking_error
    
    @staticmethod
    def calculate_beta_hedge_ratio(asset_returns, hedge_returns):
        """
        Calculate hedge ratio for beta hedging
        
        Parameters:
        -----------
        asset_returns : array-like
            Asset return series
        hedge_returns : array-like
            Hedge instrument return series
            
        Returns:
        --------
        hedge_ratio : float
            Optimal hedge ratio
        """
        asset_returns = np.array(asset_returns)
        hedge_returns = np.array(hedge_returns)
        
        # Simple linear regression
        cov = np.cov(asset_returns, hedge_returns)[0, 1]
        var = np.var(hedge_returns, ddof=1)
        
        hedge_ratio = cov / var
        
        return hedge_ratio
    
    @staticmethod
    def calculate_var_contribution(weights, var, marginal_var):
        """
        Calculate component VaR
        
        Parameters:
        -----------
        weights : array-like
            Portfolio weights
        var : float
            Portfolio VaR
        marginal_var : array-like
            Marginal VaR for each asset
            
        Returns:
        --------
        component_var : array
            Component VaR
        """
        weights = np.array(weights)
        marginal_var = np.array(marginal_var)
        
        component_var = weights * marginal_var * var
        
        return component_var
    
    @staticmethod
    def stress_test(returns, scenarios):
        """
        Perform stress testing
        
        Parameters:
        -----------
        returns : array-like
            Return series
        scenarios : dict
            Stress scenarios with multipliers
            
        Returns:
        --------
        results : dict
            Stress test results
        """
        returns = np.array(returns)
        results = {}
        
        for scenario_name, multiplier in scenarios.items():
            stressed_returns = returns * multiplier
            
            results[scenario_name] = {
                'mean_return': np.mean(stressed_returns),
                'volatility': np.std(stressed_returns, ddof=1),
                'var_95': FinancialMetrics.calculate_var(stressed_returns, 0.95),
                'max_drawdown': FinancialMetrics.calculate_max_drawdown(stressed_returns, is_prices=False)
            }
            
        return results


# ============================================================================
# PART 3: CREDIT METRICS
# ============================================================================

class CreditMetrics:
    """
    Credit risk metrics including KMV, Altman Z-Score, etc.
    """
    
    @staticmethod
    def calculate_altman_z_score(working_capital, retained_earnings, ebit, market_equity, 
                                 sales, total_assets, total_liabilities):
        """
        Calculate Altman Z-Score for bankruptcy prediction
        
        Parameters:
        -----------
        working_capital : float
            Working capital
        retained_earnings : float
            Retained earnings
        ebit : float
            Earnings before interest and taxes
        market_equity : float
            Market value of equity
        sales : float
            Sales
        total_assets : float
            Total assets
        total_liabilities : float
            Total liabilities
            
        Returns:
        --------
        z_score : float
            Altman Z-Score
        interpretation : str
            Interpretation of Z-Score
        """
        # Calculate ratios
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_equity / total_liabilities
        x5 = sales / total_assets
        
        # Z-Score formula for public manufacturing companies
        z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 0.999*x5
        
        # Interpretation
        if z_score < 1.81:
            interpretation = "Distress Zone - High probability of bankruptcy"
        elif z_score < 2.99:
            interpretation = "Grey Zone - Moderate probability of bankruptcy"
        else:
            interpretation = "Safe Zone - Low probability of bankruptcy"
        
        return {
            'z_score': z_score,
            'interpretation': interpretation,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5
        }
    
    @staticmethod
    def calculate_altman_z_prime(working_capital, retained_earnings, ebit, book_equity, 
                                 sales, total_assets, total_liabilities):
        """
        Calculate Altman Z'-Score for private manufacturing companies
        """
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = book_equity / total_liabilities
        x5 = sales / total_assets
        
        z_prime = 0.717*x1 + 0.847*x2 + 3.107*x3 + 0.420*x4 + 0.998*x5
        
        if z_prime < 1.23:
            interpretation = "Distress Zone"
        elif z_prime < 2.90:
            interpretation = "Grey Zone"
        else:
            interpretation = "Safe Zone"
            
        return {'z_prime': z_prime, 'interpretation': interpretation}
    
    @staticmethod
    def calculate_ohson_o_score(total_assets, total_liabilities, working_capital, 
                               current_liabilities, net_income, funds_from_operations):
        """
        Calculate Ohlson O-Score for bankruptcy prediction
        """
        # Size (log of total assets/GNP price-level index) - simplified
        size = np.log(total_assets / 1e6)  # Simplified
        
        # Total liabilities / total assets
        tlta = total_liabilities / total_assets
        
        # Working capital / total assets
        wcta = working_capital / total_assets
        
        # Current liabilities / current assets
        clca = current_liabilities / (working_capital + current_liabilities + 1e-10)
        
        # Net income / total assets
        nita = net_income / total_assets
        
        # Funds from operations / total liabilities
        futl = funds_from_operations / (total_liabilities + 1e-10)
        
        # INTWO (1 if net income negative for last two years, else 0) - simplified
        intwo = 1 if net_income < 0 else 0
        
        # OENEG (1 if total liabilities > total assets, else 0)
        oeneg = 1 if total_liabilities > total_assets else 0
        
        # CHIN (change in net income) - simplified
        chin = 0
        
        # Calculate O-Score
        o_score = -1.32 - 0.407*size + 6.03*tlta - 1.43*wcta + 0.0757*clca - 2.37*nita - 1.83*futl + 0.285*intwo - 1.72*oeneg - 0.521*chin
        
        # Probability of bankruptcy
        prob_bankruptcy = 1 / (1 + np.exp(-o_score))
        
        return {
            'o_score': o_score,
            'probability_bankruptcy': prob_bankruptcy,
            'interpretation': 'High Risk' if prob_bankruptcy > 0.5 else 'Low Risk'
        }
    
    @staticmethod
    def calculate_merton_dd(equity_value, debt_value, equity_volatility, risk_free_rate=0.05, time_horizon=1):
        """
        Calculate Merton Distance to Default
        
        Parameters:
        -----------
        equity_value : float
            Market value of equity
        debt_value : float
            Face value of debt
        equity_volatility : float
            Equity volatility
        risk_free_rate : float
            Risk-free rate
        time_horizon : float
            Time horizon in years
            
        Returns:
        --------
        dd : float
            Distance to Default
        """
        # Initial guess for asset value
        asset_value = equity_value + debt_value
        asset_volatility = equity_volatility * equity_value / asset_value
        
        # Solve for asset value and volatility (simplified)
        def equations(x):
            V, sigma_V = x
            d1 = (np.log(V/debt_value) + (risk_free_rate + 0.5*sigma_V**2)*time_horizon) / (sigma_V*np.sqrt(time_horizon))
            d2 = d1 - sigma_V*np.sqrt(time_horizon)
            
            E_calc = V * stats.norm.cdf(d1) - debt_value * np.exp(-risk_free_rate*time_horizon) * stats.norm.cdf(d2)
            sigma_E_calc = (V/equity_value) * stats.norm.cdf(d1) * sigma_V
            
            return [E_calc - equity_value, sigma_E_calc - equity_volatility]
        
        try:
            from scipy.optimize import fsolve
            V, sigma_V = fsolve(equations, [asset_value, asset_volatility])
            
            # Distance to Default
            d2 = (np.log(V/debt_value) + (risk_free_rate - 0.5*sigma_V**2)*time_horizon) / (sigma_V*np.sqrt(time_horizon))
            
            return {
                'distance_to_default': d2,
                'asset_value': V,
                'asset_volatility': sigma_V,
                'edf': stats.norm.cdf(-d2)
            }
        except:
            # Fallback to simplified calculation
            d2 = (np.log(asset_value/debt_value) + (risk_free_rate - 0.5*asset_volatility**2)*time_horizon) / (asset_volatility*np.sqrt(time_horizon))
            return {
                'distance_to_default': d2,
                'asset_value': asset_value,
                'asset_volatility': asset_volatility,
                'edf': stats.norm.cdf(-d2)
            }


# ============================================================================
# PART 4: STATISTICAL METRICS
# ============================================================================

class StatisticalMetrics:
    """
    Statistical metrics for data analysis
    """
    
    @staticmethod
    def calculate_hurst_exponent(ts, max_lag=20):
        """
        Calculate Hurst exponent to measure long-term memory
        
        Parameters:
        -----------
        ts : array-like
            Time series
        max_lag : int
            Maximum lag for R/S analysis
            
        Returns:
        --------
        H : float
            Hurst exponent
        """
        ts = np.array(ts)
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Split into segments
            n = len(ts) // lag
            segments = ts[:n*lag].reshape(n, lag)
            
            # Calculate R/S for each segment
            rs_values = []
            for segment in segments:
                # Cumulative deviation
                cumdev = np.cumsum(segment - np.mean(segment))
                # Range
                R = np.max(cumdev) - np.min(cumdev)
                # Standard deviation
                S = np.std(segment, ddof=1)
                if S > 0:
                    rs_values.append(R / S)
            
            tau.append(np.mean(rs_values))
        
        # Fit power law
        lags = np.array(lags)
        tau = np.array(tau)
        
        # Linear regression on log-log scale
        coeffs = np.polyfit(np.log(lags), np.log(tau), 1)
        H = coeffs[0]
        
        return H
    
    @staticmethod
    def calculate_cointegration(series1, series2):
        """
        Calculate cointegration between two time series
        
        Parameters:
        -----------
        series1, series2 : array-like
            Time series
            
        Returns:
        --------
        results : dict
            Cointegration test results
        """
        series1 = np.array(series1)
        series2 = np.array(series2)
        
        # Simple linear regression
        X = np.column_stack([series2, np.ones(len(series2))])
        beta, alpha = np.linalg.lstsq(X, series1, rcond=None)[0]
        
        # Residuals
        residuals = series1 - beta * series2 - alpha
        
        # Augmented Dickey-Fuller test on residuals
        adf_stat, p_value = StatisticalMetrics.adf_test(residuals)
        
        # Half-life of mean reversion
        hl = StatisticalMetrics.half_life_mean_reversion(residuals)
        
        return {
            'hedge_ratio': beta,
            'alpha': alpha,
            'residuals': residuals,
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'is_cointegrated': p_value < 0.05,
            'half_life': hl
        }
    
    @staticmethod
    def adf_test(series, max_lag=None):
        """
        Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        -----------
        series : array-like
            Time series
        max_lag : int
            Maximum number of lags
            
        Returns:
        --------
        adf_stat : float
            ADF statistic
        p_value : float
            P-value
        """
        from statsmodels.tsa.stattools import adfuller
        series = np.array(series)
        
        if max_lag is None:
            max_lag = int(12 * (len(series)/100)**0.25)
        
        result = adfuller(series, maxlag=max_lag, autolag='AIC')
        
        return result[0], result[1]
    
    @staticmethod
    def half_life_mean_reversion(series):
        """
        Calculate half-life of mean reversion
        
        Parameters:
        -----------
        series : array-like
            Time series
            
        Returns:
        --------
        half_life : float
            Half-life in periods
        """
        series = np.array(series)
        
        # y(t) - y(t-1) = alpha * y(t-1) + epsilon
        y_lag = series[:-1]
        y_diff = np.diff(series)
        
        # Linear regression
        X = y_lag.reshape(-1, 1)
        y = y_diff
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
        
        if beta >= 0:
            return np.inf
        
        half_life = -np.log(2) / beta
        
        return half_life


# ============================================================================
# PART 5: PERFORMANCE ATTRIBUTION METRICS
# ============================================================================

class PerformanceAttribution:
    """
    Performance attribution metrics
    """
    
    @staticmethod
    def brinson_attribution(portfolio_weights, portfolio_returns, 
                           benchmark_weights, benchmark_returns, sector_map):
        """
        Brinson attribution model
        
        Parameters:
        -----------
        portfolio_weights : dict
            Portfolio weights by asset
        portfolio_returns : dict
            Portfolio returns by asset
        benchmark_weights : dict
            Benchmark weights by asset
        benchmark_returns : dict
            Benchmark returns by asset
        sector_map : dict
            Mapping of assets to sectors
            
        Returns:
        --------
        attribution : dict
            Attribution results
        """
        # Group by sector
        sectors = set(sector_map.values())
        
        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0
        
        sector_results = {}
        
        for sector in sectors:
            # Assets in this sector
            sector_assets = [a for a in sector_map if sector_map[a] == sector]
            
            # Sector weights
            pw_sector = sum(portfolio_weights.get(a, 0) for a in sector_assets)
            bw_sector = sum(benchmark_weights.get(a, 0) for a in sector_assets)
            
            # Sector returns (weighted average)
            if pw_sector > 0:
                pr_sector = sum(portfolio_weights.get(a, 0) * portfolio_returns.get(a, 0) 
                               for a in sector_assets) / pw_sector
            else:
                pr_sector = 0
                
            if bw_sector > 0:
                br_sector = sum(benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0) 
                               for a in sector_assets) / bw_sector
            else:
                br_sector = 0
            
            # Benchmark total return
            total_br = sum(benchmark_weights.values())
            
            # Attribution effects
            allocation = (pw_sector - bw_sector) * (br_sector - total_br)
            selection = bw_sector * (pr_sector - br_sector)
            interaction = (pw_sector - bw_sector) * (pr_sector - br_sector)
            
            allocation_effect += allocation
            selection_effect += selection
            interaction_effect += interaction
            
            sector_results[sector] = {
                'allocation': allocation,
                'selection': selection,
                'interaction': interaction,
                'total': allocation + selection + interaction
            }
        
        total_excess = allocation_effect + selection_effect + interaction_effect
        
        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_excess_return': total_excess,
            'sector_details': sector_results
        }


# ============================================================================
# PART 6: METRICS CALCULATOR (MAIN CLASS)
# ============================================================================

class MetricsCalculator:
    """
    Main metrics calculator that combines all metric classes
    """
    
    def __init__(self):
        self.financial = FinancialMetrics()
        self.risk = RiskMetrics()
        self.credit = CreditMetrics()
        self.statistical = StatisticalMetrics()
        self.attribution = PerformanceAttribution()
        
    def calculate_all_metrics(self, returns, benchmark_returns=None, prices=None):
        """
        Calculate comprehensive set of metrics
        
        Parameters:
        -----------
        returns : array-like
            Return series
        benchmark_returns : array-like, optional
            Benchmark return series
        prices : array-like, optional
            Price series
            
        Returns:
        --------
        metrics : dict
            Dictionary of all calculated metrics
        """
        returns = np.array(returns)
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = np.mean(returns)
        metrics['median_return'] = np.median(returns)
        metrics['std_dev'] = np.std(returns, ddof=1)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        metrics['min_return'] = np.min(returns)
        metrics['max_return'] = np.max(returns)
        
        # Financial metrics
        metrics['sharpe_ratio'] = self.financial.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.financial.calculate_sortino_ratio(returns)
        metrics['var_95'] = self.financial.calculate_var(returns, 0.95)
        metrics['cvar_95'] = self.financial.calculate_cvar(returns, 0.95)
        metrics['omega_ratio'] = self.financial.calculate_omega_ratio(returns)
        
        if prices is not None:
            metrics['max_drawdown'] = self.financial.calculate_max_drawdown(prices)
            metrics['calmar_ratio'] = self.financial.calculate_calmar_ratio(returns)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            benchmark_returns = np.array(benchmark_returns)
            metrics['beta'] = self.financial.calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = self.financial.calculate_alpha(returns, benchmark_returns)
            metrics['treynor_ratio'] = self.financial.calculate_treynor_ratio(returns, benchmark_returns)
            metrics['information_ratio'] = self.financial.calculate_information_ratio(returns, benchmark_returns)
            metrics['tracking_error'] = self.risk.calculate_tracking_error(returns, benchmark_returns)
        
        # Statistical metrics
        if len(returns) > 30:
            metrics['hurst_exponent'] = self.statistical.calculate_hurst_exponent(returns)
        
        return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example demonstrating the metrics library
    """
    print("=" * 60)
    print("PYTHON METRICS LIBRARY - EXAMPLE USAGE")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 252
    
    # Generate returns
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    returns = np.random.normal(0.0005, 0.01, n_days)
    benchmark_returns = np.random.normal(0.0003, 0.008, n_days)
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create calculator
    calculator = MetricsCalculator()
    
    # Calculate all metrics
    metrics = calculator.calculate_all_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        prices=prices
    )
    
    print("\n" + "=" * 60)
    print("CALCULATED METRICS")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
    
    # Credit metrics example
    print("\n" + "=" * 60)
    print("CREDIT METRICS EXAMPLE")
    print("=" * 60)
    
    # Altman Z-Score example
    z_score = calculator.credit.calculate_altman_z_score(
        working_capital=500000,
        retained_earnings=2000000,
        ebit=1500000,
        market_equity=10000000,
        sales=20000000,
        total_assets=25000000,
        total_liabilities=15000000
    )
    
    print(f"\nAltman Z-Score: {z_score['z_score']:.4f}")
    print(f"Interpretation: {z_score['interpretation']}")
    
    # Merton model example
    merton = calculator.credit.calculate_merton_dd(
        equity_value=10000000,
        debt_value=15000000,
        equity_volatility=0.30,
        risk_free_rate=0.05,
        time_horizon=1
    )
    
    print(f"\nMerton Distance to Default: {merton['distance_to_default']:.4f}")
    print(f"Expected Default Frequency: {merton['edf']:.4%}")
    
    # Statistical metrics example
    print("\n" + "=" * 60)
    print("STATISTICAL METRICS EXAMPLE")
    print("=" * 60)
    
    hurst = calculator.statistical.calculate_hurst_exponent(returns)
    print(f"Hurst Exponent: {hurst:.4f}")
    
    if hurst > 0.5:
        print("Interpretation: Trending/Persistent series")
    elif hurst < 0.5:
        print("Interpretation: Mean-reverting series")
    else:
        print("Interpretation: Random walk")
    
    # Cointegration example
    series1 = np.random.randn(100).cumsum()
    series2 = 0.5 * series1 + np.random.randn(100) * 0.1
    
    coint = calculator.statistical.calculate_cointegration(series1, series2)
    print(f"\nCointegration Test:")
    print(f"Hedge Ratio: {coint['hedge_ratio']:.4f}")
    print(f"P-value: {coint['p_value']:.4f}")
    print(f"Is Cointegrated: {coint['is_cointegrated']}")
    print(f"Half-life: {coint['half_life']:.2f} periods")
    
    return metrics


def portfolio_optimization_example():
    """
    Example of using metrics for portfolio optimization
    """
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Generate sample returns for multiple assets
    np.random.seed(42)
    n_assets = 5
    n_days = 252
    
    # Create correlation matrix
    corr = 0.5 * np.ones((n_assets, n_assets)) + 0.5 * np.eye(n_assets)
    
    # Generate correlated returns
    mean_returns = np.random.uniform(0.0002, 0.001, n_assets)
    vols = np.random.uniform(0.01, 0.03, n_assets)
    
    # Create covariance matrix
    cov_matrix = np.outer(vols, vols) * corr
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns, columns=[f'Asset_{i}' for i in range(n_assets)])
    
    # Calculate metrics
    calculator = MetricsCalculator()
    
    print("\nAsset Statistics:")
    for i in range(n_assets):
        asset_metrics = calculator.calculate_all_metrics(returns[:, i])
        print(f"\nAsset {i}:")
        print(f"  Mean Return: {asset_metrics['mean_return']:.6f}")
        print(f"  Volatility: {asset_metrics['std_dev']:.6f}")
        print(f"  Sharpe Ratio: {asset_metrics['sharpe_ratio']:.4f}")
        print(f"  VaR (95%): {asset_metrics['var_95']:.6f}")
    
    # Portfolio metrics for equal weights
    weights = np.ones(n_assets) / n_assets
    portfolio_returns = returns_df.dot(weights)
    
    print("\n" + "=" * 60)
    print("EQUAL-WEIGHT PORTFOLIO METRICS")
    print("=" * 60)
    
    portfolio_metrics = calculator.calculate_all_metrics(portfolio_returns)
    for key, value in portfolio_metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
    
    return returns_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run examples
    metrics = example_usage()
    returns_df = portfolio_optimization_example()
    
    print("\n" + "=" * 60)
    print("METRICS LIBRARY DEMONSTRATION COMPLETED")
    print("=" * 60)