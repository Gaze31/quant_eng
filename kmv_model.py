import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class KMVModel:
    """
    KMV (Kealhofer-McQuown-Vasicek) Model for credit risk assessment
    Based on Merton's structural model for default probability estimation
    """
    
    def __init__(self, risk_free_rate=0.05, time_horizon=1.0):
        """
        Initialize KMV model
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate (annualized)
        time_horizon : float
            Time horizon in years (usually 1 year)
        """
        self.r = risk_free_rate
        self.T = time_horizon
        
    def calculate_firm_value_volatility(self, E, sigma_E, D, r, T):
        """
        Estimate firm value (V) and asset volatility (sigma_V) using 
        iterative process (solving Merton's equations)
        
        Parameters:
        -----------
        E : float
            Market value of equity
        sigma_E : float
            Equity volatility
        D : float
            Default point (short-term debt + 0.5*long-term debt)
        r : float
            Risk-free rate
        T : float
            Time horizon
            
        Returns:
        --------
        V : float
            Firm asset value
        sigma_V : float
            Asset volatility
        """
        
        def equations(x):
            V, sigma_V = x
            
            # Black-Scholes d1 and d2
            d1 = (np.log(V/D) + (r + 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
            d2 = d1 - sigma_V*np.sqrt(T)
            
            # Equation 1: E = V * N(d1) - D * exp(-rT) * N(d2)
            E_calc = V * stats.norm.cdf(d1) - D * np.exp(-r*T) * stats.norm.cdf(d2)
            
            # Equation 2: sigma_E = (V/E) * N(d1) * sigma_V
            sigma_E_calc = (V/E) * stats.norm.cdf(d1) * sigma_V
            
            return [E_calc - E, sigma_E_calc - sigma_E]
        
        # Initial guess
        V_init = E + D
        sigma_V_init = sigma_E * E / (E + D)
        
        # Solve system of equations
        from scipy.optimize import fsolve
        V, sigma_V = fsolve(equations, [V_init, sigma_V_init])
        
        return V, sigma_V
    
    def calculate_distance_to_default(self, V, D, sigma_V, T):
        """
        Calculate Distance to Default (DD)
        
        Parameters:
        -----------
        V : float
            Firm asset value
        D : float
            Default point
        sigma_V : float
            Asset volatility
        T : float
            Time horizon
            
        Returns:
        --------
        DD : float
            Distance to Default
        """
        DD = (np.log(V/D) + (self.r - 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
        return DD
    
    def calculate_edf(self, DD):
        """
        Calculate Expected Default Frequency (EDF)
        
        Parameters:
        -----------
        DD : float
            Distance to Default
            
        Returns:
        --------
        EDF : float
            Expected Default Frequency (probability)
        """
        # In KMV, EDF is derived from empirical mapping
        # Here we use normal distribution as approximation
        EDF = stats.norm.cdf(-DD)
        return EDF
    
    def fit(self, E, sigma_E, D):
        """
        Fit KMV model and calculate all metrics
        
        Parameters:
        -----------
        E : float
            Market value of equity
        sigma_E : float
            Equity volatility
        D : float
            Default point
            
        Returns:
        --------
        dict : Dictionary containing all KMV metrics
        """
        # Calculate firm value and asset volatility
        V, sigma_V = self.calculate_firm_value_volatility(E, sigma_E, D, self.r, self.T)
        
        # Calculate Distance to Default
        DD = self.calculate_distance_to_default(V, D, sigma_V, self.T)
        
        # Calculate EDF
        EDF = self.calculate_edf(DD)
        
        return {
            'firm_value': V,
            'asset_volatility': sigma_V,
            'distance_to_default': DD,
            'expected_default_frequency': EDF,
            'default_probability_pct': EDF * 100,
            'credit_rating': self._get_credit_rating(EDF)
        }
    
    def _get_credit_rating(self, edf):
        """
        Map EDF to credit rating (approximate mapping)
        """
        if edf < 0.0001:  # 0.01%
            return 'AAA'
        elif edf < 0.0005:  # 0.05%
            return 'AA'
        elif edf < 0.001:  # 0.1%
            return 'A'
        elif edf < 0.0025:  # 0.25%
            return 'BBB'
        elif edf < 0.01:  # 1%
            return 'BB'
        elif edf < 0.05:  # 5%
            return 'B'
        elif edf < 0.10:  # 10%
            return 'CCC'
        else:
            return 'D'


class KMVPortfolio:
    """
    KMV model implementation for a portfolio of firms
    """
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.results = {}
        
    def add_firm(self, ticker, equity_data, debt_data):
        """
        Add firm to portfolio
        
        Parameters:
        -----------
        ticker : str
            Firm identifier
        equity_data : dict
            Contains 'market_cap' and 'volatility'
        debt_data : dict
            Contains 'short_term_debt' and 'long_term_debt'
        """
        # Calculate default point (KMV approximation)
        default_point = debt_data['short_term_debt'] + 0.5 * debt_data['long_term_debt']
        
        self.results[ticker] = {
            'equity_value': equity_data['market_cap'],
            'equity_volatility': equity_data['volatility'],
            'default_point': default_point,
            'short_term_debt': debt_data['short_term_debt'],
            'long_term_debt': debt_data['long_term_debt']
        }
    
    def analyze_portfolio(self, time_horizon=1.0):
        """
        Analyze all firms in portfolio
        """
        results = []
        
        for ticker, data in self.results.items():
            kmv = KMVModel(risk_free_rate=self.risk_free_rate, time_horizon=time_horizon)
            
            try:
                result = kmv.fit(
                    E=data['equity_value'],
                    sigma_E=data['equity_volatility'],
                    D=data['default_point']
                )
                
                result['ticker'] = ticker
                result['equity_value'] = data['equity_value']
                result['default_point'] = data['default_point']
                result['short_term_debt'] = data['short_term_debt']
                result['long_term_debt'] = data['long_term_debt']
                
                results.append(result)
                
            except:
                print(f"Could not analyze {ticker}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_portfolio_risk(self, df):
        """
        Plot portfolio risk metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distance to Default
        axes[0, 0].barh(df['ticker'], df['distance_to_default'])
        axes[0, 0].set_xlabel('Distance to Default')
        axes[0, 0].set_title('Distance to Default by Firm')
        axes[0, 0].axvline(x=1, color='r', linestyle='--', label='High Risk')
        axes[0, 0].axvline(x=3, color='g', linestyle='--', label='Low Risk')
        axes[0, 0].legend()
        
        # EDF
        colors = ['red' if x > 0.05 else 'orange' if x > 0.01 else 'green' 
                 for x in df['expected_default_frequency']]
        axes[0, 1].barh(df['ticker'], df['expected_default_frequency']*100, color=colors)
        axes[0, 1].set_xlabel('Expected Default Frequency (%)')
        axes[0, 1].set_title('EDF by Firm')
        axes[0, 1].axvline(x=1, color='orange', linestyle='--', label='Warning')
        axes[0, 1].axvline(x=5, color='red', linestyle='--', label='High Risk')
        axes[0, 1].legend()
        
        # Asset vs Equity Value
        axes[1, 0].scatter(df['equity_value']/1e6, df['firm_value']/1e6, alpha=0.6)
        axes[1, 0].set_xlabel('Equity Value (Millions)')
        axes[1, 0].set_ylabel('Firm Value (Millions)')
        axes[1, 0].set_title('Firm Value vs Equity Value')
        
        # Add diagonal line
        max_val = max(df['firm_value'].max(), df['equity_value'].max())/1e6
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Credit Rating Distribution
        rating_counts = df['credit_rating'].value_counts()
        axes[1, 1].pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Credit Rating Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return fig


class RealDataKMV:
    """
    KMV model with real market data using yfinance
    """
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize with list of tickers
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def fetch_market_data(self):
        """
        Fetch market data for all tickers
        """
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get historical prices for volatility calculation
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                if len(hist) > 0:
                    # Calculate daily returns and volatility
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    
                    # Get latest market cap
                    info = stock.info
                    market_cap = info.get('marketCap', None)
                    
                    if market_cap is not None:
                        self.data[ticker] = {
                            'market_cap': market_cap,
                            'volatility': volatility,
                            'last_price': hist['Close'].iloc[-1],
                            'returns': returns
                        }
                        print(f"✓ Fetched data for {ticker}")
                    else:
                        print(f"✗ No market cap for {ticker}")
                else:
                    print(f"✗ No historical data for {ticker}")
                    
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {str(e)}")
        
        return self.data
    
    def estimate_debt(self, ticker, industry_avg_leverage=0.5):
        """
        Estimate debt structure (simplified - in practice, get from financial statements)
        """
        if ticker in self.data:
            market_cap = self.data[ticker]['market_cap']
            
            # Simplified debt estimation based on industry averages
            total_assets = market_cap / (1 - industry_avg_leverage)
            total_debt = total_assets * industry_avg_leverage
            
            # Assume 30% short-term, 70% long-term
            short_term_debt = total_debt * 0.3
            long_term_debt = total_debt * 0.7
            
            return {
                'short_term_debt': short_term_debt,
                'long_term_debt': long_term_debt
            }
        else:
            return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_single_firm():
    """
    Example: Analyze single firm with hypothetical data
    """
    print("=" * 60)
    print("KMV MODEL - SINGLE FIRM ANALYSIS")
    print("=" * 60)
    
    # Firm data (hypothetical)
    equity_value = 1000  # Market cap: $1,000 million
    equity_volatility = 0.40  # 40% annual volatility
    short_term_debt = 300  # $300 million
    long_term_debt = 700  # $700 million
    
    # Calculate default point (KMV approach)
    default_point = short_term_debt + 0.5 * long_term_debt
    
    print(f"\nFirm Input Data:")
    print(f"Equity Value: ${equity_value:.0f} million")
    print(f"Equity Volatility: {equity_volatility:.1%}")
    print(f"Short-term Debt: ${short_term_debt:.0f} million")
    print(f"Long-term Debt: ${long_term_debt:.0f} million")
    print(f"Default Point: ${default_point:.0f} million")
    
    # Initialize and fit KMV model
    kmv = KMVModel(risk_free_rate=0.05, time_horizon=1.0)
    results = kmv.fit(equity_value, equity_volatility, default_point)
    
    print(f"\nKMV Model Results:")
    print(f"Firm Asset Value: ${results['firm_value']:.2f} million")
    print(f"Asset Volatility: {results['asset_volatility']:.2%}")
    print(f"Distance to Default: {results['distance_to_default']:.3f}")
    print(f"Expected Default Frequency: {results['expected_default_frequency']:.4%}")
    print(f"Equivalent Credit Rating: {results['credit_rating']}")
    
    return results


def example_portfolio_analysis():
    """
    Example: Analyze portfolio of firms
    """
    print("\n" + "=" * 60)
    print("KMV MODEL - PORTFOLIO ANALYSIS")
    print("=" * 60)
    
    # Create portfolio with hypothetical firms
    portfolio = KMVPortfolio(risk_free_rate=0.05)
    
    # Add firms with different risk profiles
    firms = {
        'Firm_A_AAA': {'market_cap': 5000, 'volatility': 0.20, 'std': 200, 'ltd': 800},
        'Firm_B_AA': {'market_cap': 3000, 'volatility': 0.25, 'std': 400, 'ltd': 1000},
        'Firm_C_A': {'market_cap': 2000, 'volatility': 0.30, 'std': 500, 'ltd': 800},
        'Firm_D_BBB': {'market_cap': 1500, 'volatility': 0.35, 'std': 600, 'ltd': 700},
        'Firm_E_BB': {'market_cap': 1000, 'volatility': 0.40, 'std': 500, 'ltd': 600},
        'Firm_F_B': {'market_cap': 800, 'volatility': 0.45, 'std': 400, 'ltd': 500},
        'Firm_G_CCC': {'market_cap': 500, 'volatility': 0.50, 'std': 300, 'ltd': 400},
    }
    
    for firm, data in firms.items():
        portfolio.add_firm(
            ticker=firm,
            equity_data={'market_cap': data['market_cap'], 'volatility': data['volatility']},
            debt_data={'short_term_debt': data['std'], 'long_term_debt': data['ltd']}
        )
    
    # Analyze portfolio
    results_df = portfolio.analyze_portfolio()
    
    print("\nPortfolio Analysis Results:")
    print(results_df[['ticker', 'distance_to_default', 'expected_default_frequency', 
                      'credit_rating']].to_string(index=False))
    
    # Plot results
    portfolio.plot_portfolio_risk(results_df)
    
    return results_df


def example_real_data():
    """
    Example: Use real market data
    """
    print("\n" + "=" * 60)
    print("KMV MODEL - REAL MARKET DATA")
    print("=" * 60)
    
    # Define tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WFC']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch real market data
    print("\nFetching market data...")
    real_data = RealDataKMV(tickers, start_date, end_date)
    market_data = real_data.fetch_market_data()
    
    if not market_data:
        print("No data fetched. Using synthetic data instead.")
        return example_portfolio_analysis()
    
    # Create portfolio
    portfolio = KMVPortfolio(risk_free_rate=0.05)
    
    print("\nEstimating debt structure (simplified)...")
    for ticker in market_data.keys():
        debt = real_data.estimate_debt(ticker, industry_avg_leverage=0.5)
        
        if debt:
            portfolio.add_firm(
                ticker=ticker,
                equity_data={
                    'market_cap': market_data[ticker]['market_cap'],
                    'volatility': market_data[ticker]['volatility']
                },
                debt_data=debt
            )
            print(f"  {ticker}: Market Cap=${market_data[ticker]['market_cap']/1e9:.2f}B, "
                  f"Volatility={market_data[ticker]['volatility']:.2%}")
    
    # Analyze portfolio
    results_df = portfolio.analyze_portfolio()
    
    print("\n" + "=" * 60)
    print("KMV RESULTS FOR REAL COMPANIES")
    print("=" * 60)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['ticker']}:")
        print(f"  Distance to Default: {row['distance_to_default']:.3f}")
        print(f"  EDF: {row['expected_default_frequency']:.4%}")
        print(f"  Rating: {row['credit_rating']}")
        print(f"  Firm Value: ${row['firm_value']/1e9:.2f}B")
        print(f"  Asset Volatility: {row['asset_volatility']:.2%}")
    
    # Plot results
    portfolio.plot_portfolio_risk(results_df)
    
    return results_df


def sensitivity_analysis():
    """
    Sensitivity analysis of KMV model parameters
    """
    print("\n" + "=" * 60)
    print("KMV MODEL - SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Base case
    base_equity = 1000
    base_volatility = 0.30
    base_debt = 800
    
    kmv = KMVModel()
    
    # Analyze sensitivity to leverage
    leverage_levels = np.linspace(0.3, 0.8, 20)
    edf_values = []
    dd_values = []
    
    print("\nAnalyzing sensitivity to leverage (Debt/Assets ratio)...")
    
    for lev in leverage_levels:
        # Calculate debt given target leverage
        V_init = base_equity + base_debt
        target_debt = V_init * lev
        target_equity = V_init - target_debt
        
        results = kmv.fit(target_equity, base_volatility, target_debt)
        edf_values.append(results['expected_default_frequency'])
        dd_values.append(results['distance_to_default'])
    
    # Plot sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(leverage_levels, dd_values, 'b-', linewidth=2)
    axes[0].set_xlabel('Leverage (Debt/Assets)')
    axes[0].set_ylabel('Distance to Default')
    axes[0].set_title('Distance to Default vs Leverage')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(leverage_levels, np.array(edf_values)*100, 'r-', linewidth=2)
    axes[1].set_xlabel('Leverage (Debt/Assets)')
    axes[1].set_ylabel('EDF (%)')
    axes[1].set_title('Expected Default Frequency vs Leverage')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze sensitivity to volatility
    volatility_levels = np.linspace(0.1, 0.8, 20)
    edf_values = []
    dd_values = []
    
    print("\nAnalyzing sensitivity to volatility...")
    
    for vol in volatility_levels:
        results = kmv.fit(base_equity, vol, base_debt)
        edf_values.append(results['expected_default_frequency'])
        dd_values.append(results['distance_to_default'])
    
    # Plot sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(volatility_levels, dd_values, 'b-', linewidth=2)
    axes[0].set_xlabel('Equity Volatility')
    axes[0].set_ylabel('Distance to Default')
    axes[0].set_title('Distance to Default vs Volatility')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(volatility_levels, np.array(edf_values)*100, 'r-', linewidth=2)
    axes[1].set_xlabel('Equity Volatility')
    axes[1].set_ylabel('EDF (%)')
    axes[1].set_title('Expected Default Frequency vs Volatility')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def compare_with_ratings():
    """
    Compare KMV EDF with credit ratings
    """
    print("\n" + "=" * 60)
    print("KMV MODEL - EDF VS CREDIT RATINGS")
    print("=" * 60)
    
    # Historical average EDF by rating (approximate)
    rating_edf = {
        'AAA': 0.0002,   # 0.02%
        'AA': 0.0005,    # 0.05%
        'A': 0.001,      # 0.1%
        'BBB': 0.0025,   # 0.25%
        'BB': 0.01,      # 1.0%
        'B': 0.05,       # 5.0%
        'CCC': 0.15,     # 15.0%
        'D': 0.50        # 50.0%
    }
    
    ratings = list(rating_edf.keys())
    edfs = list(rating_edf.values())
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(ratings, [e*100 for e in edfs], color='skyblue', edgecolor='navy')
    
    # Add threshold lines
    plt.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Investment Grade')
    plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Speculative Grade')
    plt.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='High Yield')
    
    plt.xlabel('Credit Rating')
    plt.ylabel('Expected Default Frequency (%)')
    plt.title('KMV EDF by Credit Rating (Historical Averages)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, edf in zip(bars, edfs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{edf*100:.3f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating KMV model usage
    """
    print("=" * 60)
    print("KMV (KEALHOFER-MCQUOWN-VASICEK) MODEL")
    print("Credit Risk Assessment using Structural Approach")
    print("=" * 60)
    
    # 1. Single firm analysis
    single_firm_results = example_single_firm()
    
    # 2. Portfolio analysis
    portfolio_results = example_portfolio_analysis()
    
    # 3. Real data example (if available)
    try:
        real_results = example_real_data()
    except:
        print("\nReal data example skipped - using synthetic only")
    
    # 4. Sensitivity analysis
    sensitivity_analysis()
    
    # 5. Rating comparison
    compare_with_ratings()
    
    print("\n" + "=" * 60)
    print("KMV MODEL ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()