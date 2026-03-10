import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

class PairsTradingSimulator:
    """
    A comprehensive pairs trading simulation system
    """
    
    def __init__(self, lookback_period=60, entry_zscore=2.0, exit_zscore=0.5):
        """
        Initialize the pairs trading simulator
        
        Parameters:
        lookback_period: int - Period for calculating rolling statistics
        entry_zscore: float - Z-score threshold to enter a trade
        exit_zscore: float - Z-score threshold to exit a trade
        """
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.results = None
        
    def fetch_data(self, tickers, start_date, end_date):
        """
        Fetch historical price data for given tickers
        """
        print(f"Fetching data for {tickers}...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Handle different data structures from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Newer version of yfinance returns MultiIndex
            if 'Adj Close' in data.columns.get_level_values(0):
                data = data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                # If neither Adj Close nor Close exists, use the first price column
                price_cols = [col for col in data.columns.get_level_values(0).unique() 
                             if 'Close' in col or 'Price' in col]
                if price_cols:
                    data = data[price_cols[0]]
                else:
                    raise ValueError("Could not find price data in the downloaded data")
        else:
            # Older version of yfinance returns simple columns
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
        
        # Ensure we have a DataFrame with tickers as columns
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(tickers) == 1:
                data.columns = tickers
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == len(tickers):
                data.columns = tickers
        
        # Drop any rows with NaN values
        data = data.dropna()
        
        print(f"Downloaded {len(data)} days of data")
        return data
    
    def find_cointegrated_pairs(self, data, significance_level=0.05):
        """
        Find cointegrated pairs among multiple assets
        """
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.columns.tolist()
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                
                if pvalue < significance_level:
                    pairs.append((keys[i], keys[j], pvalue))
        
        # Sort pairs by p-value (lowest first)
        pairs.sort(key=lambda x: x[2])
        
        return score_matrix, pvalue_matrix, pairs
    
    def calculate_spread(self, series1, series2):
        """
        Calculate the spread between two series using linear regression
        """
        # Perform linear regression to find hedge ratio
        X = sm.add_constant(series1)
        model = sm.OLS(series2, X).fit()
        hedge_ratio = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        spread = series2 - hedge_ratio * series1
        
        return spread, hedge_ratio
    
    def calculate_zscore(self, spread):
        """
        Calculate rolling z-score of the spread
        """
        rolling_mean = spread.rolling(window=self.lookback_period, min_periods=int(self.lookback_period/2)).mean()
        rolling_std = spread.rolling(window=self.lookback_period, min_periods=int(self.lookback_period/2)).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def generate_signals(self, zscore):
        """
        Generate trading signals based on z-score thresholds
        """
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['position'] = 0
        
        # Enter positions when zscore crosses thresholds
        signals.loc[zscore > self.entry_zscore, 'position'] = -1  # Short spread
        signals.loc[zscore < -self.entry_zscore, 'position'] = 1  # Long spread
        
        # Exit positions when zscore reverts
        signals.loc[(signals['position'].shift(1) == -1) & (zscore < self.exit_zscore), 'position'] = 0
        signals.loc[(signals['position'].shift(1) == 1) & (zscore > -self.exit_zscore), 'position'] = 0
        
        # Forward fill positions (maintain position until exit signal)
        signals['position'] = signals['position'].replace(to_replace=0, method='ffill')
        signals['position'] = signals['position'].fillna(0)
        
        return signals
    
    def backtest_pair(self, data, ticker1, ticker2):
        """
        Run backtest for a single pair
        """
        # Calculate spread and z-score
        spread, hedge_ratio = self.calculate_spread(data[ticker1], data[ticker2])
        zscore = self.calculate_zscore(spread)
        
        # Generate signals
        signals = self.generate_signals(zscore)
        
        # Calculate returns
        returns1 = data[ticker1].pct_change()
        returns2 = data[ticker2].pct_change()
        
        # Portfolio returns (long/short positions)
        # When position = 1: Long spread (long stock2, short stock1)
        # When position = -1: Short spread (short stock2, long stock1)
        signals['portfolio_returns'] = (
            signals['position'].shift(1) * (returns2 - hedge_ratio * returns1)
        )
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['portfolio_returns']).cumprod()
        
        # Calculate metrics
        metrics = self.calculate_metrics(signals['portfolio_returns'].dropna())
        
        return signals, metrics, hedge_ratio
    
    def calculate_metrics(self, returns):
        """
        Calculate performance metrics
        """
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Total return
        metrics['total_return'] = (1 + returns).prod() - 1
        
        # Annualized return (assuming 252 trading days)
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (252/len(returns)) - 1
        
        # Annualized volatility
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        winning_trades = returns[returns > 0].count()
        total_trades = returns[returns != 0].count()
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades
        metrics['num_trades'] = total_trades
        
        return metrics
    
    def plot_results(self, data, ticker1, ticker2, signals, metrics, hedge_ratio):
        """
        Plot the pairs trading results
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle(f'Pairs Trading Analysis: {ticker1} vs {ticker2}', fontsize=16)
        
        # Plot 1: Price series (normalized)
        normalized_data = data / data.iloc[0]
        axes[0].plot(normalized_data.index, normalized_data[ticker1], label=ticker1, alpha=0.7, linewidth=1.5)
        axes[0].plot(normalized_data.index, normalized_data[ticker2], label=ticker2, alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel('Normalized Price')
        axes[0].legend(loc='upper left')
        axes[0].set_title('Normalized Price Series')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Spread and z-score
        spread, _ = self.calculate_spread(data[ticker1], data[ticker2])
        zscore = self.calculate_zscore(spread)
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(spread.index, spread, color='blue', alpha=0.5, label='Spread', linewidth=1)
        ax2.set_ylabel('Spread', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        line2 = ax2_twin.plot(zscore.index, zscore, color='red', alpha=0.7, label='Z-Score', linewidth=1.5)
        ax2_twin.axhline(y=self.entry_zscore, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2_twin.axhline(y=-self.entry_zscore, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2_twin.axhline(y=self.exit_zscore, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax2_twin.axhline(y=-self.exit_zscore, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax2_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2_twin.set_ylabel('Z-Score', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        axes[1].set_title(f'Spread and Z-Score (Hedge Ratio: {hedge_ratio:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Positions
        axes[2].fill_between(signals.index, 0, signals['position'], 
                            where=signals['position'] > 0, color='green', alpha=0.3, label='Long Spread')
        axes[2].fill_between(signals.index, 0, signals['position'], 
                            where=signals['position'] < 0, color='red', alpha=0.3, label='Short Spread')
        axes[2].set_ylabel('Position')
        axes[2].set_title('Trading Positions')
        axes[2].legend(loc='upper right')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative returns
        axes[3].plot(signals.index, signals['cumulative_returns'], color='purple', linewidth=2, label='Strategy')
        axes[3].fill_between(signals.index, 1, signals['cumulative_returns'], 
                            where=signals['cumulative_returns'] > 1, color='green', alpha=0.2)
        axes[3].fill_between(signals.index, 1, signals['cumulative_returns'], 
                            where=signals['cumulative_returns'] < 1, color='red', alpha=0.2)
        
        # Add buy and hold comparison
        bh_returns = (1 + (data[ticker2].pct_change() - data[ticker1].pct_change())).cumprod()
        axes[3].plot(bh_returns.index, bh_returns, color='gray', alpha=0.5, linewidth=1, label='Buy & Hold Spread')
        
        axes[3].axhline(y=1, color='black', linestyle='-', alpha=0.3)
        axes[3].set_ylabel('Cumulative Returns')
        axes[3].set_xlabel('Date')
        axes[3].set_title('Strategy Performance')
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print("\n" + "="*60)
        print(f"PERFORMANCE METRICS FOR {ticker1} - {ticker2}")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title():<25}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title():<25}: {value}")

def main():
    """
    Main function to run the pairs trading simulation
    """
    # Initialize simulator
    simulator = PairsTradingSimulator(lookback_period=60, entry_zscore=2.0, exit_zscore=0.5)
    
    # Example 1: Test a single known pair
    print("\n" + "="*60)
    print("EXAMPLE 1: Testing a single pair (AAPL vs MSFT)")
    print("="*60)
    
    try:
        data = simulator.fetch_data(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
        
        # Check if we have enough data
        if len(data) < simulator.lookback_period:
            print(f"Warning: Only {len(data)} days of data. Need at least {simulator.lookback_period} days.")
        else:
            # Run backtest
            signals, metrics, hedge_ratio = simulator.backtest_pair(data, 'AAPL', 'MSFT')
            
            # Plot results
            simulator.plot_results(data, 'AAPL', 'MSFT', signals, metrics, hedge_ratio)
            
    except Exception as e:
        print(f"Error in Example 1: {e}")
        print("Trying with alternative tickers...")
        
        # Try with different tickers if AAPL/MSFT fails
        try:
            data = simulator.fetch_data(['SPY', 'QQQ'], '2020-01-01', '2023-12-31')
            signals, metrics, hedge_ratio = simulator.backtest_pair(data, 'SPY', 'QQQ')
            simulator.plot_results(data, 'SPY', 'QQQ', signals, metrics, hedge_ratio)
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
    
    # Example 2: Find cointegrated pairs among multiple stocks
    print("\n" + "="*60)
    print("EXAMPLE 2: Finding cointegrated pairs among multiple stocks")
    print("="*60)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']
    
    try:
        data_multi = simulator.fetch_data(tickers, '2020-01-01', '2023-12-31')
        
        # Find cointegrated pairs
        score_matrix, pvalue_matrix, pairs = simulator.find_cointegrated_pairs(data_multi)
        
        print(f"\nFound {len(pairs)} cointegrated pairs (p-value < 0.05):")
        if pairs:
            for i, (ticker1, ticker2, pvalue) in enumerate(pairs, 1):
                print(f"{i:2d}. {ticker1:6} - {ticker2:6}: p-value = {pvalue:.6f}")
            
            # Test the best pair
            best_pair = pairs[0]
            print(f"\nTesting best pair: {best_pair[0]} vs {best_pair[1]}")
            signals_best, metrics_best, hedge_ratio_best = simulator.backtest_pair(
                data_multi, best_pair[0], best_pair[1]
            )
            simulator.plot_results(data_multi, best_pair[0], best_pair[1], 
                                 signals_best, metrics_best, hedge_ratio_best)
        else:
            print("No cointegrated pairs found with p-value < 0.05")
            
    except Exception as e:
        print(f"Error in Example 2: {e}")

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Run main function
    main()
    
    # Additional example: Test with different parameters
    print("\n" + "="*60)
    print("BONUS: Testing with different parameters")
    print("="*60)
    
    # Create simulator with different thresholds
    simulator2 = PairsTradingSimulator(lookback_period=30, entry_zscore=1.5, exit_zscore=0.3)
    
    try:
        data = simulator2.fetch_data(['JPM', 'BAC'], '2020-01-01', '2023-12-31')
        signals, metrics, hedge_ratio = simulator2.backtest_pair(data, 'JPM', 'BAC')
        simulator2.plot_results(data, 'JPM', 'BAC', signals, metrics, hedge_ratio)
    except Exception as e:
        print(f"Could not fetch bank data: {e}")