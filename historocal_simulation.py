import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class HistoricalSimulation:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize historical simulation with stock data
        
        Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        
    def fetch_data(self):
        """Fetch historical stock data - Updated for current yfinance version"""
        try:
            # Download data with auto_adjust=False to get all columns
            stock = yf.download(self.ticker, start=self.start_date, end=self.end_date, 
                               progress=False, auto_adjust=False)
            
            # Check what columns are available
            print(f"Available columns: {stock.columns.tolist()}")
            
            # Handle MultiIndex columns if present
            if isinstance(stock.columns, pd.MultiIndex):
                # For multi-index, try to get the correct column
                if ('Adj Close', self.ticker) in stock.columns:
                    self.data = stock[('Adj Close', self.ticker)]
                elif ('Close', self.ticker) in stock.columns:
                    self.data = stock[('Close', self.ticker)]
                    print("Using 'Close' prices instead of 'Adj Close'")
                else:
                    # Try to find any price column
                    for level0 in ['Adj Close', 'Close', 'Open']:
                        if (level0, self.ticker) in stock.columns:
                            self.data = stock[(level0, self.ticker)]
                            print(f"Using '{level0}' prices")
                            break
            else:
                # Simple columns
                if 'Adj Close' in stock.columns:
                    self.data = stock['Adj Close']
                elif 'Close' in stock.columns:
                    self.data = stock['Close']
                    print("Using 'Close' prices instead of 'Adj Close'")
                else:
                    # If neither exists, try to get the first price column
                    price_columns = [col for col in stock.columns if 'Close' in col or 'Price' in col]
                    if price_columns:
                        self.data = stock[price_columns[0]]
                        print(f"Using '{price_columns[0]}' prices")
                    else:
                        raise ValueError("No price data columns found")
            
            # Ensure data is a Series
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.iloc[:, 0]
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            
            print(f"Data fetched successfully for {self.ticker}")
            print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Number of observations: {len(self.data)}")
            
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_var(self, confidence_level=0.95, horizon=1):
        """
        Calculate Value at Risk using historical simulation
        
        Parameters:
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        horizon (int): Time horizon in days
        
        Returns:
        float: Value at Risk
        """
        if self.returns is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Scale returns by horizon (simplified approach)
        horizon_returns = self.returns * np.sqrt(horizon)
        
        # Calculate VaR
        var = np.percentile(horizon_returns, (1 - confidence_level) * 100)
        
        return float(var)
    
    def calculate_expected_shortfall(self, confidence_level=0.95, horizon=1):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Parameters:
        confidence_level (float): Confidence level
        horizon (int): Time horizon in days
        
        Returns:
        float: Expected Shortfall
        """
        if self.returns is None:
            raise ValueError("No data available. Please fetch data first.")
        
        horizon_returns = self.returns * np.sqrt(horizon)
        var = self.calculate_var(confidence_level, horizon)
        
        # Calculate ES as average of returns beyond VaR
        tail_returns = horizon_returns[horizon_returns <= var]
        es = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return float(es)
    
    def plot_historical_simulation(self, confidence_level=0.95):
        """Plot historical returns with VaR and ES"""
        if self.returns is None:
            raise ValueError("No data available. Please fetch data first.")
        
        var = self.calculate_var(confidence_level)
        es = self.calculate_expected_shortfall(confidence_level)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Time series of prices
        axes[0].plot(self.data.index, self.data.values)
        axes[0].set_title(f'{self.ticker} - Historical Prices')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram of returns
        axes[1].hist(self.returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1].axvline(x=var, color='red', linestyle='--', linewidth=2, 
                       label=f'VaR ({confidence_level:.0%}): {var:.4f}')
        axes[1].axvline(x=es, color='orange', linestyle='--', linewidth=2,
                       label=f'ES ({confidence_level:.0%}): {es:.4f}')
        axes[1].set_title(f'{self.ticker} - Returns Distribution')
        axes[1].set_xlabel('Daily Returns')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def simulate_future_prices(self, days=30, simulations=1000):
        """
        Simulate future price paths using historical bootstrap
        
        Parameters:
        days (int): Number of days to simulate
        simulations (int): Number of simulation paths
        
        Returns:
        numpy.array: Simulated price paths
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        last_price = float(self.data.iloc[-1])
        simulated_prices = np.zeros((days, simulations))
        
        for sim in range(simulations):
            # Randomly sample historical returns
            sampled_returns = np.random.choice(self.returns, size=days, replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + sampled_returns)
            
            # Simulate price path
            simulated_prices[:, sim] = last_price * cumulative_returns
        
        return simulated_prices
    
    def calculate_risk_metrics(self):
        """Calculate various risk metrics"""
        if self.returns is None:
            raise ValueError("No data available. Please fetch data first.")
        
        metrics = {}
        
        # Basic statistics - convert to float to ensure they're scalar
        metrics['Mean Return'] = float(self.returns.mean())
        metrics['Std Dev'] = float(self.returns.std())
        metrics['Skewness'] = float(self.returns.skew())
        metrics['Kurtosis'] = float(self.returns.kurtosis())
        
        # VaR at different confidence levels
        for cl in [0.90, 0.95, 0.99]:
            metrics[f'VaR {cl:.0%}'] = self.calculate_var(cl)
            metrics[f'ES {cl:.0%}'] = self.calculate_expected_shortfall(cl)
        
        # Maximum drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['Max Drawdown'] = float(drawdown.min())
        
        # Sharpe ratio (assuming 0% risk-free rate)
        metrics['Sharpe Ratio'] = float(self.returns.mean() / self.returns.std() * np.sqrt(252))
        
        # Additional metrics
        metrics['Positive Days'] = float((self.returns > 0).sum() / len(self.returns))
        metrics['Negative Days'] = float((self.returns < 0).sum() / len(self.returns))
        metrics['Best Day'] = float(self.returns.max())
        metrics['Worst Day'] = float(self.returns.min())
        
        return metrics

def main():
    """Main function to run the historical simulation"""
    
    # Example 1: Single stock simulation
    print("=" * 60)
    print("HISTORICAL SIMULATION - SINGLE STOCK")
    print("=" * 60)
    
    # Create historical simulation object for Apple stock
    hs = HistoricalSimulation("AAPL", "2020-01-01", "2024-01-01")
    
    # Fetch data
    if hs.fetch_data():
        # Calculate risk metrics
        metrics = hs.calculate_risk_metrics()
        
        print("\nRisk Metrics:")
        print("-" * 50)
        for key, value in metrics.items():
            if 'VaR' in key or 'ES' in key or key in ['Mean Return', 'Std Dev', 'Best Day', 'Worst Day']:
                print(f"{key:20s}: {value:.6f} ({value*100:.4f}%)")
            else:
                print(f"{key:20s}: {value:.6f}")
        
        # Plot the distribution
        hs.plot_historical_simulation(0.95)
        
        # Simulate future prices
        print("\nSimulating future price paths...")
        future_prices = hs.simulate_future_prices(days=30, simulations=1000)
        
        # Plot simulation results
        plt.figure(figsize=(12, 6))
        
        # Plot a sample of paths
        for i in range(min(100, future_prices.shape[1])):
            plt.plot(future_prices[:, i], alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot statistics
        mean_path = future_prices.mean(axis=1)
        upper_bound = np.percentile(future_prices, 95, axis=1)
        lower_bound = np.percentile(future_prices, 5, axis=1)
        
        plt.plot(mean_path, color='red', linewidth=2, label='Mean Path')
        plt.fill_between(range(len(mean_path)), lower_bound, upper_bound, 
                        alpha=0.2, color='red', label='90% Confidence Interval')
        
        plt.title(f'{hs.ticker} - Historical Bootstrap Simulation (30 days, 1000 paths)')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Summary statistics for simulation
        final_prices = future_prices[-1, :]
        current_price = float(hs.data.iloc[-1])
        
        print(f"\nSimulation Results (30-day horizon):")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Mean Final Price: ${float(final_prices.mean()):.2f}")
        print(f"Median Final Price: ${float(np.median(final_prices)):.2f}")
        print(f"5th Percentile (95% VaR price): ${float(np.percentile(final_prices, 5)):.2f}")
        print(f"95% VaR (return): {(float(np.percentile(final_prices, 5)) / current_price - 1)*100:.2f}%")
        print(f"95th Percentile: ${float(np.percentile(final_prices, 95)):.2f}")
        print(f"Probability of loss: {(final_prices < current_price).mean()*100:.2f}%")
    
    # Example 2: Compare multiple stocks
    print("\n" + "=" * 60)
    print("COMPARING MULTIPLE STOCKS")
    print("=" * 60)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    results = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        stock = HistoricalSimulation(ticker, "2020-01-01", "2024-01-01")
        if stock.fetch_data():
            var_95 = stock.calculate_var(0.95)
            es_95 = stock.calculate_expected_shortfall(0.95)
            sharpe = float(stock.returns.mean() / stock.returns.std() * np.sqrt(252))
            max_dd = float(((1 + stock.returns).cumprod().expanding().max() - (1 + stock.returns).cumprod()).max())
            
            print(f"  95% VaR: {var_95:.4f} ({var_95*100:.2f}%)")
            print(f"  95% ES: {es_95:.4f} ({es_95*100:.2f}%)")
            print(f"  Sharpe Ratio: {sharpe:.4f}")
            print(f"  Max Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
            
            results.append({
                'Ticker': ticker,
                '95% VaR': f"{var_95*100:.2f}%",
                '95% ES': f"{es_95*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.4f}",
                'Max Drawdown': f"{max_dd*100:.2f}%"
            })
    
    # Create comparison table
    if results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        comparison_df = pd.DataFrame(results)
        print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()