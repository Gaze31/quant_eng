import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MeanReversionStrategy:
    """
    Comprehensive Mean Reversion Trading Strategy
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss: float = 3.0,
                 use_zscore: bool = True,
                 use_bollinger: bool = False,
                 use_rsi: bool = False):
        """
        Initialize mean reversion strategy parameters.
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.use_zscore = use_zscore
        self.use_bollinger = use_bollinger
        self.use_rsi = use_rsi
        
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """Calculate Z-score for mean reversion"""
        rolling_mean = prices.rolling(window=self.lookback_period).mean()
        rolling_std = prices.rolling(window=self.lookback_period).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=self.lookback_period).mean()
        rolling_std = prices.rolling(window=self.lookback_period).std()
        
        bb = pd.DataFrame(index=prices.index)
        bb['middle'] = rolling_mean
        bb['upper'] = rolling_mean + (rolling_std * self.entry_threshold)
        bb['lower'] = rolling_mean - (rolling_std * self.entry_threshold)
        bb['width'] = (bb['upper'] - bb['lower']) / bb['middle']
        bb['percent_b'] = (prices - bb['lower']) / (bb['upper'] - bb['lower'])
        
        return bb
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion indicators.
        """
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices.values if isinstance(prices, pd.Series) else prices
        
        # Calculate indicators
        if self.use_zscore:
            signals['zscore'] = self.calculate_zscore(prices)
        
        if self.use_bollinger:
            bb = self.calculate_bollinger_bands(prices)
            for col in bb.columns:
                signals[col] = bb[col]
        
        if self.use_rsi:
            signals['rsi'] = self.calculate_rsi(prices)
        
        # Generate entry signals
        signals['signal'] = 0
        signals['position'] = 0
        
        if self.use_zscore:
            # Long signal when zscore < -entry_threshold (oversold)
            # Short signal when zscore > entry_threshold (overbought)
            signals.loc[signals['zscore'] < -self.entry_threshold, 'signal'] = 1
            signals.loc[signals['zscore'] > self.entry_threshold, 'signal'] = -1
        
        elif self.use_bollinger:
            # Long when price crosses below lower band
            # Short when price crosses above upper band
            signals.loc[signals['price'] < signals['lower'], 'signal'] = 1
            signals.loc[signals['price'] > signals['upper'], 'signal'] = -1
        
        elif self.use_rsi:
            # Long when RSI < 30 (oversold)
            # Short when RSI > 70 (overbought)
            signals.loc[signals['rsi'] < 30, 'signal'] = 1
            signals.loc[signals['rsi'] > 70, 'signal'] = -1
        
        # Generate exit signals
        if self.use_zscore:
            # Exit when zscore crosses back towards zero
            exit_long = (signals['zscore'] > -self.exit_threshold) & (signals['signal'].shift(1) == 1)
            exit_short = (signals['zscore'] < self.exit_threshold) & (signals['signal'].shift(1) == -1)
            signals.loc[exit_long | exit_short, 'signal'] = 0
        
        return signals
    
    def backtest(self, 
                prices: pd.Series, 
                initial_capital: float = 10000,
                position_size: float = 0.1,
                commission: float = 0.001) -> Dict:
        """
        Backtest the mean reversion strategy.
        """
        signals = self.generate_signals(prices)
        
        # Calculate positions (cumulative signal)
        signals['position'] = signals['signal'].cumsum()
        
        # Ensure position is within bounds
        signals['position'] = signals['position'].clip(-1, 1)
        
        # Calculate returns
        signals['returns'] = prices.pct_change()
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        
        # Account for transaction costs
        signals['trades'] = signals['position'].diff().abs()
        signals['strategy_returns'] -= signals['trades'] * commission
        
        # Calculate equity curve
        signals['equity'] = initial_capital * (1 + signals['strategy_returns']).cumprod()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(signals, initial_capital)
        
        return {
            'signals': signals,
            'metrics': metrics,
            'trades': self.get_trades(signals, prices)
        }
    
    def calculate_performance_metrics(self, 
                                    signals: pd.DataFrame,
                                    initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        strategy_returns = signals['strategy_returns'].dropna()
        price_returns = signals['returns'].dropna()
        
        # Basic metrics
        total_return = (signals['equity'].iloc[-1] - initial_capital) / initial_capital * 100
        
        # Annualized return (assuming 252 trading days)
        days = len(signals)
        annualized_return = ((1 + total_return/100) ** (252/days) - 1) * 100 if days > 0 else 0
        
        # Volatility
        annualized_vol = strategy_returns.std() * np.sqrt(252) * 100 if len(strategy_returns) > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        if len(strategy_returns) > 0:
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100 if not pd.isna(drawdown.min()) else 0
        else:
            max_drawdown = 0
        
        # Win rate
        trades = signals['trades'].sum() / 2  # Each round trip
        if trades > 0:
            winning_trades = ((signals['strategy_returns'] > 0) & (signals['trades'] > 0)).sum()
            win_rate = winning_trades / trades * 100
        else:
            win_rate = 0
        
        # Alpha and Beta
        if len(price_returns) > 0 and len(strategy_returns) > 0:
            covariance = strategy_returns.cov(price_returns)
            variance = price_returns.var()
            beta = covariance / variance if variance > 0 else 0
            alpha = (strategy_returns.mean() - beta * price_returns.mean()) * 252 * 100
        else:
            beta = 0
            alpha = 0
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'annualized_volatility_pct': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'number_of_trades': int(trades),
            'alpha': alpha,
            'beta': beta
        }
    
    def get_trades(self, signals: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Extract individual trades from signals"""
        entry_dates = []
        exit_dates = []
        entry_prices = []
        exit_prices = []
        trade_types = []
        trade_pnl = []
        
        position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(len(signals)):
            if signals['signal'].iloc[i] != 0 and position == 0:
                # Enter trade
                position = signals['signal'].iloc[i]
                entry_price = prices.iloc[i]
                entry_date = signals.index[i]
                
            elif signals['signal'].iloc[i] == 0 and position != 0:
                # Exit trade
                exit_price = prices.iloc[i]
                exit_date = signals.index[i]
                
                # Calculate P&L
                if position > 0:  # Long trade
                    pnl = (exit_price - entry_price) / entry_price * 100
                else:  # Short trade
                    pnl = (entry_price - exit_price) / entry_price * 100
                
                entry_dates.append(entry_date)
                exit_dates.append(exit_date)
                entry_prices.append(entry_price)
                exit_prices.append(exit_price)
                trade_types.append('LONG' if position > 0 else 'SHORT')
                trade_pnl.append(pnl)
                
                position = 0
        
        # Create trades DataFrame
        trades_df = pd.DataFrame({
            'entry_date': entry_dates,
            'exit_date': exit_dates,
            'type': trade_types,
            'entry_price': entry_prices,
            'exit_price': exit_prices,
            'pnl_pct': trade_pnl
        })
        
        return trades_df


# Visualization and Analysis Class - FIXED VERSION
class StrategyVisualizer:
    """Helper class for visualizing mean reversion strategies"""
    
    def __init__(self, entry_threshold: float = 2.0):
        """Initialize visualizer with strategy parameters"""
        self.entry_threshold = entry_threshold
    
    def plot_signals(self, prices: pd.Series, signals: pd.DataFrame, title: str = "Mean Reversion Signals"):
        """Plot price with trading signals"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Ensure prices is a Series with proper index
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) == 1:
                prices = prices.iloc[:, 0]
            else:
                prices = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        
        # Price and signals
        axes[0].plot(prices.index, prices.values, label='Price', color='black', alpha=0.7, linewidth=1)
        
        # Buy signals
        buy_signals = signals[signals['signal'] == 1]
        if not buy_signals.empty:
            buy_prices = prices.loc[buy_signals.index]
            axes[0].scatter(buy_signals.index, buy_prices.values, 
                           color='green', marker='^', s=100, label='Buy', zorder=5, edgecolors='black')
        
        # Sell signals
        sell_signals = signals[signals['signal'] == -1]
        if not sell_signals.empty:
            sell_prices = prices.loc[sell_signals.index]
            axes[0].scatter(sell_signals.index, sell_prices.values, 
                           color='red', marker='v', s=100, label='Sell', zorder=5, edgecolors='black')
        
        # Exit signals
        exit_mask = (signals['signal'] == 0) & (signals['signal'].shift(1) != 0)
        exit_indices = signals[exit_mask].index
        if not exit_indices.empty:
            exit_prices = prices.loc[exit_indices]
            axes[0].scatter(exit_indices, exit_prices.values, 
                           color='blue', marker='o', s=50, label='Exit', alpha=0.5, zorder=5, edgecolors='black')
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Z-score if available
        if 'zscore' in signals.columns:
            axes[1].plot(signals.index, signals['zscore'], color='purple', label='Z-Score', linewidth=1)
            axes[1].axhline(y=self.entry_threshold, color='r', linestyle='--', alpha=0.5, 
                           label=f'Upper Threshold ({self.entry_threshold})')
            axes[1].axhline(y=-self.entry_threshold, color='g', linestyle='--', alpha=0.5, 
                           label=f'Lower Threshold (-{self.entry_threshold})')
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].fill_between(signals.index, -self.entry_threshold, self.entry_threshold, 
                                alpha=0.1, color='gray')
            axes[1].set_title('Z-Score', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Z-Score')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
        elif 'rsi' in signals.columns:
            axes[1].plot(signals.index, signals['rsi'], color='orange', label='RSI', linewidth=1)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
            axes[1].axhline(y=50, color='black', linestyle='-', alpha=0.3)
            axes[1].fill_between(signals.index, 30, 70, alpha=0.1, color='gray')
            axes[1].set_title('RSI', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('RSI')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
        elif 'upper' in signals.columns:
            axes[1].plot(signals.index, signals['upper'], color='r', linestyle='--', label='Upper Band', alpha=0.7)
            axes[1].plot(signals.index, signals['middle'], color='b', label='Middle Band', alpha=0.7)
            axes[1].plot(signals.index, signals['lower'], color='g', linestyle='--', label='Lower Band', alpha=0.7)
            axes[1].fill_between(signals.index, signals['lower'], signals['upper'], alpha=0.1, color='gray')
            axes[1].set_title('Bollinger Bands', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Price')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        # Position
        axes[2].plot(signals.index, signals['position'], color='blue', drawstyle='steps-post', linewidth=2)
        axes[2].fill_between(signals.index, 0, signals['position'], alpha=0.3, step='post')
        axes[2].set_title('Position', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_equity_curve(self, signals: pd.DataFrame, benchmark: Optional[pd.Series] = None):
        """Plot equity curve with benchmark comparison"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Equity curve
        axes[0].plot(signals.index, signals['equity'], label='Strategy', color='blue', linewidth=2)
        
        if benchmark is not None:
            benchmark_equity = 10000 * (1 + benchmark.pct_change()).cumprod()
            axes[0].plot(benchmark.index, benchmark_equity, label='Benchmark (Buy & Hold)', 
                        color='gray', alpha=0.7, linewidth=1.5)
        
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        if 'strategy_returns' in signals.columns:
            cumulative = (1 + signals['strategy_returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            axes[1].fill_between(signals.index, 0, drawdown, color='red', alpha=0.3)
            axes[1].plot(signals.index, drawdown, color='red', linewidth=1)
            axes[1].set_title('Drawdown (%)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame):
        """Plot trade analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if trades_df.empty:
            print("No trades to analyze")
            return
        
        # Trade P&L distribution
        axes[0, 0].hist(trades_df['pnl_pct'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('P&L (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative P&L
        trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
        axes[0, 1].plot(range(len(trades_df)), trades_df['cumulative_pnl'], 'b-', linewidth=2)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Trade type performance
        long_trades = trades_df[trades_df['type'] == 'LONG']['pnl_pct']
        short_trades = trades_df[trades_df['type'] == 'SHORT']['pnl_pct']
        
        axes[1, 0].boxplot([long_trades, short_trades], labels=['LONG', 'SHORT'])
        axes[1, 0].set_title('Performance by Trade Type', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('P&L (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win/Loss ratio
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        losses = len(trades_df[trades_df['pnl_pct'] < 0])
        axes[1, 1].pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                       colors=['green', 'red'], explode=(0.05, 0))
        axes[1, 1].set_title(f'Win/Loss Ratio (Total: {len(trades_df)})', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# Function to get price data correctly
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Safely get price data from yfinance"""
    import yfinance as yf
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Handle different data structures
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Try to get Adjusted Close first, then Close
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    elif isinstance(data.columns, pd.MultiIndex):
        # Handle MultiIndex columns
        if ('Adj Close', ticker) in data.columns:
            prices = data[('Adj Close', ticker)]
        elif ('Close', ticker) in data.columns:
            prices = data[('Close', ticker)]
        else:
            # Take the first price column
            price_cols = [col for col in data.columns if col[0] in ['Adj Close', 'Close']]
            if price_cols:
                prices = data[price_cols[0]]
            else:
                prices = data.iloc[:, 0]
    else:
        # Take the first column if it's the only one
        prices = data.iloc[:, 0] if data.shape[1] == 1 else data['Close']
    
    # Ensure we return a Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    return prices


# Example Usage
if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("MEAN REVERSION STRATEGY BACKTEST")
    print("=" * 60)
    
    # Download sample data
    ticker = "AAPL"
    try:
        prices = get_price_data(ticker, "2020-01-01", "2023-12-31")
        print(f"\nSuccessfully loaded {len(prices)} days of data for {ticker}")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Price range: ${prices.min():.2f} to ${prices.max():.2f}")
        
        # Create and test strategy
        strategy = MeanReversionStrategy(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss=3.0,
            use_zscore=True
        )
        
        # Generate signals
        signals = strategy.generate_signals(prices)
        signal_count = len(signals[signals['signal'] != 0])
        print(f"\nGenerated {signal_count} signals")
        
        # Backtest
        results = strategy.backtest(prices, initial_capital=10000, commission=0.001)
        
        # Print metrics
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric:25}: {value:,.2f}")
            else:
                print(f"{metric:25}: {value}")
        
        # Show trades
        print("\n" + "=" * 60)
        print("RECENT TRADES")
        print("=" * 60)
        trades_df = results['trades']
        if not trades_df.empty:
            print(trades_df.tail(10).to_string(index=False))
            print(f"\nTotal Trades: {len(trades_df)}")
            print(f"Win Rate: {(trades_df['pnl_pct'] > 0).mean() * 100:.1f}%")
            print(f"Avg Win: {trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean():.2f}%")
            print(f"Avg Loss: {trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean():.2f}%")
        
        # Visualize - FIXED: Create visualizer instance with entry_threshold
        visualizer = StrategyVisualizer(entry_threshold=strategy.entry_threshold)
        visualizer.plot_signals(prices, results['signals'], f"{ticker} - Mean Reversion Strategy")
        visualizer.plot_equity_curve(results['signals'])
        visualizer.plot_trade_analysis(trades_df)
        
        # Parameter sensitivity analysis
        print("\n" + "=" * 60)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 60)
        
        thresholds = [1.5, 2.0, 2.5, 3.0]
        lookbacks = [10, 20, 30, 50]
        
        sensitivity_results = []
        
        for threshold in thresholds:
            for lookback in lookbacks:
                test_strategy = MeanReversionStrategy(
                    lookback_period=lookback,
                    entry_threshold=threshold,
                    exit_threshold=threshold * 0.25,
                    use_zscore=True
                )
                test_results = test_strategy.backtest(prices, initial_capital=10000, commission=0.001)
                sensitivity_results.append({
                    'threshold': threshold,
                    'lookback': lookback,
                    'sharpe': test_results['metrics']['sharpe_ratio'],
                    'return': test_results['metrics']['total_return_pct'],
                    'trades': test_results['metrics']['number_of_trades']
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        print("\nTop 5 combinations by Sharpe Ratio:")
        print(sensitivity_df.nlargest(5, 'sharpe')[['threshold', 'lookback', 'sharpe', 'return', 'trades']].to_string(index=False))
        
        # Plot sensitivity heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_table = sensitivity_df.pivot(index='lookback', columns='threshold', values='sharpe')
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Sharpe Ratio by Parameter Combination', fontsize=14, fontweight='bold')
        ax.set_xlabel('Entry Threshold')
        ax.set_ylabel('Lookback Period')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()