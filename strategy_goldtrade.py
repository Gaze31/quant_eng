"""
Gold Trading Strategy System
Multiple strategy implementations with backtesting and risk management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldTradingStrategy:
    """Base class for gold trading strategies"""
    
    def __init__(self, initial_capital=10000, position_size=0.1, stop_loss=0.02, take_profit=0.05):
        """
        Initialize strategy parameters
        
        Args:
            initial_capital: Starting capital in USD
            position_size: Fraction of capital to use per trade (0.1 = 10%)
            stop_loss: Stop loss percentage (0.02 = 2%)
            take_profit: Take profit percentage (0.05 = 5%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.portfolio_value = []
        
    def calculate_position_size(self, price, capital):
        """Calculate how many ounces to buy"""
        trade_amount = capital * self.position_size
        ounces = trade_amount / price
        return ounces, trade_amount
    
    def execute_trade(self, date, price, signal, capital):
        """Execute a trade based on signal"""
        if signal == 'BUY':
            ounces, cost = self.calculate_position_size(price, capital)
            return {
                'date': date,
                'type': 'BUY',
                'price': price,
                'ounces': ounces,
                'cost': cost,
                'stop_loss': price * (1 - self.stop_loss),
                'take_profit': price * (1 + self.take_profit)
            }
        return None
    
    def check_exit(self, current_price, trade):
        """Check if trade should be exited"""
        if current_price <= trade['stop_loss']:
            return 'STOP_LOSS'
        elif current_price >= trade['take_profit']:
            return 'TAKE_PROFIT'
        return None


class MovingAverageCrossover(GoldTradingStrategy):
    """
    Strategy 1: Moving Average Crossover
    Buy when short MA crosses above long MA (Golden Cross)
    Sell when short MA crosses below long MA (Death Cross)
    """
    
    def __init__(self, short_window=20, long_window=50, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Crossover ({short_window}/{long_window})"
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = df.copy()
        
        # Calculate moving averages
        signals['SMA_Short'] = signals['Close'].rolling(window=self.short_window).mean()
        signals['SMA_Long'] = signals['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['Signal'] = 0
        signals['Signal'][self.short_window:] = np.where(
            signals['SMA_Short'][self.short_window:] > signals['SMA_Long'][self.short_window:], 
            1, 0
        )
        
        # Generate trading orders (1 = buy, -1 = sell, 0 = hold)
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class RSIStrategy(GoldTradingStrategy):
    """
    Strategy 2: RSI (Relative Strength Index)
    Buy when RSI < 30 (oversold)
    Sell when RSI > 70 (overbought)
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI Strategy ({rsi_period})"
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = df.copy()
        
        # Calculate RSI
        signals['RSI'] = self.calculate_rsi(signals['Close'], self.rsi_period)
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['RSI'] < self.oversold, 'Signal'] = 1  # Buy
        signals.loc[signals['RSI'] > self.overbought, 'Signal'] = -1  # Sell
        
        # Generate positions
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class BollingerBandsStrategy(GoldTradingStrategy):
    """
    Strategy 3: Bollinger Bands
    Buy when price touches lower band
    Sell when price touches upper band
    """
    
    def __init__(self, bb_period=20, num_std=2, **kwargs):
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.num_std = num_std
        self.name = f"Bollinger Bands ({bb_period}, {num_std}σ)"
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = df.copy()
        
        # Calculate Bollinger Bands
        signals['BB_Middle'] = signals['Close'].rolling(window=self.bb_period).mean()
        signals['BB_Std'] = signals['Close'].rolling(window=self.bb_period).std()
        signals['BB_Upper'] = signals['BB_Middle'] + (signals['BB_Std'] * self.num_std)
        signals['BB_Lower'] = signals['BB_Middle'] - (signals['BB_Std'] * self.num_std)
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['Close'] <= signals['BB_Lower'], 'Signal'] = 1  # Buy
        signals.loc[signals['Close'] >= signals['BB_Upper'], 'Signal'] = -1  # Sell
        
        # Generate positions
        signals['Position'] = signals['Signal'].diff()
        
        return signals


class MACDStrategy(GoldTradingStrategy):
    """
    Strategy 4: MACD (Moving Average Convergence Divergence)
    Buy when MACD crosses above signal line
    Sell when MACD crosses below signal line
    """
    
    def __init__(self, fast=12, slow=26, signal=9, **kwargs):
        super().__init__(**kwargs)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD ({fast}/{slow}/{signal})"
    
    def calculate_macd(self, prices):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        return macd, signal_line
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = df.copy()
        
        # Calculate MACD
        signals['MACD'], signals['Signal_Line'] = self.calculate_macd(signals['Close'])
        signals['MACD_Histogram'] = signals['MACD'] - signals['Signal_Line']
        
        # Generate signals
        signals['Signal'] = 0
        signals['Signal'][1:] = np.where(
            (signals['MACD'][1:] > signals['Signal_Line'][1:]) & 
            (signals['MACD'][:-1].values <= signals['Signal_Line'][:-1].values), 
            1, 0
        )
        signals['Signal'][1:] = np.where(
            (signals['MACD'][1:] < signals['Signal_Line'][1:]) & 
            (signals['MACD'][:-1].values >= signals['Signal_Line'][:-1].values), 
            -1, signals['Signal'][1:]
        )
        
        # Generate positions
        signals['Position'] = signals['Signal']
        
        return signals


class TrendFollowingStrategy(GoldTradingStrategy):
    """
    Strategy 5: Trend Following with ADX
    Uses ADX to identify strong trends and trades in direction of trend
    """
    
    def __init__(self, adx_period=14, adx_threshold=25, **kwargs):
        super().__init__(**kwargs)
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.name = f"Trend Following (ADX {adx_period})"
    
    def calculate_adx(self, df):
        """Calculate ADX (Average Directional Index)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate smoothed indicators
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=self.adx_period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx, plus_di, minus_di
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = df.copy()
        
        # Calculate ADX and directional indicators
        signals['ADX'], signals['Plus_DI'], signals['Minus_DI'] = self.calculate_adx(signals)
        
        # Calculate trend
        signals['Trend'] = signals['Close'].rolling(window=20).mean()
        
        # Generate signals (only trade when ADX shows strong trend)
        signals['Signal'] = 0
        strong_trend = signals['ADX'] > self.adx_threshold
        
        signals.loc[strong_trend & (signals['Plus_DI'] > signals['Minus_DI']), 'Signal'] = 1  # Buy
        signals.loc[strong_trend & (signals['Plus_DI'] < signals['Minus_DI']), 'Signal'] = -1  # Sell
        
        # Generate positions
        signals['Position'] = signals['Signal'].diff()
        
        return signals


def backtest_strategy(strategy, df):
    """
    Backtest a trading strategy
    """
    signals = strategy.generate_signals(df)
    
    capital = strategy.initial_capital
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = [capital]
    
    for i in range(len(signals)):
        current_price = signals['Close'].iloc[i]
        current_date = signals.index[i]
        
        # Check if we have a position
        if position > 0:
            # Check stop loss / take profit
            pnl_pct = (current_price - entry_price) / entry_price
            
            if current_price <= entry_price * (1 - strategy.stop_loss):
                # Stop loss hit
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'ounces': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': 'STOP_LOSS'
                })
                
                position = 0
                
            elif current_price >= entry_price * (1 + strategy.take_profit):
                # Take profit hit
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                capital += pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'ounces': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': 'TAKE_PROFIT'
                })
                
                position = 0
        
        # Check for new signals
        if 'Position' in signals.columns and i > 0:
            if signals['Position'].iloc[i] == 1 and position == 0:  # Buy signal
                ounces, cost = strategy.calculate_position_size(current_price, capital)
                capital -= cost
                position = ounces
                entry_price = current_price
                entry_date = current_date
                
            elif signals['Position'].iloc[i] == -1 and position > 0:  # Sell signal
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                capital += exit_value
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'ounces': position,
                    'pnl': pnl,
                    'pnl_pct': ((current_price - entry_price) / entry_price) * 100,
                    'exit_reason': 'SIGNAL'
                })
                
                position = 0
        
        # Calculate current portfolio value
        current_value = capital + (position * current_price if position > 0 else 0)
        portfolio_values.append(current_value)
    
    # Close any open positions
    if position > 0:
        current_price = signals['Close'].iloc[-1]
        exit_value = position * current_price
        pnl = exit_value - (position * entry_price)
        capital += exit_value
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': signals.index[-1],
            'entry_price': entry_price,
            'exit_price': current_price,
            'ounces': position,
            'pnl': pnl,
            'pnl_pct': ((current_price - entry_price) / entry_price) * 100,
            'exit_reason': 'END_OF_PERIOD'
        })
    
    return {
        'trades': trades,
        'portfolio_values': portfolio_values,
        'final_capital': capital,
        'signals': signals
    }


def calculate_performance_metrics(results, initial_capital):
    """Calculate strategy performance metrics"""
    trades = results['trades']
    portfolio_values = results['portfolio_values']
    
    if len(trades) == 0:
        return {
            'total_return': 0,
            'total_return_pct': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    trades_df = pd.DataFrame(trades)
    
    # Basic metrics
    total_return = results['final_capital'] - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    # Win rate
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    win_rate = (num_wins / len(trades)) * 100 if len(trades) > 0 else 0
    
    # Average win/loss
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    # Profit factor
    total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    # Max drawdown
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Sharpe ratio (simplified - assuming daily returns)
    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_wins': num_wins,
        'num_losses': num_losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade_pnl': trades_df['pnl'].mean(),
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'best_trade': trades_df['pnl'].max(),
        'worst_trade': trades_df['pnl'].min()
    }


def generate_sample_data(days=365):
    """Generate sample gold price data"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    actual_days = len(dates)
    
    base_price = 2000
    trend = np.linspace(0, 150, actual_days)
    seasonal = 30 * np.sin(np.linspace(0, 4*np.pi, actual_days))
    noise = np.random.normal(0, 15, actual_days)
    
    close_prices = base_price + trend + seasonal + noise
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.normal(0, 5, actual_days),
        'High': close_prices + abs(np.random.normal(10, 5, actual_days)),
        'Low': close_prices - abs(np.random.normal(10, 5, actual_days)),
        'Close': close_prices,
        'Volume': np.random.randint(100000, 500000, actual_days)
    }, index=dates)
    
    return df


def plot_strategy_comparison(strategies_results, df):
    """Plot comparison of all strategies"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Portfolio values over time
    ax1 = axes[0]
    for strategy_name, results in strategies_results.items():
        portfolio_values = results['portfolio_values']
        dates = pd.date_range(start=df.index[0], periods=len(portfolio_values), freq='D')
        ax1.plot(dates, portfolio_values, label=strategy_name, linewidth=2)
    
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Strategy Performance Comparison - Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns comparison
    ax2 = axes[1]
    strategy_names = list(strategies_results.keys())
    returns = [calculate_performance_metrics(results, 10000)['total_return_pct'] 
               for results in strategies_results.values()]
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax2.bar(strategy_names, returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Total Return by Strategy (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 3: Win rate comparison
    ax3 = axes[2]
    win_rates = [calculate_performance_metrics(results, 10000)['win_rate'] 
                 for results in strategies_results.values()]
    
    bars = ax3.bar(strategy_names, win_rates, color='steelblue', alpha=0.7)
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50% Benchmark')
    ax3.set_title('Win Rate by Strategy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('gold_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Strategy comparison chart saved as 'gold_strategy_comparison.png'")
    
    return fig


def main():
    """Main execution function"""
    print("=" * 80)
    print(" " * 25 + "GOLD TRADING STRATEGY SYSTEM")
    print("=" * 80)
    
    # Generate sample data
    print("\n📊 Loading historical gold price data...")
    df = generate_sample_data(days=365)
    print(f"✓ Loaded {len(df)} days of data")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Initialize strategies
    initial_capital = 10000
    
    strategies = [
        MovingAverageCrossover(short_window=20, long_window=50, initial_capital=initial_capital),
        RSIStrategy(rsi_period=14, initial_capital=initial_capital),
        BollingerBandsStrategy(bb_period=20, initial_capital=initial_capital),
        MACDStrategy(fast=12, slow=26, signal=9, initial_capital=initial_capital),
        TrendFollowingStrategy(adx_period=14, initial_capital=initial_capital)
    ]
    
    print(f"\n🎯 Testing {len(strategies)} different strategies...")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Position Size: 10% of capital")
    print(f"   Stop Loss: 2%")
    print(f"   Take Profit: 5%")
    
    # Backtest all strategies
    strategies_results = {}
    
    for strategy in strategies:
        print(f"\n   Testing: {strategy.name}...", end=' ')
        results = backtest_strategy(strategy, df)
        strategies_results[strategy.name] = results
        metrics = calculate_performance_metrics(results, initial_capital)
        print(f"✓ ({metrics['num_trades']} trades)")
    
    # Display results
    print("\n" + "=" * 80)
    print(" " * 30 + "STRATEGY RESULTS")
    print("=" * 80)
    
    results_data = []
    
    for strategy_name, results in strategies_results.items():
        metrics = calculate_performance_metrics(results, initial_capital)
        results_data.append({
            'Strategy': strategy_name,
            'Final Value': f"${metrics['total_return'] + initial_capital:,.2f}",
            'Return': f"{metrics['total_return_pct']:.2f}%",
            'Trades': metrics['num_trades'],
            'Win Rate': f"{metrics['win_rate']:.1f}%",
            'Profit Factor': f"{metrics['profit_factor']:.2f}",
            'Max DD': f"{metrics['max_drawdown']:.2f}%",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}"
        })
    
    results_df = pd.DataFrame(results_data)
    print("\n" + results_df.to_string(index=False))
    
    # Find best strategy
    best_strategy = max(strategies_results.items(), 
                       key=lambda x: calculate_performance_metrics(x[1], initial_capital)['total_return_pct'])
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST PERFORMING STRATEGY: {best_strategy[0]}")
    print("=" * 80)
    
    best_metrics = calculate_performance_metrics(best_strategy[1], initial_capital)
    
    print(f"\nPerformance Summary:")
    print(f"  Initial Capital:        ${initial_capital:,.2f}")
    print(f"  Final Value:            ${best_metrics['total_return'] + initial_capital:,.2f}")
    print(f"  Total Return:           ${best_metrics['total_return']:,.2f} ({best_metrics['total_return_pct']:.2f}%)")
    print(f"\nTrading Statistics:")
    print(f"  Total Trades:           {best_metrics['num_trades']}")
    
    if best_metrics['num_trades'] > 0:
        print(f"  Winning Trades:         {best_metrics['num_wins']}")
        print(f"  Losing Trades:          {best_metrics['num_losses']}")
        print(f"  Win Rate:               {best_metrics['win_rate']:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"  Average Win:            ${best_metrics['avg_win']:.2f}")
        print(f"  Average Loss:           ${best_metrics['avg_loss']:.2f}")
        print(f"  Profit Factor:          {best_metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown:           {best_metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:           {best_metrics['sharpe_ratio']:.2f}")
        print(f"\nBest/Worst Trades:")
        print(f"  Best Trade:             ${best_metrics['best_trade']:.2f}")
        print(f"  Worst Trade:            ${best_metrics['worst_trade']:.2f}")
    else:
        print(f"  No trades executed - conditions not met during backtest period")
    
    # Show recent trades from best strategy
    if len(best_strategy[1]['trades']) > 0:
        print("\n" + "=" * 80)
        print("RECENT TRADES (Last 5)")
        print("=" * 80)
        
        recent_trades = pd.DataFrame(best_strategy[1]['trades']).tail(5)
        print(recent_trades[['entry_date', 'exit_date', 'entry_price', 'exit_price', 
                            'pnl', 'pnl_pct', 'exit_reason']].to_string(index=False))
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("📊 Generating visualizations...")
    plot_strategy_comparison(strategies_results, df)
    
    # Export results
    print("\n📁 Exporting results...")
    results_df.to_csv('strategy_comparison.csv', index=False)
    print("✓ Strategy comparison saved to 'strategy_comparison.csv'")
    
    # Export best strategy trades
    if len(best_strategy[1]['trades']) > 0:
        trades_df = pd.DataFrame(best_strategy[1]['trades'])
        trades_df.to_csv('best_strategy_trades.csv', index=False)
        print("✓ Trade history saved to 'best_strategy_trades.csv'")
    
    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 80)
    print("""
Past performance does NOT guarantee future results
These strategies are backtested on historical data and may not work in live markets
Market conditions change constantly
Always use proper risk management
Never risk more than you can afford to lose
This is for EDUCATIONAL PURPOSES ONLY - NOT financial advice
Consult a licensed financial advisor before trading
    """)
    print("=" * 80)
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()