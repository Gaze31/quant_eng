"""
Algorithmic Trading Strategy Simulator
A comprehensive framework for backtesting multiple trading strategies
Fixed version for yfinance data handling
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================================
# TECHNICAL INDICATORS (Manual implementation)
# ============================================================================

def calculate_sma(series, window):
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=window).mean()

def calculate_ema(series, window):
    """Exponential Moving Average"""
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Bollinger Bands"""
    sma = calculate_sma(series, window)
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_atr(high, low, close, window=14):
    """Average True Range"""
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_stochastic(high, low, close, window=14, smooth_k=3, smooth_d=3):
    """Stochastic Oscillator"""
    low_min = low.rolling(window=window).min()
    high_max = high.rolling(window=window).max()
    
    # %K
    stoch_k = 100 * ((close - low_min) / (high_max - low_min))
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    
    # %D
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    
    return stoch_k, stoch_d

def calculate_obv(close, volume):
    """On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_volume_ratio(volume, window=20):
    """Volume Ratio (current volume / average volume)"""
    volume_sma = calculate_sma(volume, window)
    volume_ratio = volume / volume_sma
    return volume_ratio

# ============================================================================
# MAIN TRADING STRATEGY CLASS
# ============================================================================

class AlgoTradingStrategy:
    """
    Comprehensive algorithmic trading strategy framework
    """
    
    def __init__(self, symbol, start_date, end_date, initial_capital=100000):
        """
        Initialize the trading strategy system
        
        Parameters:
        symbol: str - Trading symbol
        start_date: str - Start date for data
        end_date: str - End date for data
        initial_capital: float - Initial capital for backtesting
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.results = None
        
    def fetch_data(self):
        """
        Fetch historical data and calculate technical indicators
        """
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
        
        try:
            # Download data
            raw_data = yf.download(self.symbol, start=self.start_date, 
                                   end=self.end_date, progress=False, auto_adjust=False)
            
            if raw_data.empty:
                # Try with different date range
                print(f"No data found for {self.symbol}, trying with expanded date range...")
                raw_data = yf.download(self.symbol, start='2023-01-01', 
                                      end='2024-12-31', progress=False, auto_adjust=False)
            
            # Handle MultiIndex columns
            if isinstance(raw_data.columns, pd.MultiIndex):
                print("Detected MultiIndex columns, flattening...")
                # Extract the symbol level
                symbols = raw_data.columns.get_level_values(1).unique()
                self.data = pd.DataFrame()
                
                # Get the first symbol (should be our ticker)
                for col in raw_data.columns:
                    if col[1] == self.symbol or len(symbols) == 1:
                        self.data[col[0]] = raw_data[col]
                
                # Rename columns to standard names
                self.data.columns = [col.split(',')[0] if ',' in col else col for col in self.data.columns]
            else:
                self.data = raw_data.copy()
            
            # Ensure we have the required columns
            required_cols = ['Adj Close', 'Close', 'High', 'Low', 'Volume']
            available_cols = self.data.columns.tolist()
            
            # Map available columns
            column_mapping = {}
            for col in required_cols:
                if col in available_cols:
                    column_mapping[col] = col
                elif 'Close' in col and 'Adj' not in col and 'Adj Close' not in available_cols:
                    column_mapping['Adj Close'] = col
                elif 'High' in col and 'High' not in column_mapping:
                    column_mapping['High'] = col
                elif 'Low' in col and 'Low' not in column_mapping:
                    column_mapping['Low'] = col
                elif 'Volume' in col and 'Volume' not in column_mapping:
                    column_mapping['Volume'] = col
            
            # Rename columns for consistency
            self.data = self.data.rename(columns=column_mapping)
            
            # If Adj Close doesn't exist, use Close
            if 'Adj Close' not in self.data.columns and 'Close' in self.data.columns:
                self.data['Adj Close'] = self.data['Close']
            
            # Check if we have all required columns
            missing_cols = [col for col in ['Adj Close', 'High', 'Low', 'Volume'] 
                           if col not in self.data.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Try to infer missing columns
                if 'High' not in self.data.columns and 'Close' in self.data.columns:
                    self.data['High'] = self.data['Close'] * 1.02
                if 'Low' not in self.data.columns and 'Close' in self.data.columns:
                    self.data['Low'] = self.data['Close'] * 0.98
                if 'Volume' not in self.data.columns:
                    self.data['Volume'] = 1000000  # Default volume
            
            # Drop any rows with NaN values in essential columns
            self.data = self.data.dropna(subset=['Adj Close', 'High', 'Low'])
            
            print(f"Successfully downloaded {len(self.data)} days of data")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"Columns: {self.data.columns.tolist()}")
            
            # Basic calculations
            self.data['Returns'] = self.data['Adj Close'].pct_change()
            self.data['Log_Returns'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift(1))
            
            # Calculate technical indicators
            self.calculate_technical_indicators()
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Create sample data for demonstration
            print("Creating sample data for demonstration...")
            self.create_sample_data()
            return self.data
    
    def create_sample_data(self):
        """Create sample data for demonstration when download fails"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        np.random.seed(42)
        
        # Generate random walk price
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'Adj Close': price,
            'Close': price,
            'High': price * (1 + np.random.uniform(0, 0.02, len(dates))),
            'Low': price * (1 - np.random.uniform(0, 0.02, len(dates))),
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Open': price * (1 + np.random.uniform(-0.01, 0.01, len(dates)))
        }, index=dates)
        
        self.data['Returns'] = self.data['Adj Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift(1))
        self.calculate_technical_indicators()
        
        print(f"Created {len(self.data)} days of sample data")
    
    def calculate_technical_indicators(self):
        """
        Calculate various technical indicators for analysis
        """
        close = self.data['Adj Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']
        
        # Moving Averages
        self.data['SMA_20'] = calculate_sma(close, 20)
        self.data['SMA_50'] = calculate_sma(close, 50)
        self.data['SMA_200'] = calculate_sma(close, 200)
        self.data['EMA_12'] = calculate_ema(close, 12)
        self.data['EMA_26'] = calculate_ema(close, 26)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram
        
        # RSI
        self.data['RSI'] = calculate_rsi(close, 14)
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(close, 20, 2)
        self.data['BB_Upper'] = upper
        self.data['BB_Middle'] = middle
        self.data['BB_Lower'] = lower
        self.data['BB_Width'] = (upper - lower) / middle
        self.data['BB_Position'] = (close - lower) / (upper - lower)
        
        # ATR for volatility
        self.data['ATR'] = calculate_atr(high, low, close, 14)
        self.data['ATR_Pct'] = self.data['ATR'] / close * 100
        
        # Volume indicators
        self.data['Volume_SMA'] = calculate_sma(volume, 20)
        self.data['Volume_Ratio'] = volume / self.data['Volume_SMA']
        self.data['OBV'] = calculate_obv(close, volume)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        self.data['Stoch_K'] = stoch_k
        self.data['Stoch_D'] = stoch_d
        
        # Price patterns
        self.data['High_Low_Ratio'] = high / low
        self.data['Close_Position'] = (close - low) / (high - low)
        self.data['Price_ROC'] = close.pct_change(periods=5)  # Rate of change
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        print(f"Calculated {len(self.data.columns)} indicators")
    
    # [Rest of the methods remain the same as before...]
    def trend_following_strategy(self, fast_ma=20, slow_ma=50):
        """Trend following strategy using moving average crossover"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Adj Close']
        
        # Generate signals
        signals['fast_ma'] = self.data[f'SMA_{fast_ma}']
        signals['slow_ma'] = self.data[f'SMA_{slow_ma}']
        
        # Initial signal
        signals['signal'] = 0
        signals.loc[signals['fast_ma'] > signals['slow_ma'], 'signal'] = 1
        signals.loc[signals['fast_ma'] <= signals['slow_ma'], 'signal'] = -1
        
        # Add confidence score based on distance between MAs
        signals['ma_distance'] = (signals['fast_ma'] - signals['slow_ma']) / signals['slow_ma']
        signals['confidence'] = abs(signals['ma_distance']) * 100
        
        return signals
    
    def mean_reversion_strategy(self, lookback=20, oversold_threshold=30, overbought_threshold=70):
        """Mean reversion strategy using Bollinger Bands and RSI"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Adj Close']
        
        # Get indicators
        signals['rsi'] = self.data['RSI']
        signals['bb_upper'] = self.data['BB_Upper']
        signals['bb_lower'] = self.data['BB_Lower']
        signals['bb_middle'] = self.data['BB_Middle']
        signals['bb_position'] = self.data['BB_Position']
        
        # Generate signals
        signals['signal'] = 0
        
        # Buy signal: Price below lower band AND RSI oversold
        buy_condition = (signals['price'] <= signals['bb_lower']) & \
                        (signals['rsi'] < oversold_threshold)
        signals.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: Price above upper band AND RSI overbought
        sell_condition = (signals['price'] >= signals['bb_upper']) & \
                         (signals['rsi'] > overbought_threshold)
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add confidence score based on how extreme the position is
        signals['confidence'] = abs(0.5 - signals['bb_position']) * 200
        
        return signals
    
    def momentum_strategy(self, momentum_period=20, volume_threshold=1.2):
        """Momentum strategy based on price momentum and volume confirmation"""
        signals = pd.DataFrame(index=self.data.index)
        signals['price'] = self.data['Adj Close']
        
        # Calculate momentum
        signals['momentum'] = self.data['Returns'].rolling(window=momentum_period).sum()
        signals['volume_ratio'] = self.data['Volume_Ratio']
        signals['macd'] = self.data['MACD']
        signals['macd_signal'] = self.data['MACD_Signal']
        signals['macd_histogram'] = self.data['MACD_Histogram']
        signals['rsi_momentum'] = self.data['RSI'].diff(3)
        
        # Generate signals
        signals['signal'] = 0
        
        # Buy conditions: Positive momentum, MACD bullish, volume confirmation
        buy_condition = (signals['momentum'] > 0) & \
                        (signals['macd'] > signals['macd_signal']) & \
                        (signals['volume_ratio'] > volume_threshold) & \
                        (signals['rsi_momentum'] > 0)
        signals.loc[buy_condition, 'signal'] = 1
        
        # Sell conditions: Negative momentum, MACD bearish, volume confirmation
        sell_condition = (signals['momentum'] < 0) & \
                         (signals['macd'] < signals['macd_signal']) & \
                         (signals['volume_ratio'] > volume_threshold) & \
                         (signals['rsi_momentum'] < 0)
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add confidence score
        signals['confidence'] = abs(signals['momentum'] * 100) * signals['volume_ratio']
        
        return signals
    
    def ml_strategy(self, features=None, test_size=0.3):
        """Machine Learning based strategy using Random Forest"""
        if features is None:
            features = ['RSI', 'MACD', 'BB_Width', 'Volume_Ratio', 
                       'Stoch_K', 'ATR_Pct', 'High_Low_Ratio', 'Close_Position',
                       'Price_ROC', 'OBV']
        
        # Prepare data
        ml_data = self.data[features].copy()
        
        # Create target variable (1 if next 5-day return > 0, 0 otherwise)
        future_returns = self.data['Returns'].shift(-5).rolling(5).sum()
        ml_data['target'] = (future_returns > 0.02).astype(int)  # 2% threshold
        
        # Drop NaN values
        ml_data = ml_data.dropna()
        
        if len(ml_data) < 100:
            print("Warning: Not enough data for ML strategy")
            return pd.DataFrame(), None, None, None
        
        # Split data
        X = ml_data[features]
        y = ml_data['target']
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                      random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Generate signals for all data
        X_all_scaled = scaler.transform(X)
        predictions = model.predict(X_all_scaled)
        probabilities = model.predict_proba(X_all_scaled)[:, 1]
        
        signals = pd.DataFrame(index=ml_data.index)
        signals['price'] = self.data.loc[ml_data.index, 'Adj Close']
        signals['signal'] = predictions * 2 - 1  # Convert 0/1 to -1/1
        signals['confidence'] = abs(probabilities - 0.5) * 200
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return signals, model, scaler, feature_importance
    
    def backtest_strategy(self, signals, strategy_name, 
                         stop_loss=None, take_profit=None,
                         transaction_cost=0.001):
        """Backtest a trading strategy with realistic assumptions"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {strategy_name}")
        print('='*60)
        
        # Create backtest DataFrame
        backtest = signals.copy()
        backtest['returns'] = self.data['Returns']
        
        # Calculate strategy returns with transaction costs
        backtest['position_changed'] = backtest['signal'].diff().fillna(0) != 0
        backtest['strategy_returns'] = backtest['signal'].shift(1) * backtest['returns']
        
        # Apply transaction costs when position changes
        backtest.loc[backtest['position_changed'], 'strategy_returns'] -= transaction_cost
        
        # Apply stop loss and take profit if specified
        if stop_loss is not None or take_profit is not None:
            backtest = self.apply_risk_management(backtest, stop_loss, take_profit)
        
        # Calculate equity curves
        backtest['cumulative_returns'] = (1 + backtest['strategy_returns']).cumprod()
        backtest['buy_hold_returns'] = (1 + backtest['returns']).cumprod()
        backtest['strategy_value'] = self.initial_capital * backtest['cumulative_returns']
        
        # Calculate metrics
        metrics = self.calculate_metrics(backtest['strategy_returns'].dropna(), 
                                        backtest['returns'].dropna())
        
        self.results = {
            'backtest': backtest,
            'metrics': metrics,
            'strategy_name': strategy_name
        }
        
        # Print summary
        self.print_backtest_summary(backtest, metrics)
        
        return backtest, metrics
    
    def apply_risk_management(self, backtest, stop_loss, take_profit):
        """Apply stop loss and take profit rules"""
        position = 0
        entry_price = 0
        entry_index = 0
        
        for i in range(1, len(backtest)):
            current_signal = backtest['signal'].iloc[i]
            
            # Enter position
            if current_signal != 0 and position == 0:
                position = current_signal
                entry_price = backtest['price'].iloc[i]
                entry_index = i
            
            # Manage existing position
            elif position != 0:
                current_price = backtest['price'].iloc[i]
                returns_since_entry = (current_price - entry_price) / entry_price * position
                
                # Check stop loss
                if stop_loss and returns_since_entry < -stop_loss:
                    backtest.loc[backtest.index[i], 'signal'] = 0
                    position = 0
                
                # Check take profit
                elif take_profit and returns_since_entry > take_profit:
                    backtest.loc[backtest.index[i], 'signal'] = 0
                    position = 0
        
        return backtest
    
    def calculate_metrics(self, strategy_returns, benchmark_returns):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Return metrics
        metrics['Total Return'] = (1 + strategy_returns).prod() - 1
        metrics['Benchmark Return'] = (1 + benchmark_returns).prod() - 1
        metrics['Excess Return'] = metrics['Total Return'] - metrics['Benchmark Return']
        
        # Annualized metrics (252 trading days)
        years = len(strategy_returns) / 252
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (1/years) - 1 if years > 0 else 0
        metrics['Annualized Volatility'] = strategy_returns.std() * np.sqrt(252)
        metrics['Benchmark Volatility'] = benchmark_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility'] \
                                  if metrics['Annualized Volatility'] != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        metrics['Max Drawdown Days'] = max_duration
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0].count()
        total_trades = strategy_returns[strategy_returns != 0].count()
        metrics['Win Rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades
        metrics['Number of Trades'] = total_trades
        
        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        # Average trade
        metrics['Avg Trade'] = strategy_returns[strategy_returns != 0].mean() if total_trades > 0 else 0
        
        # Calmar Ratio
        metrics['Calmar Ratio'] = metrics['Annualized Return'] / abs(metrics['Max Drawdown']) \
                                  if metrics['Max Drawdown'] != 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['Sortino Ratio'] = metrics['Annualized Return'] / downside_deviation if downside_deviation != 0 else 0
        
        return metrics
    
    def print_backtest_summary(self, backtest, metrics):
        """Print a formatted backtest summary"""
        print("\n" + "-" * 60)
        print("BACKTEST SUMMARY")
        print("-" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Ratio' in key or 'Rate' in key:
                    print(f"{key:<25}: {value:.4f}")
                elif 'Days' in key:
                    print(f"{key:<25}: {int(value)}")
                else:
                    print(f"{key:<25}: {value:.2%}")
            else:
                print(f"{key:<25}: {value}")
    
    def plot_results(self, backtest, strategy_name, metrics):
        """Plot comprehensive backtest results"""
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f'{self.symbol} - {strategy_name} Strategy', fontsize=16, y=0.95)
        
        # Create grid for subplots
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Price and signals
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(backtest.index, backtest['price'], label='Price', color='black', alpha=0.7, linewidth=1)
        
        # Buy signals
        buy_signals = backtest[backtest['signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['price'], 
                       marker='^', color='green', s=50, label='Buy', alpha=0.7, zorder=5)
        
        # Sell signals
        sell_signals = backtest[backtest['signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['price'], 
                       marker='v', color='red', s=50, label='Sell', alpha=0.7, zorder=5)
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Chart with Trading Signals')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(backtest.index, backtest['cumulative_returns'], 
                label='Strategy', color='blue', linewidth=2)
        ax2.plot(backtest.index, backtest['buy_hold_returns'], 
                label='Buy & Hold', color='gray', linewidth=1, alpha=0.7, linestyle='--')
        
        # Fill between
        ax2.fill_between(backtest.index, 1, backtest['cumulative_returns'],
                        where=backtest['cumulative_returns'] >= 1, 
                        color='green', alpha=0.2)
        ax2.fill_between(backtest.index, 1, backtest['cumulative_returns'],
                        where=backtest['cumulative_returns'] < 1, 
                        color='red', alpha=0.2)
        
        ax2.set_ylabel('Cumulative Returns')
        ax2.set_title('Equity Curve')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = fig.add_subplot(gs[2, :])
        cumulative = backtest['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        ax3.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown, color='red', alpha=0.7, linewidth=1)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe
        ax4 = fig.add_subplot(gs[3, 0])
        rolling_sharpe = backtest['strategy_returns'].rolling(63).mean() / \
                        backtest['strategy_returns'].rolling(63).std() * np.sqrt(252)
        ax4.plot(rolling_sharpe.index, rolling_sharpe, color='purple', linewidth=1)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe')
        ax4.set_title('Rolling 3-Month Sharpe')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Monthly Returns Heatmap
        ax5 = fig.add_subplot(gs[3, 1])
        monthly_returns = backtest['strategy_returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        if len(monthly_returns) > 0:
            monthly_pivot = pd.pivot_table(
                pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values
                }),
                values='Return',
                index='Month',
                columns='Year',
                aggfunc='mean'
            )
            
            sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=ax5, cbar_kws={'label': 'Return %'})
            ax5.set_title('Monthly Returns (%)')
        
        # Plot 6: Distribution of Returns
        ax6 = fig.add_subplot(gs[3, 2])
        ax6.hist(backtest['strategy_returns'].dropna() * 100, bins=50, 
                color='blue', alpha=0.7, edgecolor='black')
        ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax6.set_xlabel('Daily Return %')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Returns')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def compare_strategies(all_results):
    """Compare performance of multiple strategies"""
    # Create comparison DataFrame
    comparison = pd.DataFrame(all_results).T
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*70)
    
    # Format the output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(comparison.round(4))
    
    # Find best strategies
    best_sharpe = comparison['Sharpe Ratio'].idxmax()
    best_return = comparison['Total Return'].idxmax()
    best_drawdown = comparison['Max Drawdown'].idxmax()  # Least negative
    
    print(f"\n{'='*70}")
    print("BEST PERFORMERS:")
    print(f"{'='*70}")
    print(f"Best Sharpe Ratio: {best_sharpe} ({comparison.loc[best_sharpe, 'Sharpe Ratio']:.4f})")
    print(f"Best Total Return: {best_return} ({comparison.loc[best_return, 'Total Return']:.2%})")
    print(f"Best Drawdown: {best_drawdown} ({comparison.loc[best_drawdown, 'Max Drawdown']:.2%})")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Strategy Comparison', fontsize=16)
    
    # Total Return comparison
    ax1 = axes[0, 0]
    returns = comparison['Total Return'] * 100
    colors = ['green' if x > 0 else 'red' for x in returns]
    returns.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Total Return Comparison')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Sharpe Ratio comparison
    ax2 = axes[0, 1]
    sharpe = comparison['Sharpe Ratio']
    colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sharpe]
    sharpe.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Max Drawdown comparison
    ax3 = axes[1, 0]
    drawdown = comparison['Max Drawdown'] * 100
    drawdown.plot(kind='bar', ax=ax3, color='red')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Maximum Drawdown Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Win Rate comparison
    ax4 = axes[1, 1]
    winrate = comparison['Win Rate'] * 100
    winrate.plot(kind='bar', ax=ax4, color='blue')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison

def run_multi_strategy_comparison(symbol='AAPL', start_date='2020-01-01', 
                                 end_date='2023-12-31'):
    """Run and compare multiple trading strategies"""
    # Initialize strategy system
    trader = AlgoTradingStrategy(symbol, start_date, end_date)
    
    try:
        # Fetch data
        data = trader.fetch_data()
        
        if data is None or len(data) < 50:
            print("Insufficient data for analysis")
            return None, None, None
        
        # Dictionary to store results
        all_results = {}
        
        # Strategy 1: Trend Following
        print("\n" + "#"*70)
        print("STRATEGY 1: Trend Following (MA Crossover)")
        print("#"*70)
        signals_trend = trader.trend_following_strategy(20, 50)
        backtest_trend, metrics_trend = trader.backtest_strategy(
            signals_trend, "Trend Following",
            stop_loss=0.05, take_profit=0.10, transaction_cost=0.001
        )
        trader.plot_results(backtest_trend, "Trend Following", metrics_trend)
        all_results['Trend Following'] = metrics_trend
        
        # Strategy 2: Mean Reversion
        print("\n" + "#"*70)
        print("STRATEGY 2: Mean Reversion (Bollinger Bands + RSI)")
        print("#"*70)
        signals_meanrev = trader.mean_reversion_strategy(20, 30, 70)
        backtest_meanrev, metrics_meanrev = trader.backtest_strategy(
            signals_meanrev, "Mean Reversion",
            stop_loss=0.03, take_profit=0.05, transaction_cost=0.001
        )
        trader.plot_results(backtest_meanrev, "Mean Reversion", metrics_meanrev)
        all_results['Mean Reversion'] = metrics_meanrev
        
        # Strategy 3: Momentum
        print("\n" + "#"*70)
        print("STRATEGY 3: Momentum Strategy")
        print("#"*70)
        signals_momentum = trader.momentum_strategy(20, 1.2)
        backtest_momentum, metrics_momentum = trader.backtest_strategy(
            signals_momentum, "Momentum",
            stop_loss=0.04, take_profit=0.08, transaction_cost=0.001
        )
        trader.plot_results(backtest_momentum, "Momentum", metrics_momentum)
        all_results['Momentum'] = metrics_momentum
        
        # Strategy 4: Machine Learning
        print("\n" + "#"*70)
        print("STRATEGY 4: Machine Learning (Random Forest)")
        print("#"*70)
        signals_ml, model, scaler, feature_importance = trader.ml_strategy()
        
        if not signals_ml.empty:
            backtest_ml, metrics_ml = trader.backtest_strategy(
                signals_ml, "Machine Learning",
                stop_loss=0.03, take_profit=0.06, transaction_cost=0.001
            )
            trader.plot_results(backtest_ml, "Machine Learning", metrics_ml)
            all_results['Machine Learning'] = metrics_ml
        
        # Compare all strategies
        comparison = compare_strategies(all_results)
        
        return all_results, trader, comparison
        
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function to run the algorithmic trading system"""
    print("="*70)
    print("ALGORITHMIC TRADING STRATEGY SIMULATOR")
    print("="*70)
    print("\nThis program will backtest multiple trading strategies and compare their performance.")
    
    # Get user input
    symbol = input("\nEnter stock symbol (default: AAPL): ").upper().strip()
    if not symbol:
        symbol = 'AAPL'
    
    start_date = input("Enter start date (default: 2020-01-01): ").strip()
    if not start_date:
        start_date = '2020-01-01'
    
    end_date = input("Enter end date (default: 2023-12-31): ").strip()
    if not end_date:
        end_date = '2023-12-31'
    
    # Validate dates
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Using defaults.")
        start_date = '2020-01-01'
        end_date = '2023-12-31'
    
    # Run comparison
    print(f"\nAnalyzing {symbol} from {start_date} to {end_date}")
    print("-" * 70)
    
    all_results, trader, comparison = run_multi_strategy_comparison(symbol, start_date, end_date)
    
    if trader and trader.data is not None:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        # Save results to CSV if available
        if comparison is not None:
            filename = f"{symbol}_strategy_comparison.csv"
            comparison.to_csv(filename)
            print(f"\nResults saved to {filename}")
    
    return all_results, trader

if __name__ == "__main__":
    # Set style for better plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
    
    # Run main
    results, trader = main()