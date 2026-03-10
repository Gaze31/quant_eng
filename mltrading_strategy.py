# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')

# class MLTradingStrategy:
#     def __init__(self, lookback=20):
#         self.lookback = lookback
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.scaler = StandardScaler()
#         self.is_trained = False
        
#     def generate_sample_data(self, n_days=1000):
#         """Generate synthetic price data for demonstration"""
#         np.random.seed(42)
#         dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
#         # Generate realistic price movements
#         returns = np.random.randn(n_days) * 0.02
#         price = 100 * np.exp(np.cumsum(returns))
        
#         df = pd.DataFrame({
#             'date': dates,
#             'close': price,
#             'high': price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
#             'low': price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
#             'volume': np.random.randint(1000000, 5000000, n_days)
#         })
        
#         df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
#         return df.set_index('date')
    
#     def calculate_features(self, df):
#         """Calculate technical indicators as features"""
#         features = pd.DataFrame(index=df.index)
        
#         # Price-based features
#         features['returns'] = df['close'].pct_change()
#         features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
#         # Moving averages
#         for period in [5, 10, 20, 50]:
#             features[f'sma_{period}'] = df['close'].rolling(period).mean()
#             features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
#         # Volatility
#         features['volatility_10'] = features['returns'].rolling(10).std()
#         features['volatility_20'] = features['returns'].rolling(20).std()
        
#         # RSI (Relative Strength Index)
#         delta = df['close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / loss
#         features['rsi'] = 100 - (100 / (1 + rs))
        
#         # MACD
#         exp1 = df['close'].ewm(span=12, adjust=False).mean()
#         exp2 = df['close'].ewm(span=26, adjust=False).mean()
#         features['macd'] = exp1 - exp2
#         features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
#         # Bollinger Bands
#         features['bb_middle'] = df['close'].rolling(20).mean()
#         bb_std = df['close'].rolling(20).std()
#         features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
#         features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
#         features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
#         # Volume features
#         features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
#         # Momentum
#         features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
#         features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
#         return features
    
#     def create_labels(self, df, forward_days=5, threshold=0.02):
#         """Create target labels: 1 for buy signal, 0 for hold/sell"""
#         future_returns = df['close'].shift(-forward_days) / df['close'] - 1
#         labels = (future_returns > threshold).astype(int)
#         return labels
    
#     def prepare_data(self, df, forward_days=5, threshold=0.02):
#         """Prepare features and labels for training"""
#         features = self.calculate_features(df)
#         labels = self.create_labels(df, forward_days, threshold)
        
#         # Combine and drop NaN values
#         data = features.join(labels.rename('target'))
#         data = data.dropna()
        
#         X = data.drop('target', axis=1)
#         y = data['target']
        
#         return X, y
    
#     def train(self, df, test_size=0.2):
#         """Train the ML model"""
#         X, y = self.prepare_data(df)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, shuffle=False
#         )
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Train model
#         self.model.fit(X_train_scaled, y_train)
#         self.is_trained = True
        
#         # Evaluate
#         train_score = self.model.score(X_train_scaled, y_train)
#         test_score = self.model.score(X_test_scaled, y_test)
        
#         print(f"Training Accuracy: {train_score:.4f}")
#         print(f"Testing Accuracy: {test_score:.4f}")
        
#         # Feature importance
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\nTop 10 Most Important Features:")
#         print(feature_importance.head(10).to_string(index=False))
        
#         return train_score, test_score
    
#     def predict(self, df):
#         """Generate trading signals"""
#         if not self.is_trained:
#             raise ValueError("Model must be trained before prediction")
        
#         features = self.calculate_features(df)
#         features = features.dropna()
        
#         X_scaled = self.scaler.transform(features)
#         predictions = self.model.predict(X_scaled)
#         probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
#         signals = pd.DataFrame({
#             'signal': predictions,
#             'probability': probabilities
#         }, index=features.index)
        
#         return signals
    
#     def backtest(self, df, initial_capital=10000):
#         """Simple backtest of the strategy"""
#         signals = self.predict(df)
        
#         # Align signals with price data
#         prices = df['close'].loc[signals.index]
        
#         # Calculate returns
#         strategy_returns = []
#         positions = []
#         capital = initial_capital
#         shares = 0
        
#         for i in range(len(signals)):
#             if signals['signal'].iloc[i] == 1 and shares == 0:
#                 # Buy signal
#                 shares = capital / prices.iloc[i]
#                 capital = 0
#                 positions.append('LONG')
#             elif signals['signal'].iloc[i] == 0 and shares > 0:
#                 # Sell signal
#                 capital = shares * prices.iloc[i]
#                 shares = 0
#                 positions.append('FLAT')
#             else:
#                 positions.append('HOLD')
            
#             # Calculate portfolio value
#             portfolio_value = capital + (shares * prices.iloc[i])
#             strategy_returns.append(portfolio_value)
        
#         results = pd.DataFrame({
#             'price': prices,
#             'signal': signals['signal'],
#             'probability': signals['probability'],
#             'position': positions,
#             'portfolio_value': strategy_returns
#         }, index=signals.index)
        
#         # Performance metrics
#         total_return = (results['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
#         buy_hold_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
#         print(f"\n=== Backtest Results ===")
#         print(f"Initial Capital: ${initial_capital:,.2f}")
#         print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
#         print(f"Strategy Return: {total_return:.2f}%")
#         print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
#         print(f"Outperformance: {total_return - buy_hold_return:.2f}%")
        
#         return results

# # Example usage
# if __name__ == "__main__":
#     print("=== ML Trading Strategy Demo ===\n")
    
#     # Initialize strategy
#     strategy = MLTradingStrategy(lookback=20)
    
#     # Generate sample data
#     print("Generating sample price data...")
#     data = strategy.generate_sample_data(n_days=1000)
#     print(f"Generated {len(data)} days of data\n")
    
#     # Train the model
#     print("Training the model...")
#     strategy.train(data, test_size=0.2)
    
#     # Run backtest
#     print("\nRunning backtest...")
#     results = strategy.backtest(data, initial_capital=10000)
    
#     print("\nSample of recent signals:")
#     print(results[['price', 'signal', 'probability', 'position']].tail(10))



# import yfinance as yf

# # Download real NSE data
# stock_symbol = "RELIANCE.NS"  # .NS suffix for NSE stocks
# data = yf.download(stock_symbol, start='2020-01-01', end='2025-12-31')

# # Use this data with the strategy
# strategy = MLTradingStrategy()
# strategy.train(data)
# results = strategy.backtest(data, initial_capital=100000)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, provide helpful message if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Note: yfinance not installed. Install with: pip install yfinance")
    print("Using sample data for demonstration.\n")

class MLTradingStrategy:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.stock_symbol = None
    
    def fetch_indian_stock_data(self, symbol, start_date=None, end_date=None, period='2y'):
        """
        Fetch real Indian stock data from NSE using yfinance
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
            Function will automatically add .NS suffix for NSE
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        period : str, optional
            Period to download (e.g., '1y', '2y', '5y', 'max')
        """
        if not YFINANCE_AVAILABLE:
            print("Error: yfinance is not installed.")
            print("Please install it using: pip install yfinance")
            print("Falling back to sample data...\n")
            return self.generate_sample_data()
        
        # Add .NS suffix if not already present (for NSE stocks)
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = f"{symbol}.NS"
        
        self.stock_symbol = symbol
        
        try:
            print(f"Fetching data for {symbol} from NSE...")
            
            if start_date and end_date:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(symbol, period=period, progress=False)
            
            if data.empty:
                print(f"Warning: No data found for {symbol}")
                print("This could mean:")
                print("  1. The symbol is incorrect")
                print("  2. The stock is delisted")
                print("  3. There's a connectivity issue")
                print("\nTrying alternative symbol with .BO (BSE) suffix...")
                
                # Try BSE if NSE fails
                if symbol.endswith('.NS'):
                    alt_symbol = symbol.replace('.NS', '.BO')
                    data = yf.download(alt_symbol, period=period, progress=False)
                    if not data.empty:
                        self.stock_symbol = alt_symbol
                        print(f"Success! Found data on BSE: {alt_symbol}")
                
                if data.empty:
                    print("\nFalling back to sample data for demonstration...")
                    return self.generate_sample_data()
            
            # Standardize column names (yfinance returns multi-index for single stocks)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            print(f"Successfully fetched {len(data)} days of data")
            print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"Latest close price: ₹{data['close'].iloc[-1]:.2f}\n")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            print("Falling back to sample data...\n")
            return self.generate_sample_data()
    
    def get_stock_info(self, symbol):
        """Get detailed information about an Indian stock"""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available. Install it to get stock info.")
            return None
        
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            print(f"\n{'='*60}")
            print(f"Stock Information: {symbol}")
            print(f"{'='*60}")
            print(f"Company Name: {info.get('longName', 'N/A')}")
            print(f"Sector: {info.get('sector', 'N/A')}")
            print(f"Industry: {info.get('industry', 'N/A')}")
            print(f"Market Cap: ₹{info.get('marketCap', 0):,.0f}")
            print(f"52 Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
            print(f"52 Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")
            print(f"Average Volume: {info.get('averageVolume', 'N/A'):,}")
            print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
            print(f"{'='*60}\n")
            
            return info
            
        except Exception as e:
            print(f"Could not fetch stock info: {str(e)}\n")
            return None
        
    def generate_sample_data(self, n_days=1000):
        """Generate synthetic price data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Generate realistic price movements
        returns = np.random.randn(n_days) * 0.02
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'close': price,
            'high': price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
            'low': price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
            'volume': np.random.randint(1000000, 5000000, n_days)
        })
        
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        return df.set_index('date')
    
    def calculate_features(self, df):
        """Calculate technical indicators as features"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # Volatility
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        features['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
        features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Momentum
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return features
    
    def create_labels(self, df, forward_days=5, threshold=0.02):
        """Create target labels: 1 for buy signal, 0 for hold/sell"""
        future_returns = df['close'].shift(-forward_days) / df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        return labels
    
    def prepare_data(self, df, forward_days=5, threshold=0.02):
        """Prepare features and labels for training"""
        features = self.calculate_features(df)
        labels = self.create_labels(df, forward_days, threshold)
        
        # Combine and drop NaN values
        data = features.join(labels.rename('target'))
        data = data.dropna()
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        return X, y
    
    def train(self, df, test_size=0.2):
        """Train the ML model"""
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return train_score, test_score
    
    def predict(self, df):
        """Generate trading signals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self.calculate_features(df)
        features = features.dropna()
        
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        signals = pd.DataFrame({
            'signal': predictions,
            'probability': probabilities
        }, index=features.index)
        
        return signals
    
    def backtest(self, df, initial_capital=10000):
        """Simple backtest of the strategy"""
        signals = self.predict(df)
        
        # Align signals with price data
        prices = df['close'].loc[signals.index]
        
        # Calculate returns
        strategy_returns = []
        positions = []
        capital = initial_capital
        shares = 0
        
        for i in range(len(signals)):
            if signals['signal'].iloc[i] == 1 and shares == 0:
                # Buy signal
                shares = capital / prices.iloc[i]
                capital = 0
                positions.append('LONG')
            elif signals['signal'].iloc[i] == 0 and shares > 0:
                # Sell signal
                capital = shares * prices.iloc[i]
                shares = 0
                positions.append('FLAT')
            else:
                positions.append('HOLD')
            
            # Calculate portfolio value
            portfolio_value = capital + (shares * prices.iloc[i])
            strategy_returns.append(portfolio_value)
        
        results = pd.DataFrame({
            'price': prices,
            'signal': signals['signal'],
            'probability': signals['probability'],
            'position': positions,
            'portfolio_value': strategy_returns
        }, index=signals.index)
        
        # Performance metrics
        total_return = (results['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        buy_hold_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
        print(f"\n=== Backtest Results ===")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
        print(f"Strategy Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Outperformance: {total_return - buy_hold_return:.2f}%")
        
        return results

# Example usage with Indian Stocks
if __name__ == "__main__":
    print("=== ML Trading Strategy for Indian Stocks (NSE) ===\n")
    
    # Popular Indian stocks on NSE
    indian_stocks = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank",
        "INFY": "Infosys",
        "ICICIBANK": "ICICI Bank",
        "TATASTEEL": "Tata Steel",
        "AXISBANK": "Axis Bank",
        "BHARTIARTL": "Bharti Airtel",
        "ITC": "ITC Limited",
        "KOTAKBANK": "Kotak Mahindra Bank"
    }
    
    print("Popular Indian Stocks Available:")
    print("=" * 60)
    for i, (symbol, name) in enumerate(indian_stocks.items(), 1):
        print(f"{i:2d}. {symbol:12s} - {name}")
    print("=" * 60)
    
    # Initialize strategy
    strategy = MLTradingStrategy(lookback=20)
    
    # Select a stock (you can change this to any symbol from the list above)
    selected_stock = "RELIANCE"  # Change this to test different stocks
    
    print(f"\n📊 Analyzing: {selected_stock} ({indian_stocks[selected_stock]})\n")
    
    # Fetch real data from NSE
    if YFINANCE_AVAILABLE:
        # Get stock information
        strategy.get_stock_info(selected_stock)
        
        # Fetch historical data (last 2 years)
        data = strategy.fetch_indian_stock_data(
            symbol=selected_stock,
            period='2y'  # Options: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
        )
        
        # Alternative: Use specific date range
        # data = strategy.fetch_indian_stock_data(
        #     symbol=selected_stock,
        #     start_date='2023-01-01',
        #     end_date='2025-12-31'
        # )
    else:
        print("Using sample data for demonstration...")
        print("Install yfinance to use real NSE data: pip install yfinance\n")
        data = strategy.generate_sample_data(n_days=500)
    
    # Check if we have enough data
    if len(data) < 100:
        print("Warning: Not enough data for reliable analysis.")
        print("Fetching more data or using sample data...\n")
        data = strategy.generate_sample_data(n_days=500)
    
    # Train the model
    print("Training the ML model...")
    print("-" * 60)
    strategy.train(data, test_size=0.2)
    
    # Run backtest
    print("\nRunning backtest...")
    print("-" * 60)
    results = strategy.backtest(data, initial_capital=100000)  # ₹1 Lakh
    
    print("\n📈 Recent Trading Signals:")
    print("=" * 60)
    recent_signals = results[['price', 'signal', 'probability', 'position']].tail(10)
    for idx, row in recent_signals.iterrows():
        signal_emoji = "🟢 BUY" if row['signal'] == 1 else "🔴 SELL"
        print(f"{idx.date()} | ₹{row['price']:8.2f} | {signal_emoji} | "
              f"Confidence: {row['probability']*100:5.1f}% | {row['position']}")
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("  • Install yfinance: pip install yfinance")
    print("  • Change 'selected_stock' variable to test different stocks")
    print("  • Adjust 'period' parameter for more/less historical data")
    print("  • Use start_date/end_date for specific date ranges")
    print("=" * 60)