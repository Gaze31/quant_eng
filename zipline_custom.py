import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class CustomZipline:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # symbol: shares
        self.portfolio = pd.DataFrame()
        self.trades = []
        
    def get_data(self, symbols, start_date, end_date):
        """Fetch historical data for given symbols"""
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            data[symbol] = df['Close']
        
        self.prices = pd.DataFrame(data)
        self.prices = self.prices.fillna(method='ffill')
        return self.prices
    
    def order(self, symbol, shares):
        """Place an order for a symbol"""
        if symbol not in self.prices.columns:
            raise ValueError(f"Symbol {symbol} not in price data")
        
        current_price = self.prices.loc[self.current_date, symbol]
        cost = shares * current_price
        
        if abs(cost) > self.capital and shares > 0:
            shares = int(self.capital / current_price)
            cost = shares * current_price
            print(f"Insufficient capital. Adjusting order to {shares} shares")
        
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        self.capital -= cost
        
        self.trades.append({
            'date': self.current_date,
            'symbol': symbol,
            'shares': shares,
            'price': current_price,
            'cost': cost
        })
        
        return shares
    
    def record(self, **kwargs):
        """Record custom variables"""
        for key, value in kwargs.items():
            if key not in self.portfolio.columns:
                self.portfolio[key] = np.nan
            self.portfolio.loc[self.current_date, key] = value
    
    def calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        total_value = self.capital
        for symbol, shares in self.positions.items():
            if shares != 0:
                current_price = self.prices.loc[self.current_date, symbol]
                total_value += shares * current_price
        return total_value
    
    def run(self, strategy_class, symbols, start_date, end_date):
        """Main backtesting loop"""
        # Get data
        self.prices = self.get_data(symbols, start_date, end_date)
        self.dates = self.prices.index
        
        # Initialize strategy
        strategy = strategy_class(self)
        
        # Initialize portfolio tracking
        self.portfolio['capital'] = np.nan
        self.portfolio['portfolio_value'] = np.nan
        self.portfolio['returns'] = np.nan
        
        # Run backtest
        for date in self.dates:
            self.current_date = date
            
            # Call strategy logic
            strategy.handle_data(date)
            
            # Record portfolio state
            self.portfolio.loc[date, 'capital'] = self.capital
            portfolio_value = self.calculate_portfolio_value()
            self.portfolio.loc[date, 'portfolio_value'] = portfolio_value
            
            # Calculate returns
            if date != self.dates[0]:
                prev_value = self.portfolio.loc[self.dates[self.dates.get_loc(date)-1], 'portfolio_value']
                self.portfolio.loc[date, 'returns'] = (portfolio_value - prev_value) / prev_value
        
        return self.portfolio
    
    def get_orders(self):
        """Return all trades"""
        return pd.DataFrame(self.trades)
    
    def analyze(self):
        """Basic performance analysis"""
        if self.portfolio.empty:
            print("No portfolio data to analyze")
            return
        
        total_return = (self.portfolio['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio['portfolio_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if len(self.portfolio['returns'].dropna()) > 1:
            sharpe = np.sqrt(252) * self.portfolio['returns'].mean() / self.portfolio['returns'].std()
            print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Maximum drawdown
        cumulative = (1 + self.portfolio['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        print(f"Max Drawdown: {max_drawdown:.2f}%")

# Example Strategy
class MovingAverageCrossStrategy:
    def __init__(self, zipline_instance, short_window=20, long_window=50):
        self.zipline = zipline_instance
        self.short_window = short_window
        self.long_window = long_window
        self.in_position = False
        
    def handle_data(self, date):
        """Called on each trading day"""
        # Calculate moving averages
        if len(self.zipline.prices) >= self.long_window:
            # Get data up to current date
            data = self.zipline.prices.loc[:date]
            
            if len(data) >= self.long_window:
                # Calculate MAs
                ma_short = data['AAPL'].iloc[-self.short_window:].mean()
                ma_long = data['AAPL'].iloc[-self.long_window:].mean()
                
                # Trading logic
                current_price = self.zipline.prices.loc[date, 'AAPL']
                
                if ma_short > ma_long and not self.in_position:
                    # Buy signal
                    shares = int(self.zipline.capital * 0.95 / current_price)
                    if shares > 0:
                        self.zipline.order('AAPL', shares)
                        self.in_position = True
                        self.zipline.record(signal='BUY')
                
                elif ma_short < ma_long and self.in_position:
                    # Sell signal
                    shares = self.zipline.positions.get('AAPL', 0)
                    if shares > 0:
                        self.zipline.order('AAPL', -shares)
                        self.in_position = False
                        self.zipline.record(signal='SELL')

# Usage Example
def main():
    # Create backtest instance
    bt = CustomZipline(initial_capital=100000)
    
    # Run backtest
    portfolio = bt.run(
        strategy_class=MovingAverageCrossStrategy,
        symbols=['AAPL'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Analyze results
    bt.analyze()
    
    # Get trades
    trades = bt.get_orders()
    print("\nTrades:")
    print(trades)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price and moving averages
    axes[0].plot(bt.prices.index, bt.prices['AAPL'], label='AAPL Price')
    axes[0].plot(portfolio.index, bt.prices['AAPL'].rolling(20).mean(), label='20-day MA')
    axes[0].plot(portfolio.index, bt.prices['AAPL'].rolling(50).mean(), label='50-day MA')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].set_title('Price and Moving Averages')
    
    # Portfolio value
    axes[1].plot(portfolio.index, portfolio['portfolio_value'])
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Portfolio Value Over Time')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()