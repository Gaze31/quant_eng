import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import List, Dict, Tuple, Optional

class CostAdjustedReturns:
    """
    A class to calculate cost-adjusted returns considering various trading costs.
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005,   # 0.05% slippage
                 tax_rate: float = 0.0,            # No tax by default
                 min_commission: float = 0.0):     # Minimum commission per trade
        """
        Initialize cost parameters.
        
        Parameters:
        - commission_rate: Percentage commission per trade (e.g., 0.001 for 0.1%)
        - slippage_rate: Expected slippage as percentage of price
        - tax_rate: Tax rate on profits (e.g., 0.15 for 15%)
        - min_commission: Minimum commission per trade
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.tax_rate = tax_rate
        self.min_commission = min_commission
    
    def calculate_trade_cost(self, 
                            price: float, 
                            quantity: int, 
                            is_buy: bool) -> float:
        """
        Calculate the cost of a single trade.
        """
        trade_value = price * quantity
        
        # Calculate commission (percentage or minimum)
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Calculate slippage cost
        slippage_cost = trade_value * self.slippage_rate
        
        return commission + slippage_cost
    
    def calculate_cost_adjusted_return(self,
                                     purchase_price: float,
                                     current_price: float,
                                     quantity: int,
                                     include_tax: bool = True) -> Dict:
        """
        Calculate cost-adjusted return for a position.
        """
        # Initial investment (including purchase costs)
        buy_cost = self.calculate_trade_cost(purchase_price, quantity, is_buy=True)
        initial_investment = (purchase_price * quantity) + buy_cost
        
        # Current value (accounting for selling costs)
        sell_cost = self.calculate_trade_cost(current_price, quantity, is_buy=False)
        current_value = (current_price * quantity) - sell_cost
        
        # Gross and net returns
        gross_return = (current_price * quantity) - (purchase_price * quantity)
        net_return = current_value - initial_investment
        
        # Calculate taxes if applicable
        taxes = 0
        if include_tax and net_return > 0:
            taxes = net_return * self.tax_rate
            net_return -= taxes
        
        # Return calculations
        gross_return_pct = (gross_return / (purchase_price * quantity)) * 100
        net_return_pct = (net_return / initial_investment) * 100
        
        return {
            'initial_investment': initial_investment,
            'current_value': current_value,
            'gross_return': gross_return,
            'net_return': net_return,
            'gross_return_pct': gross_return_pct,
            'net_return_pct': net_return_pct,
            'total_costs': buy_cost + sell_cost + taxes,
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'taxes': taxes
        }
    
    def calculate_portfolio_returns(self,
                                  trades: pd.DataFrame,
                                  current_prices: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate cost-adjusted returns for a portfolio of trades.
        
        Parameters:
        - trades: DataFrame with columns ['symbol', 'purchase_price', 'quantity', 'purchase_date']
        - current_prices: Dictionary of current prices for each symbol
        """
        results = []
        
        for _, trade in trades.iterrows():
            symbol = trade['symbol']
            
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
            else:
                current_price = trade.get('current_price', trade['purchase_price'])
            
            result = self.calculate_cost_adjusted_return(
                trade['purchase_price'],
                current_price,
                trade['quantity']
            )
            result['symbol'] = symbol
            result['purchase_date'] = trade.get('purchase_date')
            results.append(result)
        
        return pd.DataFrame(results)


class TimeSeriesCostAdjustedReturns(CostAdjustedReturns):
    """
    Extended class for time series analysis of cost-adjusted returns.
    """
    
    def calculate_returns_series(self,
                               prices: pd.Series,
                               trades: List[Dict],
                               initial_capital: float = 10000) -> pd.DataFrame:
        """
        Calculate cost-adjusted returns over time.
        
        Parameters:
        - prices: Time series of prices (DateTime index)
        - trades: List of trade dictionaries {'date': date, 'action': 'buy'/'sell', 'quantity': int}
        - initial_capital: Starting capital
        """
        # Convert string dates to datetime if necessary
        for trade in trades:
            if isinstance(trade['date'], str):
                trade['date'] = pd.Timestamp(trade['date'])
        
        # Initialize tracking variables
        capital = initial_capital
        position = 0
        cost_basis = 0
        returns_series = []
        
        for date, price in prices.items():
            # Check for trades on this date
            day_trades = [t for t in trades if t['date'].date() == date.date()]
            
            for trade in day_trades:
                if trade['action'] == 'buy':
                    # Calculate buy cost
                    trade_cost = self.calculate_trade_cost(price, trade['quantity'], is_buy=True)
                    total_cost = (price * trade['quantity']) + trade_cost
                    
                    if capital >= total_cost:
                        capital -= total_cost
                        position += trade['quantity']
                        # Update cost basis (weighted average)
                        if position > 0:
                            cost_basis = (cost_basis * (position - trade['quantity']) + 
                                        price * trade['quantity']) / position
                
                elif trade['action'] == 'sell' and position >= trade['quantity']:
                    # Calculate sell proceeds (net of costs)
                    trade_cost = self.calculate_trade_cost(price, trade['quantity'], is_buy=False)
                    proceeds = (price * trade['quantity']) - trade_cost
                    
                    capital += proceeds
                    position -= trade['quantity']
                    
                    # Update cost basis if still have position
                    if position == 0:
                        cost_basis = 0
            
            # Calculate current portfolio value
            current_value = capital + (position * price)
            
            # Calculate total return
            total_return = ((current_value - initial_capital) / initial_capital) * 100
            
            returns_series.append({
                'date': date,
                'price': price,
                'position': position,
                'capital': capital,
                'portfolio_value': current_value,
                'cost_basis': cost_basis if position > 0 else 0,
                'total_return': total_return
            })
        
        return pd.DataFrame(returns_series).set_index('date')
    
    def calculate_performance_metrics(self, returns_df: pd.DataFrame) -> Dict:
        """
        Calculate key performance metrics from returns series.
        """
        # Calculate daily returns
        returns_df['daily_return'] = returns_df['portfolio_value'].pct_change()
        
        # Remove NaN values
        daily_returns = returns_df['daily_return'].dropna()
        
        # Calculate metrics
        total_return = (returns_df['portfolio_value'].iloc[-1] - 
                       returns_df['portfolio_value'].iloc[0]) / returns_df['portfolio_value'].iloc[0]
        
        # Annualized return (assuming 252 trading days)
        days = len(returns_df)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100 if not pd.isna(max_drawdown) else 0,
            'final_portfolio_value': returns_df['portfolio_value'].iloc[-1],
            'total_trades': len(returns_df[returns_df['position'].diff() != 0]) // 2
        }


def get_stock_prices(ticker, start_date, end_date):
    """
    Robust function to extract price data from yfinance.
    """
    # Download data with auto_adjust=False to maintain compatibility
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        # Try to get Adjusted Close for the specific ticker
        if ('Adj Close', ticker) in data.columns:
            prices = data[('Adj Close', ticker)]
            print("Using Adjusted Close price")
        elif ('Close', ticker) in data.columns:
            prices = data[('Close', ticker)]
            print("Using Close price (Adjusted Close not available)")
        else:
            # Try to find any price column
            price_cols = [col for col in data.columns if col[0] in ['Adj Close', 'Close']]
            if price_cols:
                prices = data[price_cols[0]]
                print(f"Using {price_cols[0][0]} price")
            else:
                # Use the first column as fallback
                prices = data.iloc[:, 0]
                print("Using first column as price data")
    else:
        # Simple columns case
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
            print("Using Adjusted Close price")
        elif 'Close' in data.columns:
            prices = data['Close']
            print("Using Close price")
        else:
            prices = data.iloc[:, 3] if data.shape[1] > 3 else data.iloc[:, 0]
            print("Using column index as price data")
    
    return prices


# Example 1: Single trade calculation
print("=" * 60)
print("SINGLE TRADE ANALYSIS")
print("=" * 60)

calculator = CostAdjustedReturns(
    commission_rate=0.001,  # 0.1%
    slippage_rate=0.0005,   # 0.05%
    tax_rate=0.15,          # 15% tax
    min_commission=1.0       # $1 minimum commission
)

# Calculate returns for a single trade
result = calculator.calculate_cost_adjusted_return(
    purchase_price=100.0,
    current_price=110.0,
    quantity=100,
    include_tax=True
)

print(f"Initial Investment: ${result['initial_investment']:.2f}")
print(f"Current Value: ${result['current_value']:.2f}")
print(f"Gross Return: ${result['gross_return']:.2f} ({result['gross_return_pct']:.2f}%)")
print(f"Net Return: ${result['net_return']:.2f} ({result['net_return_pct']:.2f}%)")
print(f"Total Costs: ${result['total_costs']:.2f}")
print(f"  - Buy Cost: ${result['buy_cost']:.2f}")
print(f"  - Sell Cost: ${result['sell_cost']:.2f}")
print(f"  - Taxes: ${result['taxes']:.2f}")

# Example 2: Time series analysis
print("\n" + "=" * 60)
print("TIME SERIES ANALYSIS")
print("=" * 60)

# Download price data using robust function
ticker = "AAPL"
try:
    prices = get_stock_prices(ticker, "2023-01-01", "2023-12-31")
    
    print(f"\nSuccessfully extracted prices for {ticker}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Number of trading days: {len(prices)}")
    print(f"First few prices:\n{prices.head()}")
    print(f"Last few prices:\n{prices.tail()}")
    
    # Define trades
    trades = [
        {'date': '2023-01-15', 'action': 'buy', 'quantity': 10},
        {'date': '2023-03-20', 'action': 'buy', 'quantity': 5},
        {'date': '2023-06-10', 'action': 'sell', 'quantity': 8},
    ]
    
    # Create time series calculator
    ts_calculator = TimeSeriesCostAdjustedReturns(
        commission_rate=0.001,
        slippage_rate=0.0005,
        tax_rate=0.15
    )
    
    # Calculate returns series
    returns_df = ts_calculator.calculate_returns_series(
        prices, trades, initial_capital=10000
    )
    
    # Calculate performance metrics
    metrics = ts_calculator.calculate_performance_metrics(returns_df)
    
    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25}: {value:,.2f}")
        else:
            print(f"{key:25}: {value}")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Portfolio value over time
    axes[0].plot(returns_df.index, returns_df['portfolio_value'], 'b-', linewidth=2)
    axes[0].set_title(f'{ticker} - Portfolio Value Over Time (Cost-Adjusted)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(returns_df.index, returns_df['portfolio_value'], alpha=0.1)
    
    # Add trade markers
    for trade in trades:
        trade_date = pd.Timestamp(trade['date'])
        if trade_date in returns_df.index:
            value = returns_df.loc[trade_date, 'portfolio_value']
            color = 'green' if trade['action'] == 'buy' else 'red'
            marker = '^' if trade['action'] == 'buy' else 'v'
            axes[0].scatter(trade_date, value, color=color, marker=marker, s=100, 
                          zorder=5, label=f"{trade['action'].capitalize()} {trade['quantity']} shares")
    
    # Position over time
    axes[1].plot(returns_df.index, returns_df['position'], 'g-', 
                 drawstyle='steps-post', linewidth=2)
    axes[1].set_title('Position Size Over Time', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Shares')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(returns_df.index, returns_df['position'], alpha=0.1, step='post')
    
    # Returns over time
    axes[2].plot(returns_df.index, returns_df['total_return'], 'r-', linewidth=2)
    axes[2].set_title('Total Return (%)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Return (%)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(returns_df.index, returns_df['total_return'], alpha=0.1)
    
    # Add horizontal line at 0% return
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed trade analysis
    print("\n" + "=" * 60)
    print("DETAILED TRADE ANALYSIS")
    print("=" * 60)
    
    for i, trade in enumerate(trades, 1):
        trade_date = pd.Timestamp(trade['date'])
        
        # Find nearest price if exact date not available
        if trade_date in prices.index:
            price = prices[trade_date]
            date_used = trade_date
        else:
            # Find nearest date
            nearest_idx = prices.index.get_indexer([trade_date], method='nearest')[0]
            price = prices.iloc[nearest_idx]
            date_used = prices.index[nearest_idx]
            print(f"\nNote: Trade {i} date {trade_date.date()} not in data, using {date_used.date()}")
        
        trade_cost = ts_calculator.calculate_trade_cost(price, trade['quantity'], 
                                                        is_buy=(trade['action'] == 'buy'))
        
        print(f"\nTrade {i}: {trade['action'].upper()} {trade['quantity']} shares at ${price:.2f}")
        print(f"  Trade Date: {date_used.date()}")
        print(f"  Trade Value: ${price * trade['quantity']:,.2f}")
        print(f"  Trade Cost: ${trade_cost:.2f}")
        
        if trade['action'] == 'buy':
            print(f"  Total Cost: ${(price * trade['quantity']) + trade_cost:,.2f}")
        else:
            print(f"  Net Proceeds: ${(price * trade['quantity']) - trade_cost:,.2f}")

except Exception as e:
    print(f"Error in time series analysis: {e}")
    import traceback
    traceback.print_exc()