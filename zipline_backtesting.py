"""
Zipline Backtesting Example
A comprehensive example demonstrating algorithmic trading backtesting with Zipline
"""

from zipline.utils.run_algo import run_algorithm

from zipline.api import order_target_percent, record, symbol, set_commission
from zipline.finance import commission
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# Define the trading strategy
def initialize(context):
    """
    Initialize function called once at the start of the backtest.
    Set up any state needed for the strategy.
    """
    # Set the stock we want to trade
    context.asset = symbol('AAPL')
    
    # Set commission model (e.g., $0.001 per share)
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))
    
    # Strategy parameters
    context.short_window = 20  # Short moving average window
    context.long_window = 50   # Long moving average window
    
    # Track whether we're currently invested
    context.invested = False
    
    print("Strategy initialized with moving average crossover")


def handle_data(context, data):
    """
    Called every trading day/minute. This is where trading logic goes.
    """
    # Get historical price data
    prices = data.history(
        context.asset, 
        'price', 
        context.long_window + 1, 
        '1d'
    )
    
    # Calculate moving averages
    short_mavg = prices[-context.short_window:].mean()
    long_mavg = prices[-context.long_window:].mean()
    
    # Get current price
    current_price = data.current(context.asset, 'price')
    
    # Trading logic: Moving Average Crossover Strategy
    # Buy signal: short MA crosses above long MA
    if short_mavg > long_mavg and not context.invested:
        # Buy - allocate 100% of portfolio
        order_target_percent(context.asset, 1.0)
        context.invested = True
        print(f"BUY signal at {current_price:.2f}")
        
    # Sell signal: short MA crosses below long MA
    elif short_mavg < long_mavg and context.invested:
        # Sell - liquidate position
        order_target_percent(context.asset, 0.0)
        context.invested = False
        print(f"SELL signal at {current_price:.2f}")
    
    # Record values for analysis
    record(
        price=current_price,
        short_mavg=short_mavg,
        long_mavg=long_mavg,
        invested=context.invested
    )


def analyze(context, perf):
    """
    Called at the end of the backtest to analyze results.
    """
    # Calculate performance metrics
    returns = perf['returns']
    
    print("\n=== Backtest Results ===")
    print(f"Total Return: {perf['algorithm_period_return'].iloc[-1]:.2%}")
    print(f"Sharpe Ratio: {returns.mean() / returns.std() * (252 ** 0.5):.2f}")
    print(f"Max Drawdown: {(perf['max_drawdown'] * 100).min():.2f}%")
    print(f"Total Trades: {len(perf[perf['orders'] != '[]'])}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot portfolio value
    ax1.plot(perf.index, perf['portfolio_value'])
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.grid(True)
    
    # Plot price and moving averages
    ax2.plot(perf.index, perf['price'], label='Price', alpha=0.7)
    ax2.plot(perf.index, perf['short_mavg'], label=f'{context.short_window}-day MA')
    ax2.plot(perf.index, perf['long_mavg'], label=f'{context.long_window}-day MA')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Asset Price and Moving Averages')
    ax2.legend()
    ax2.grid(True)
    
    # Plot positions
    ax3.fill_between(perf.index, 0, perf['invested'], alpha=0.3, label='Invested')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.set_title('Investment Position')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'backtest_results.png'")
    
    return perf


# Run the backtest
if __name__ == '__main__':
    # Define backtest parameters
    start_date = pd.Timestamp('2020-01-01', tz='utc')
    end_date = pd.Timestamp('2023-12-31', tz='utc')
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Run the algorithm
    results = run_algorithm(
        start=start_date,
        end=end_date,
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=100000,  # Starting capital
        data_frequency='daily',
        bundle='quantopian-quandl'  # Data bundle (needs to be ingested first)
    )
    
    # Save results to CSV
    results.to_csv('backtest_results.csv')
    print("\nResults saved to 'backtest_results.csv'")


"""
ALTERNATIVE: Mean Reversion Strategy
Uncomment below for a different strategy example
"""

# def initialize_mean_reversion(context):
#     context.asset = symbol('SPY')
#     context.lookback = 20
#     context.std_threshold = 2
#     set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))

# def handle_data_mean_reversion(context, data):
#     prices = data.history(context.asset, 'price', context.lookback, '1d')
#     mean_price = prices.mean()
#     std_price = prices.std()
#     current_price = data.current(context.asset, 'price')
#     
#     # Buy when price is 2 std below mean
#     if current_price < mean_price - context.std_threshold * std_price:
#         order_target_percent(context.asset, 1.0)
#     
#     # Sell when price is 2 std above mean
#     elif current_price > mean_price + context.std_threshold * std_price:
#         order_target_percent(context.asset, 0.0)
#     
#     record(price=current_price, mean=mean_price)