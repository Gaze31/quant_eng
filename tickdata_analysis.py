import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Function to generate sample tick data
def generate_sample_tick_data(days=5, trades_per_day=5000, start_price=100.0):
    """
    Generate realistic sample tick data for demonstration
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    start_date = datetime(2024, 1, 2, 9, 30, 0)  # Market open
    total_ticks = days * trades_per_day
    
    # Create time index (trading hours only)
    timestamps = []
    current_time = start_date
    
    for i in range(total_ticks):
        # Random milliseconds between trades (50-500ms)
        current_time += timedelta(milliseconds=np.random.randint(50, 500))
        
        # Skip outside trading hours (9:30 AM - 4:00 PM)
        if current_time.hour >= 16 and current_time.minute > 0:
            current_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            current_time += timedelta(days=1)
        
        timestamps.append(current_time)
    
    # Generate price series with random walk
    prices = [start_price]
    returns = np.random.normal(0, 0.0002, total_ticks)  # Small random returns
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Generate volume (random with some clustering)
    volumes = np.random.gamma(2, 1000, total_ticks).astype(int)
    
    # Generate bid/ask
    spreads = np.random.gamma(2, 0.5, total_ticks)  # Spread in cents
    bid_prices = prices - spreads/200
    ask_prices = prices + spreads/200
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'bid': bid_prices,
        'ask': ask_prices,
        'spread': spreads/100
    })
    
    return df

# Generate sample data
print("Generating sample tick data...")
tick_data = generate_sample_tick_data(days=3, trades_per_day=10000)
print(f"Generated {len(tick_data):,} tick records")
print(tick_data.head())

# Set timestamp as index
tick_data.set_index('timestamp', inplace=True)

# Basic data inspection
print("\n=== DATA OVERVIEW ===")
print(f"Date range: {tick_data.index.min()} to {tick_data.index.max()}")
print(f"Total ticks: {len(tick_data):,}")
print(f"Columns: {tick_data.columns.tolist()}")
print(f"Memory usage: {tick_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check for missing values
print(f"\nMissing values:\n{tick_data.isnull().sum()}")

# Basic statistics
print(f"\nPrice statistics:\n{tick_data['price'].describe()}")

# Check for outliers (prices beyond 3 standard deviations)
mean_price = tick_data['price'].mean()
std_price = tick_data['price'].std()
outliers = tick_data[abs(tick_data['price'] - mean_price) > 3 * std_price]
print(f"\nPotential outliers: {len(outliers)} ticks ({len(outliers)/len(tick_data)*100:.2f}%)")

def create_bars(tick_data, bar_type='time', bar_size=1):
    """
    Create different types of bars from tick data
    
    Parameters:
    - bar_type: 'time', 'volume', 'tick', or 'dollar'
    - bar_size: size of the bar (minutes for time, shares for volume, etc.)
    """
    if bar_type == 'time':
        # Time bars
        bars = tick_data['price'].resample(f'{bar_size}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
    elif bar_type == 'volume':
        # Volume bars - cumulate until volume threshold reached
        volume_threshold = bar_size * 1000  # Convert to thousands
        cum_volume = 0
        bar_data = []
        bar_open = bar_high = bar_low = bar_close = bar_volume = None
        
        for idx, row in tick_data.iterrows():
            if cum_volume == 0:
                bar_open = row['price']
                bar_high = row['price']
                bar_low = row['price']
                bar_volume = 0
            
            bar_high = max(bar_high, row['price'])
            bar_low = min(bar_low, row['price'])
            bar_volume += row['volume']
            cum_volume += row['volume']
            
            if cum_volume >= volume_threshold:
                bar_close = row['price']
                bar_data.append({
                    'timestamp': idx,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': bar_volume
                })
                cum_volume = 0
        
        bars = pd.DataFrame(bar_data).set_index('timestamp')
    
    elif bar_type == 'tick':
        # Tick bars - fixed number of ticks
        tick_threshold = bar_size
        bars = tick_data.groupby(np.arange(len(tick_data)) // tick_threshold).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        bars.index = tick_data.iloc[::tick_threshold].index[:len(bars)]
    
    elif bar_type == 'dollar':
        # Dollar bars - fixed dollar volume
        dollar_threshold = bar_size * 1000000  # $1M bars
        cum_dollar = 0
        bar_data = []
        
        for idx, row in tick_data.iterrows():
            if cum_dollar == 0:
                bar_open = row['price']
                bar_high = row['price']
                bar_low = row['price']
                bar_volume = 0
            
            dollar_volume = row['price'] * row['volume']
            bar_high = max(bar_high, row['price'])
            bar_low = min(bar_low, row['price'])
            bar_volume += row['volume']
            cum_dollar += dollar_volume
            
            if cum_dollar >= dollar_threshold:
                bar_close = row['price']
                bar_data.append({
                    'timestamp': idx,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': bar_close,
                    'volume': bar_volume
                })
                cum_dollar = 0
        
        bars = pd.DataFrame(bar_data).set_index('timestamp')
    
    # Add additional columns
    bars['returns'] = bars['close'].pct_change()
    bars['range'] = bars['high'] - bars['low']
    bars['vwap'] = (bars['volume'] * (bars['high'] + bars['low'] + bars['close']) / 3) / bars['volume']
    
    return bars.dropna()

# Create different bar types
print("\n=== CREATING DIFFERENT BAR TYPES ===")
time_bars = create_bars(tick_data, bar_type='time', bar_size=5)  # 5-min bars
volume_bars = create_bars(tick_data, bar_type='volume', bar_size=100)  # 100k volume bars
tick_bars = create_bars(tick_data, bar_type='tick', bar_size=1000)  # 1000-tick bars
dollar_bars = create_bars(tick_data, bar_type='dollar', bar_size=1)  # $1M bars

print(f"Time bars (5-min): {len(time_bars)} bars")
print(f"Volume bars: {len(volume_bars)} bars")
print(f"Tick bars: {len(tick_bars)} bars")
print(f"Dollar bars: {len(dollar_bars)} bars")

def plot_tick_analysis(tick_data, time_bars):
    """
    Comprehensive visualization of tick data
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Tick Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Price series with volume
    ax1 = axes[0, 0]
    ax1.plot(tick_data.index[:5000], tick_data['price'][:5000], linewidth=0.5, alpha=0.7)
    ax1.set_title('Price Series (First 5000 ticks)')
    ax1.set_ylabel('Price')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 2. Bid-Ask spread distribution
    ax2 = axes[0, 1]
    ax2.hist(tick_data['spread'].dropna() * 100, bins=50, alpha=0.7, color='green')
    ax2.set_title('Bid-Ask Spread Distribution')
    ax2.set_xlabel('Spread (cents)')
    ax2.set_ylabel('Frequency')
    
    # 3. Candlestick-like chart for time bars
    ax3 = axes[1, 0]
    last_20_bars = time_bars.iloc[-20:]
    width = 0.6
    
    # Plot up and down bars separately
    up = last_20_bars[last_20_bars['close'] >= last_20_bars['open']]
    down = last_20_bars[last_20_bars['close'] < last_20_bars['open']]
    
    # Plot up bars
    ax3.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.7)
    ax3.bar(up.index, up['high'] - up['close'], width/10, bottom=up['close'], color='green', alpha=0.7)
    ax3.bar(up.index, up['low'] - up['open'], width/10, bottom=up['open'], color='green', alpha=0.7)
    
    # Plot down bars
    ax3.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red', alpha=0.7)
    ax3.bar(down.index, down['high'] - down['open'], width/10, bottom=down['open'], color='red', alpha=0.7)
    ax3.bar(down.index, down['low'] - down['close'], width/10, bottom=down['close'], color='red', alpha=0.7)
    
    ax3.set_title('Last 20 Time Bars (Candlestick Pattern)')
    ax3.set_ylabel('Price')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 4. Volume profile
    ax4 = axes[1, 1]
    price_bins = pd.cut(tick_data['price'], bins=30)
    volume_by_price = tick_data.groupby(price_bins)['volume'].sum()
    ax4.barh(range(len(volume_by_price)), volume_by_price.values, height=0.8)
    ax4.set_title('Volume Profile')
    ax4.set_xlabel('Volume')
    ax4.set_ylabel('Price Range')
    
    # 5. Returns distribution
    ax5 = axes[2, 0]
    returns = tick_data['price'].pct_change().dropna() * 100  # in basis points
    ax5.hist(returns, bins=100, alpha=0.7, density=True)
    ax5.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}bps')
    ax5.axvline(returns.median(), color='green', linestyle='--', label=f'Median: {returns.median():.2f}bps')
    ax5.set_title('Tick Returns Distribution (bps)')
    ax5.set_xlabel('Return (basis points)')
    ax5.set_ylabel('Density')
    ax5.legend()
    
    # 6. Intraday pattern
    ax6 = axes[2, 1]
    tick_data['hour'] = tick_data.index.hour
    tick_data['minute'] = tick_data.index.minute
    tick_data['time_of_day'] = tick_data['hour'] + tick_data['minute']/60
    
    volume_pattern = tick_data.groupby('time_of_day')['volume'].mean()
    price_pattern = tick_data.groupby('time_of_day')['price'].mean()
    
    ax6_twin = ax6.twinx()
    line1 = ax6.plot(volume_pattern.index, volume_pattern.values, color='blue', label='Volume', alpha=0.7)
    line2 = ax6_twin.plot(price_pattern.index, price_pattern.values, color='red', label='Price', alpha=0.7)
    ax6.set_xlabel('Time of Day')
    ax6.set_ylabel('Average Volume', color='blue')
    ax6_twin.set_ylabel('Average Price', color='red')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    ax6.set_title('Intraday Pattern')
    
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_tick_analysis(tick_data, time_bars)



def calculate_microstructure_metrics(tick_data, time_bars):
    """
    Calculate advanced market microstructure metrics
    """
    metrics = {}
    
    # 1. Realized volatility (5-minute)
    returns_5min = time_bars['returns'].dropna()
    metrics['Realized Vol (5min)'] = returns_5min.std() * np.sqrt(252 * 78) * 100  # Annualized
    
    # 2. Bid-ask spread statistics
    metrics['Avg Spread (cents)'] = tick_data['spread'].mean() * 100
    metrics['Median Spread (cents)'] = tick_data['spread'].median() * 100
    metrics['Spread Std'] = tick_data['spread'].std() * 100
    
    # 3. Trading intensity
    tick_data['time_diff'] = tick_data.index.to_series().diff().dt.total_seconds()
    metrics['Avg Time Between Ticks (s)'] = tick_data['time_diff'].mean()
    metrics['Median Time Between Ticks (s)'] = tick_data['time_diff'].median()
    
    # 4. Volume-weighted average price (VWAP)
    metrics['Daily VWAP'] = (tick_data['price'] * tick_data['volume']).sum() / tick_data['volume'].sum()
    
    # 5. Price impact (simplified)
    tick_data['price_change'] = tick_data['price'].diff()
    tick_data['signed_volume'] = tick_data['volume'] * np.sign(tick_data['price_change'])
    
    # Price impact coefficient
    model_data = tick_data[abs(tick_data['price_change']) > 0].dropna()
    if len(model_data) > 0:
        X = model_data['signed_volume'].values.reshape(-1, 1)
        y = abs(model_data['price_change']).values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        metrics['Price Impact Coefficient'] = model.coef_[0]
    
    return metrics

# Calculate metrics
print("\n=== MARKET MICROSTRUCTURE METRICS ===")
micro_metrics = calculate_microstructure_metrics(tick_data, time_bars)
for key, value in micro_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")
def analyze_order_flow(tick_data, window=50):
    """
    Analyze order flow and calculate flow imbalance
    """
    # Determine trade direction (simplified Lee-Ready algorithm)
    tick_data['mid_price'] = (tick_data['bid'] + tick_data['ask']) / 2
    
    # Classify trades as buy or sell
    conditions = [
        tick_data['price'] > tick_data['mid_price'],  # Buy
        tick_data['price'] < tick_data['mid_price']   # Sell
    ]
    choices = [1, -1]
    tick_data['trade_direction'] = np.select(conditions, choices, default=0)
    
    # Calculate order flow
    tick_data['signed_volume'] = tick_data['volume'] * tick_data['trade_direction']
    
    # Calculate flow imbalance
    tick_data['flow_imbalance'] = tick_data['signed_volume'].rolling(window=window).sum()
    tick_data['flow_imbalance_norm'] = tick_data['flow_imbalance'] / tick_data['volume'].rolling(window=window).sum()
    
    # Calculate cumulative order flow
    tick_data['cumulative_flow'] = tick_data['signed_volume'].cumsum()
    
    # Plot order flow analysis
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Trade direction
    ax1 = axes[0]
    direction_counts = tick_data['trade_direction'].value_counts()
    ax1.bar(['Sell', 'Neutral', 'Buy'], 
            [direction_counts.get(-1, 0), direction_counts.get(0, 0), direction_counts.get(1, 0)])
    ax1.set_title('Trade Direction Distribution')
    ax1.set_ylabel('Number of Trades')
    
    # 2. Flow imbalance over time
    ax2 = axes[1]
    ax2.plot(tick_data.index[-1000:], tick_data['flow_imbalance_norm'][-1000:], linewidth=0.8)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Normalized Order Flow Imbalance (Last 1000 ticks)')
    ax2.set_ylabel('Flow Imbalance')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 3. Cumulative flow vs price
    ax3 = axes[2]
    # Normalize both series for comparison
    price_norm = (tick_data['price'] - tick_data['price'].min()) / (tick_data['price'].max() - tick_data['price'].min())
    flow_norm = (tick_data['cumulative_flow'] - tick_data['cumulative_flow'].min()) / (tick_data['cumulative_flow'].max() - tick_data['cumulative_flow'].min())
    
    ax3.plot(tick_data.index[-2000:], price_norm[-2000:], label='Price (normalized)', linewidth=1)
    ax3.plot(tick_data.index[-2000:], flow_norm[-2000:], label='Cumulative Flow (normalized)', linewidth=1, alpha=0.7)
    ax3.set_title('Price vs Cumulative Order Flow')
    ax3.set_ylabel('Normalized Value')
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    plt.show()
    
    return tick_data

# Run order flow analysis
tick_data = analyze_order_flow(tick_data)

class RealTimeTickProcessor:
    """
    Simulate real-time tick data processing
    """
    def __init__(self, tick_data, update_interval=100):
        self.tick_data = tick_data
        self.update_interval = update_interval
        self.current_position = 0
        self.processed_data = []
        self.metrics_history = []
        
    def process_next_batch(self):
        """Process next batch of ticks"""
        start_idx = self.current_position
        end_idx = min(self.current_position + self.update_interval, len(self.tick_data))
        
        batch = self.tick_data.iloc[start_idx:end_idx]
        self.current_position = end_idx
        
        # Calculate real-time metrics
        metrics = {
            'timestamp': batch.index[-1],
            'last_price': batch['price'].iloc[-1],
            'vwap': (batch['price'] * batch['volume']).sum() / batch['volume'].sum() if batch['volume'].sum() > 0 else 0,
            'volume_sum': batch['volume'].sum(),
            'price_range': batch['price'].max() - batch['price'].min(),
            'avg_spread': batch['spread'].mean() * 100,  # in cents
            'trades_processed': len(batch),
            'buy_ratio': (batch['trade_direction'] == 1).sum() / len(batch) if len(batch) > 0 else 0
        }
        
        self.metrics_history.append(metrics)
        self.processed_data.append(batch)
        
        return metrics
    
    def run_simulation(self):
        """Run complete simulation"""
        print("Starting real-time tick processing simulation...")
        
        while self.current_position < len(self.tick_data):
            metrics = self.process_next_batch()
            
            if len(self.metrics_history) % 10 == 0:  # Print every 10 updates
                print(f"\n--- Batch {len(self.metrics_history)} ---")
                print(f"Time: {metrics['timestamp']}")
                print(f"Last Price: {metrics['last_price']:.4f}")
                print(f"VWAP: {metrics['vwap']:.4f}")
                print(f"Volume: {metrics['volume_sum']:,.0f}")
                print(f"Buy Ratio: {metrics['buy_ratio']:.2%}")
        
        return pd.DataFrame(self.metrics_history)

# Run real-time simulation
print("\n=== REAL-TIME TICK PROCESSING SIMULATION ===")
processor = RealTimeTickProcessor(tick_data, update_interval=250)
metrics_df = processor.run_simulation()

# Plot simulation results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Price evolution
axes[0, 0].plot(metrics_df['timestamp'], metrics_df['last_price'], linewidth=1)
axes[0, 0].set_title('Real-time Price Updates')
axes[0, 0].set_ylabel('Price')
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# 2. VWAP vs Last Price
axes[0, 1].plot(metrics_df['timestamp'], metrics_df['last_price'] - metrics_df['vwap'], linewidth=1)
axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Price - VWAP Deviation')
axes[0, 1].set_ylabel('Deviation')
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# 3. Buy ratio over time
axes[1, 0].plot(metrics_df['timestamp'], metrics_df['buy_ratio'], linewidth=1)
axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Buy Ratio Over Time')
axes[1, 0].set_ylabel('Buy Ratio')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# 4. Average spread
axes[1, 1].plot(metrics_df['timestamp'], metrics_df['avg_spread'], linewidth=1, color='green')
axes[1, 1].set_title('Average Bid-Ask Spread')
axes[1, 1].set_ylabel('Spread (cents)')
axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.show()

def complete_tick_analysis_pipeline(tick_data):
    """
    Run complete tick analysis pipeline
    """
    print("=" * 60)
    print("COMPLETE TICK DATA ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Data overview
    print("\n[Step 1] Data Overview")
    print(f"Total ticks: {len(tick_data):,}")
    print(f"Date range: {tick_data.index.min()} to {tick_data.index.max()}")
    
    # Step 2: Create bars
    print("\n[Step 2] Creating different bar types...")
    time_bars = create_bars(tick_data, 'time', 5)
    volume_bars = create_bars(tick_data, 'volume', 100)
    print(f"Created {len(time_bars)} time bars and {len(volume_bars)} volume bars")
    
    # Step 3: Visualize
    print("\n[Step 3] Generating visualizations...")
    plot_tick_analysis(tick_data, time_bars)
    
    # Step 4: Statistical analysis
    print("\n[Step 4] Running statistical analysis...")
    bars_dict = {'Time Bars': time_bars, 'Volume Bars': volume_bars}
  
    print("\nBar Type Comparison:")
   
    
    # Step 5: Microstructure analysis
    print("\n[Step 5] Calculating microstructure metrics...")
    micro_metrics = calculate_microstructure_metrics(tick_data, time_bars)
    for key, value in list(micro_metrics.items())[:5]:  # Show first 5
        print(f"{key}: {value:.4f}")
    
    # Step 6: Order flow analysis
    print("\n[Step 6] Analyzing order flow...")
    tick_data = analyze_order_flow(tick_data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return {
        'tick_data': tick_data,
        'time_bars': time_bars,
        'volume_bars': volume_bars,
        'micro_metrics': micro_metrics,
       
    }

# Run the complete pipeline
results = complete_tick_analysis_pipeline(tick_data)