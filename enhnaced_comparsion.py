import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: Generate or Load Tick Data
# ============================================

def generate_sample_tick_data(days=3, trades_per_day=10000, start_price=100.0):
    """Generate sample tick data"""
    np.random.seed(42)
    
    start_date = datetime(2024, 1, 2, 9, 30, 0)
    total_ticks = days * trades_per_day
    
    # Generate timestamps
    timestamps = []
    current_time = start_date
    
    for i in range(total_ticks):
        current_time += timedelta(milliseconds=np.random.randint(50, 500))
        
        # Skip outside trading hours
        if current_time.hour >= 16 and current_time.minute > 0:
            current_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            current_time += timedelta(days=1)
        
        timestamps.append(current_time)
    
    # Generate price series
    prices = [start_price]
    returns = np.random.normal(0, 0.0002, total_ticks)
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Generate volume
    volumes = np.random.gamma(2, 1000, total_ticks).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    return df

# ============================================
# STEP 2: Create Bar Types Function
# ============================================

def create_bars(tick_data, bar_type='time', bar_size=5):
    """
    Create different types of bars from tick data
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
        # Volume bars
        volume_threshold = bar_size * 1000
        cum_volume = 0
        bar_data = []
        
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
        # Tick bars
        tick_threshold = bar_size
        bars = tick_data.groupby(np.arange(len(tick_data)) // tick_threshold).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        bars.index = tick_data.iloc[::tick_threshold].index[:len(bars)]
    
    # Add returns
    bars['returns'] = bars['close'].pct_change()
    bars.dropna(inplace=True)
    
    return bars

# ============================================
# STEP 3: Comparison Function
# ============================================

def compare_bar_statistics(bars_dict, verbose=True):
    """
    Compare statistical properties of different bar types
    """
    results = []
    
    for name, bars in bars_dict.items():
        if bars.empty or 'returns' not in bars.columns:
            print(f"Warning: No valid data for {name}")
            continue
            
        returns = bars['returns'].dropna()
        
        if len(returns) == 0:
            print(f"Warning: No returns data for {name}")
            continue
        
        try:
            # Basic statistics
            mean_return = returns.mean() * 100
            std_return = returns.std() * 100
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            # ADF test
            adf_result = adfuller(returns, autolag='AIC')
            adf_stat, adf_pvalue = adf_result[0], adf_result[1]
            
            # Autocorrelation
            autocorr_lag1 = returns.autocorr()
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe = mean_return / std_return if std_return != 0 else np.nan
            
            stats_dict = {
                'Bar Type': name,
                'N Bars': len(bars),
                'N Returns': len(returns),
                'Mean Return (%)': mean_return,
                'Std Dev (%)': std_return,
                'Sharpe Ratio': sharpe,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'JB p-value': jb_pvalue,
                'ADF p-value': adf_pvalue,
                'AutoCorr Lag1': autocorr_lag1
            }
            
            results.append(stats_dict)
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    if verbose and not comparison_df.empty:
        print("\n" + "="*100)
        print("BAR TYPE COMPARISON STATISTICS")
        print("="*100)
        
        # Display results
        display_cols = ['Bar Type', 'N Bars', 'Mean Return (%)', 'Std Dev (%)', 
                       'Sharpe Ratio', 'Skewness', 'Kurtosis', 'JB p-value', 
                       'ADF p-value', 'AutoCorr Lag1']
        
        for col in display_cols:
            if col not in comparison_df.columns:
                display_cols.remove(col)
        
        print("\n" + comparison_df[display_cols].to_string(index=False))
        
        # Add interpretation
        print("\n" + "-"*100)
        print("INTERPRETATION:")
        print("-"*100)
        
        for idx, row in comparison_df.iterrows():
            print(f"\n{row['Bar Type']}:")
            
            # Sharpe ratio
            if row['Sharpe Ratio'] > 1:
                print(f"  ✅ Excellent risk-adjusted returns (Sharpe > 1)")
            elif row['Sharpe Ratio'] > 0:
                print(f"  ⚠️  Positive but modest risk-adjusted returns")
            else:
                print(f"  ❌ Negative risk-adjusted returns")
            
            # Normality
            if row['JB p-value'] > 0.05:
                print(f"  ✅ Returns appear normal (p={row['JB p-value']:.4f})")
            else:
                print(f"  ❌ Returns non-normal (p={row['JB p-value']:.4f})")
            
            # Stationarity
            if row['ADF p-value'] < 0.05:
                print(f"  ✅ Stationary series (p={row['ADF p-value']:.4f})")
            elif row['ADF p-value'] < 0.1:
                print(f"  ⚠️  Weakly stationary (p={row['ADF p-value']:.4f})")
            else:
                print(f"  ❌ Non-stationary (p={row['ADF p-value']:.4f})")
            
            # Autocorrelation
            if abs(row['AutoCorr Lag1']) < 0.1:
                print(f"  ✅ Low autocorrelation (ρ={row['AutoCorr Lag1']:.4f})")
            elif abs(row['AutoCorr Lag1']) < 0.2:
                print(f"  ⚠️  Moderate autocorrelation (ρ={row['AutoCorr Lag1']:.4f})")
            else:
                print(f"  ❌ High autocorrelation (ρ={row['AutoCorr Lag1']:.4f})")
    
    return comparison_df

# ============================================
# STEP 4: Main Execution
# ============================================

print("="*100)
print("TICK DATA ANALYSIS - BAR TYPE COMPARISON")
print("="*100)

# Generate tick data
print("\n📊 Step 1: Generating sample tick data...")
tick_data = generate_sample_tick_data(days=3, trades_per_day=10000)
print(f"   Generated {len(tick_data):,} tick records")
print(f"   Date range: {tick_data.index.min()} to {tick_data.index.max()}")

# Create different bar types
print("\n📊 Step 2: Creating different bar types...")

# Define bar types to create
bars_dict = {}

# Time bars (1-minute and 5-minute)
print("   - Creating 1-minute time bars...")
bars_dict['Time Bars (1min)'] = create_bars(tick_data, bar_type='time', bar_size=1)

print("   - Creating 5-minute time bars...")
bars_dict['Time Bars (5min)'] = create_bars(tick_data, bar_type='time', bar_size=5)

# Volume bars
print("   - Creating volume bars (50k volume)...")
bars_dict['Volume Bars'] = create_bars(tick_data, bar_type='volume', bar_size=50)

# Tick bars
print("   - Creating tick bars (1000 ticks)...")
bars_dict['Tick Bars'] = create_bars(tick_data, bar_type='tick', bar_size=1000)

# Print summary of created bars
print("\n📊 Step 3: Bar creation summary:")
for name, bars in bars_dict.items():
    print(f"   • {name}: {len(bars)} bars, {bars['volume'].sum():,.0f} total volume")

# Run comparison
print("\n📊 Step 4: Running statistical comparison...")
print("\n" + "="*100)
print("=== BAR TYPE COMPARISON ===")
print("="*100)

comparison_df = compare_bar_statistics(bars_dict, verbose=True)

# ============================================
# STEP 5: Additional Analysis
# ============================================

if not comparison_df.empty:
    print("\n" + "="*100)
    print("ADDITIONAL INSIGHTS")
    print("="*100)
    
    # Find best performer in each category
    print("\n🏆 Best Performers:")
    
    # Best Sharpe ratio
    best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
    print(f"   • Best Sharpe Ratio: {comparison_df.loc[best_sharpe_idx, 'Bar Type']} "
          f"({comparison_df.loc[best_sharpe_idx, 'Sharpe Ratio']:.3f})")
    
    # Most normal
    best_normal_idx = comparison_df['JB p-value'].idxmax()
    print(f"   • Most Normal Returns: {comparison_df.loc[best_normal_idx, 'Bar Type']} "
          f"(p={comparison_df.loc[best_normal_idx, 'JB p-value']:.4f})")
    
    # Most stationary
    best_stat_idx = comparison_df['ADF p-value'].idxmin()
    print(f"   • Most Stationary: {comparison_df.loc[best_stat_idx, 'Bar Type']} "
          f"(p={comparison_df.loc[best_stat_idx, 'ADF p-value']:.4f})")
    
    # Lowest autocorrelation
    best_ac_idx = comparison_df['AutoCorr Lag1'].abs().idxmin()
    print(f"   • Most Independent: {comparison_df.loc[best_ac_idx, 'Bar Type']} "
          f"(ρ={comparison_df.loc[best_ac_idx, 'AutoCorr Lag1']:.4f})")
    
    # Lowest volatility
    best_vol_idx = comparison_df['Std Dev (%)'].idxmin()
    print(f"   • Lowest Volatility: {comparison_df.loc[best_vol_idx, 'Bar Type']} "
          f"({comparison_df.loc[best_vol_idx, 'Std Dev (%)']:.4f}%)")
    
    # Save results
    print("\n💾 Saving results to CSV...")
    comparison_df.to_csv('bar_comparison_results.csv', index=False)
    print("   Results saved to 'bar_comparison_results.csv'")
    
    # Summary statistics
    print("\n📊 Summary Statistics Across Bar Types:")
    print("-" * 50)
    summary = comparison_df[['Mean Return (%)', 'Std Dev (%)', 'Sharpe Ratio', 
                            'Skewness', 'Kurtosis', 'JB p-value', 'ADF p-value']].describe()
    print(summary.to_string())

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)