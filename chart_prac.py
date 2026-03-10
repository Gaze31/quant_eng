"""
Gold Price Analysis and Prediction Script (macOS Compatible)
This version saves files to the current directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

def generate_sample_gold_data(days=365):
    """
    Generate sample gold price data for demonstration
    In production, replace this with real data from yfinance or another API
    """
    np.random.seed(42)
    
    # Create dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    actual_days = len(dates)
    
    # Generate realistic gold price data
    # Starting around $2000/oz with some trend and volatility
    base_price = 2000
    trend = np.linspace(0, 150, actual_days)  # Upward trend
    seasonal = 30 * np.sin(np.linspace(0, 4*np.pi, actual_days))  # Seasonal variation
    noise = np.random.normal(0, 15, actual_days)  # Random volatility
    
    prices = base_price + trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 5, actual_days),
        'High': prices + abs(np.random.normal(10, 5, actual_days)),
        'Low': prices - abs(np.random.normal(10, 5, actual_days)),
        'Close': prices,
        'Volume': np.random.randint(100000, 500000, actual_days)
    }, index=dates)
    
    return df

def fetch_real_gold_data():
    """
    Function to fetch real gold data using yfinance
    Uncomment and use this when you have yfinance installed
    
    Usage:
        import yfinance as yf
        gold = yf.Ticker("GC=F")  # Gold Futures
        df = gold.history(period="1y")
        return df
    """
    pass

def get_current_price(df):
    """Get the most recent gold price"""
    current_price = df['Close'].iloc[-1]
    current_date = df.index[-1]
    
    return current_price, current_date

def calculate_statistics(df):
    """Calculate key statistics"""
    stats = {
        'Current Price': df['Close'].iloc[-1],
        'Average Price (Period)': df['Close'].mean(),
        'Highest Price': df['Close'].max(),
        'Lowest Price': df['Close'].min(),
        'Volatility (Std Dev)': df['Close'].std(),
        '30-Day Average': df['Close'].iloc[-30:].mean(),
        '90-Day Average': df['Close'].iloc[-90:].mean(),
        'Price Change (%)': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100),
        '30-Day Change (%)': ((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100)
    }
    
    return stats

def simple_moving_average_prediction(df, window=30, forecast_days=30):
    """
    Predict using Simple Moving Average with trend analysis
    """
    # Calculate moving average
    df['SMA'] = df['Close'].rolling(window=window).mean()
    
    # Calculate trend from recent period
    recent_prices = df['Close'].iloc[-window:].values
    days_array = np.arange(len(recent_prices))
    
    # Linear regression on recent trend
    coefficients = np.polyfit(days_array, recent_prices, 1)
    trend_slope = coefficients[0]
    
    # Generate predictions
    last_price = df['Close'].iloc[-1]
    predictions = []
    
    for i in range(1, forecast_days + 1):
        pred_price = last_price + (trend_slope * i)
        predictions.append(pred_price)
    
    return predictions, trend_slope

def linear_regression_prediction(df, forecast_days=30, training_days=180):
    """
    Predict using Linear Regression on recent data
    """
    # Use last N days for training
    df_recent = df.iloc[-training_days:].copy()
    df_recent['Days'] = range(len(df_recent))
    
    X = df_recent['Days'].values.reshape(-1, 1)
    y = df_recent['Close'].values
    
    # Calculate linear regression manually
    X_mean = X.mean()
    y_mean = y.mean()
    
    numerator = ((X.flatten() - X_mean) * (y - y_mean)).sum()
    denominator = ((X.flatten() - X_mean) ** 2).sum()
    
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    
    # Make predictions
    last_day = len(df_recent)
    future_days = np.array(range(last_day, last_day + forecast_days))
    predictions = slope * future_days + intercept
    
    # Calculate accuracy metrics
    historical_pred = slope * X.flatten() + intercept
    mae = np.mean(np.abs(y - historical_pred))
    rmse = np.sqrt(np.mean((y - historical_pred) ** 2))
    r_squared = 1 - (np.sum((y - historical_pred) ** 2) / np.sum((y - y_mean) ** 2))
    
    return predictions, mae, rmse, r_squared, slope

def exponential_smoothing_prediction(df, alpha=0.3, forecast_days=30):
    """
    Predict using Exponential Smoothing (Holt's method)
    """
    prices = df['Close'].values
    
    # Initialize
    level = prices[0]
    trend = (prices[1] - prices[0])
    
    smoothed_values = [level]
    trend_values = [trend]
    
    # Calculate exponential smoothing with trend
    beta = 0.3  # Trend smoothing parameter
    
    for i in range(1, len(prices)):
        prev_level = level
        prev_trend = trend
        
        level = alpha * prices[i] + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        
        smoothed_values.append(level)
        trend_values.append(trend)
    
    # Forecast
    predictions = []
    current_level = smoothed_values[-1]
    current_trend = trend_values[-1]
    
    for i in range(1, forecast_days + 1):
        forecast = current_level + (i * current_trend)
        predictions.append(forecast)
    
    return predictions, current_trend

def calculate_confidence_intervals(predictions, historical_volatility, forecast_days):
    """
    Calculate confidence intervals for predictions
    """
    # Uncertainty increases with forecast horizon
    uncertainties = []
    
    for i in range(forecast_days):
        # Uncertainty grows with square root of time
        daily_uncertainty = historical_volatility * np.sqrt(i + 1)
        uncertainties.append(daily_uncertainty)
    
    upper_bound = predictions + (1.96 * np.array(uncertainties))  # 95% CI
    lower_bound = predictions - (1.96 * np.array(uncertainties))
    
    return upper_bound, lower_bound

def plot_predictions(df, lr_pred, sma_pred, es_pred, forecast_days=30, output_dir='.'):
    """
    Plot historical data and predictions with confidence intervals
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full historical and predictions
    ax1.plot(df.index, df['Close'], label='Historical Price', linewidth=2, color='#2E86AB', alpha=0.8)
    
    # Create future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    # Plot predictions
    ax1.plot(future_dates, lr_pred, label='Linear Regression', linestyle='--', linewidth=2, color='#A23B72', marker='o', markersize=3)
    ax1.plot(future_dates, sma_pred, label='Moving Average', linestyle='--', linewidth=2, color='#F18F01', marker='s', markersize=3)
    ax1.plot(future_dates, es_pred, label='Exponential Smoothing', linestyle='--', linewidth=2, color='#C73E1D', marker='^', markersize=3)
    
    # Add confidence interval for average prediction
    avg_pred = (np.array(lr_pred) + np.array(sma_pred) + np.array(es_pred)) / 3
    volatility = df['Close'].std()
    upper_ci, lower_ci = calculate_confidence_intervals(avg_pred, volatility, forecast_days)
    
    ax1.fill_between(future_dates, lower_ci, upper_ci, alpha=0.2, color='gray', label='95% Confidence Interval')
    
    ax1.set_title('Gold Price Analysis and Predictions', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (USD per Troy Ounce)', fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Recent 60 days + predictions (zoomed in)
    recent_df = df.iloc[-60:]
    ax2.plot(recent_df.index, recent_df['Close'], label='Recent Historical', linewidth=2.5, color='#2E86AB')
    ax2.plot(future_dates, lr_pred, label='Linear Regression', linestyle='--', linewidth=2, color='#A23B72', marker='o', markersize=4)
    ax2.plot(future_dates, sma_pred, label='Moving Average', linestyle='--', linewidth=2, color='#F18F01', marker='s', markersize=4)
    ax2.plot(future_dates, es_pred, label='Exponential Smoothing', linestyle='--', linewidth=2, color='#C73E1D', marker='^', markersize=4)
    
    ax2.fill_between(future_dates, lower_ci, upper_ci, alpha=0.2, color='gray')
    
    ax2.set_title('Recent Price Action and Predictions (Last 60 Days)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price (USD)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save to current directory or specified output directory
    output_path = os.path.join(output_dir, 'gold_price_prediction.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved as '{output_path}'")
    
    return fig

def export_predictions_to_csv(future_dates, lr_pred, sma_pred, es_pred, current_price, output_dir='.'):
    """
    Export predictions to CSV file
    """
    avg_pred = (np.array(lr_pred) + np.array(sma_pred) + np.array(es_pred)) / 3
    
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Linear_Regression': lr_pred,
        'Moving_Average': sma_pred,
        'Exponential_Smoothing': es_pred,
        'Average_Prediction': avg_pred,
        'Change_from_Current_$': avg_pred - current_price,
        'Change_from_Current_%': ((avg_pred - current_price) / current_price) * 100
    })
    
    # Save to current directory or specified output directory
    output_path = os.path.join(output_dir, 'gold_price_predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Predictions exported to '{output_path}'")
    
    return predictions_df

def main():
    """
    Main function to run the analysis
    """
    # Determine output directory (current directory by default)
    output_dir = os.getcwd()
    
    print("=" * 70)
    print(" " * 15 + "GOLD PRICE ANALYSIS AND PREDICTION")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    
    # Fetch data (using sample data for this demo)
    print("\n📊 Fetching gold price data...")
    df = generate_sample_gold_data(days=365)
    print(f"✓ Loaded {len(df)} days of historical data")
    
    # NOTE: To use real data, uncomment these lines and install yfinance:
    # import yfinance as yf
    # gold = yf.Ticker("GC=F")
    # df = gold.history(period="1y")
    
    # Current price
    current_price, current_date = get_current_price(df)
    print(f"\n💰 Current Gold Price: ${current_price:.2f}")
    print(f"   As of: {current_date.strftime('%Y-%m-%d')}")
    
    # Statistics
    print("\n" + "=" * 70)
    print(" " * 20 + "PRICE STATISTICS (Last Year)")
    print("=" * 70)
    stats = calculate_statistics(df)
    for key, value in stats.items():
        if '%' in key:
            print(f"  {key:.<50} {value:>8.2f}%")
        else:
            print(f"  {key:.<50} ${value:>8.2f}")
    
    # Predictions
    forecast_days = 30
    print(f"\n" + "=" * 70)
    print(f" " * 18 + f"PREDICTIONS FOR NEXT {forecast_days} DAYS")
    print("=" * 70)
    
    # Linear Regression
    lr_predictions, mae, rmse, r_squared, lr_slope = linear_regression_prediction(df, forecast_days)
    print(f"\n📈 Linear Regression Model:")
    print(f"  Predicted price in {forecast_days} days: ${lr_predictions[-1]:.2f}")
    print(f"  Daily trend: ${lr_slope:.2f}/day")
    print(f"  Model R²: {r_squared:.4f}")
    print(f"  Historical MAE: ${mae:.2f}")
    print(f"  Historical RMSE: ${rmse:.2f}")
    
    # Moving Average
    sma_predictions, sma_slope = simple_moving_average_prediction(df, forecast_days=forecast_days)
    print(f"\n📊 Moving Average Model:")
    print(f"  Predicted price in {forecast_days} days: ${sma_predictions[-1]:.2f}")
    print(f"  Daily trend: ${sma_slope:.2f}/day")
    
    # Exponential Smoothing
    es_predictions, es_trend = exponential_smoothing_prediction(df, forecast_days=forecast_days)
    print(f"\n📉 Exponential Smoothing Model:")
    print(f"  Predicted price in {forecast_days} days: ${es_predictions[-1]:.2f}")
    print(f"  Daily trend: ${es_trend:.2f}/day")
    
    # Average prediction
    avg_prediction = (lr_predictions[-1] + sma_predictions[-1] + es_predictions[-1]) / 3
    change_amount = avg_prediction - current_price
    change_percent = (change_amount / current_price) * 100
    
    print(f"\n" + "=" * 70)
    print(f"🎯 CONSENSUS FORECAST (Average of All Models):")
    print(f"  Predicted price in {forecast_days} days: ${avg_prediction:.2f}")
    print(f"  Expected change from current: ${change_amount:+.2f} ({change_percent:+.2f}%)")
    
    if change_percent > 0:
        print(f"  Trend: ⬆ BULLISH")
    elif change_percent < 0:
        print(f"  Trend: ⬇ BEARISH")
    else:
        print(f"  Trend: ➡ NEUTRAL")
    
    # Plot
    print("\n" + "=" * 70)
    print("📊 Generating visualizations...")
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
    plot_predictions(df, lr_predictions, sma_predictions, es_predictions, forecast_days, output_dir)
    
    # Export predictions
    print("\n📁 Exporting predictions...")
    predictions_df = export_predictions_to_csv(future_dates, lr_predictions, sma_predictions, 
                                               es_predictions, current_price, output_dir)
    
    # Summary table
    print("\n" + "=" * 70)
    print("📅 PREDICTION SUMMARY (First 7 Days):")
    print("=" * 70)
    print(predictions_df.head(7).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 70)
    print("""
  • This analysis is for EDUCATIONAL PURPOSES ONLY
  • Predictions are based on historical data and statistical models
  • Past performance does NOT guarantee future results
  • Actual prices may vary significantly due to:
    - Economic events and policy changes
    - Geopolitical developments
    - Market sentiment and speculation
    - Currency fluctuations
    - Supply and demand dynamics
  • DO NOT use this as financial advice
  • Always consult a licensed financial advisor before making investment decisions
  • The creator assumes NO responsibility for financial losses
    """)
    print("=" * 70)
    
    print("\n✅ Analysis complete!")
    print("\nGenerated files in current directory:")
    print("  • gold_price_prediction.png - Visualization chart")
    print("  • gold_price_predictions.csv - Detailed predictions data")

if __name__ == "__main__":
    main()