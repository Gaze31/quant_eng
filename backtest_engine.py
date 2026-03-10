import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import uuid
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    quantity: int
    order_type: str
    side: str
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
@dataclass
class Trade:
    """Represents an executed trade"""
    symbol: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


class Portfolio:
    """Manages positions, cash, and portfolio value"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
    def update_position(self, symbol: str, quantity: int, price: float, timestamp: datetime):
        """Update position after a trade"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        position = self.positions[symbol]
        
        if quantity > 0:  # Buy
            total_cost = quantity * price
            if total_cost > self.cash:
                print(f"Insufficient cash: {self.cash} < {total_cost}")
                return False
            
            new_quantity = position.quantity + quantity
            if position.quantity != 0:
                position.avg_price = (position.avg_price * position.quantity + price * quantity) / new_quantity
            else:
                position.avg_price = price
            self.cash -= total_cost
            position.quantity = new_quantity
            
        else:  # Sell (quantity negative)
            abs_quantity = abs(quantity)
            if abs_quantity > position.quantity:
                print(f"Insufficient position: {position.quantity} < {abs_quantity}")
                return False
            
            # Calculate realized P&L
            realized_pnl = (price - position.avg_price) * abs_quantity
            position.realized_pnl += realized_pnl
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                quantity=abs_quantity,
                entry_price=position.avg_price,
                exit_price=price,
                entry_time=timestamp,
                exit_time=timestamp,
                pnl=realized_pnl,
                pnl_pct=(price - position.avg_price) / position.avg_price * 100
            )
            self.trades.append(trade)
            
            self.cash += abs_quantity * price
            position.quantity -= abs_quantity
            
        if position.quantity == 0:
            del self.positions[symbol]
            
        return True
            
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """Update unrealized P&L for all positions"""
        total_unrealized = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.unrealized_pnl = (current_prices[symbol] - position.avg_price) * position.quantity
                total_unrealized += position.unrealized_pnl
        return total_unrealized
    
    def total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity"""
        return self.cash + sum(
            position.quantity * current_prices.get(position.symbol, 0)
            for position in self.positions.values()
        )


class DataHandler:
    """Handles market data for backtesting"""
    
    def __init__(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        if isinstance(data, pd.DataFrame):
            self.data = {'default': data}
            self.symbols = ['default']
        else:
            self.data = data
            self.symbols = list(data.keys())
            
        self.current_idx = 0
        self.dates = self._get_common_dates()
        
    def _get_common_dates(self) -> List[datetime]:
        dates = None
        for symbol in self.symbols:
            if dates is None:
                dates = set(self.data[symbol].index)
            else:
                dates = dates.intersection(set(self.data[symbol].index))
        return sorted(list(dates))
    
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        if self.current_idx < n:
            return pd.DataFrame()
        
        start_idx = max(0, self.current_idx - n)
        end_idx = self.current_idx
        dates = self.dates[start_idx:end_idx]
        
        return self.data[symbol].loc[dates]
    
    def get_current_price(self, symbol: str) -> float:
        current_date = self.dates[self.current_idx]
        return self.data[symbol].loc[current_date, 'close']
    
    def next(self) -> bool:
        self.current_idx += 1
        return self.current_idx < len(self.dates)
    
    def get_current_date(self) -> datetime:
        if self.current_idx < len(self.dates):
            return self.dates[self.current_idx]
        return None


class Strategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.portfolio: Optional[Portfolio] = None
        self.data_handler: Optional[DataHandler] = None
        
    def set_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio
        
    def set_data_handler(self, data_handler: DataHandler):
        self.data_handler = data_handler
        
    def on_start(self):
        pass
        
    def on_data(self, timestamp: datetime):
        raise NotImplementedError
        
    def on_end(self):
        pass
        
    def buy(self, symbol: str, quantity: int):
        if self.portfolio and self.data_handler:
            current_price = self.data_handler.get_current_price(symbol)
            return self.portfolio.update_position(symbol, quantity, current_price, self.data_handler.get_current_date())
            
    def sell(self, symbol: str, quantity: int):
        if self.portfolio and self.data_handler:
            current_price = self.data_handler.get_current_price(symbol)
            return self.portfolio.update_position(symbol, -quantity, current_price, self.data_handler.get_current_date())


class BacktestEngine:
    """Main backtest engine"""
    
    def __init__(self, 
                 strategy: Strategy,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        
        self.strategy = strategy
        self.data_handler = DataHandler(data)
        self.portfolio = Portfolio(initial_capital)
        self.commission = commission
        self.slippage = slippage
        
        self.strategy.set_portfolio(self.portfolio)
        self.strategy.set_data_handler(self.data_handler)
        
        self.metrics = {}
        
    def run(self) -> Dict:
        print("Starting backtest...")
        self.strategy.on_start()
        
        bar_count = 0
        while self.data_handler.next():
            current_date = self.data_handler.get_current_date()
            bar_count += 1
            
            if bar_count % 100 == 0:
                print(f"Processing bar {bar_count}...")
            
            current_prices = {
                symbol: self.data_handler.get_current_price(symbol)
                for symbol in self.data_handler.symbols
            }
            
            self.strategy.on_data(current_date)
            
            total_equity = self.portfolio.total_equity(current_prices)
            self.portfolio.equity_curve.append(total_equity)
            self.portfolio.timestamps.append(current_date)
            
        self.strategy.on_end()
        
        print(f"Backtest complete. Processed {bar_count} bars.")
        self._calculate_metrics()
        
        return self.metrics
    
    def _calculate_metrics(self):
        equity_curve = np.array(self.portfolio.equity_curve)
        
        if len(equity_curve) < 2:
            self.metrics = {
                'initial_capital': self.portfolio.initial_capital,
                'final_equity': self.portfolio.initial_capital,
                'total_return': 0,
                'total_trades': len(self.portfolio.trades)
            }
            return
        
        self.metrics['initial_capital'] = self.portfolio.initial_capital
        self.metrics['final_equity'] = equity_curve[-1]
        self.metrics['total_return'] = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(returns) > 0:
            self.metrics['annualized_return'] = (1 + self.metrics['total_return']/100) ** (252 / len(returns)) - 1
            self.metrics['annualized_return'] *= 100
            self.metrics['volatility'] = np.std(returns) * np.sqrt(252) * 100
            self.metrics['sharpe_ratio'] = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            self.metrics['max_drawdown_pct'] = self._calculate_max_drawdown_pct(equity_curve) * 100
            
        trades = self.portfolio.trades
        self.metrics['total_trades'] = len(trades)
        
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            self.metrics['winning_trades'] = len(winning_trades)
            self.metrics['losing_trades'] = len(trades) - len(winning_trades)
            self.metrics['win_rate'] = len(winning_trades) / len(trades) * 100
            self.metrics['avg_win'] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            self.metrics['avg_loss'] = np.mean([t.pnl for t in trades if t.pnl < 0]) if len([t for t in trades if t.pnl < 0]) > 0 else 0
            self.metrics['profit_factor'] = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in trades if t.pnl < 0])) if sum([t.pnl for t in trades if t.pnl < 0]) != 0 else 0
            self.metrics['total_pnl'] = sum([t.pnl for t in trades])
    
    def _calculate_max_drawdown_pct(self, equity_curve: np.array) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown_pct = (peak - equity_curve) / peak
        return np.max(drawdown_pct) if len(drawdown_pct) > 0 else 0


class SimpleStrategy(Strategy):
    """A simple strategy that buys when price is below 20-day MA and sells when above"""
    
    def __init__(self, symbol: str = 'default'):
        super().__init__(name="Simple_Strategy")
        self.symbol = symbol
        self.in_position = False
        
    def on_data(self, timestamp: datetime):
        data = self.data_handler.get_latest_data(self.symbol, 30)
        
        if len(data) < 20:
            return
            
        current_price = data['close'].iloc[-1]
        ma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        
        # Buy when price crosses below MA (potential oversold)
        if not self.in_position and current_price < ma_20 * 0.98:  # 2% below MA
            quantity = int(self.portfolio.cash / current_price * 0.5)  # Use 50% of cash
            if quantity > 0:
                if self.buy(self.symbol, quantity):
                    self.in_position = True
                    print(f"BUY: {quantity} shares at ${current_price:.2f}")
        
        # Sell when price crosses above MA or take profit
        elif self.in_position:
            position = self.portfolio.positions.get(self.symbol)
            if position and position.quantity > 0:
                if current_price > ma_20 * 1.02:  # 2% above MA
                    if self.sell(self.symbol, position.quantity):
                        self.in_position = False
                        print(f"SELL: {position.quantity} shares at ${current_price:.2f}")


def generate_sample_data(days: int = 500):
    """Generate more realistic sample data with trends"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # Create a trending price series
    trend = np.linspace(0, 20, days)  # Upward trend
    noise = np.random.randn(days) * 2  # Random noise
    cycles = 10 * np.sin(np.linspace(0, 4*np.pi, days))  # Cyclical component
    
    prices = 100 + trend + cycles + noise
    prices = np.maximum(prices, 10)  # Ensure no negative prices
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.005),
        'high': prices * (1 + abs(np.random.randn(days) * 0.01)),
        'low': prices * (1 - abs(np.random.randn(days) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, days)
    }, index=dates)
    
    return df


def run_backtest():
    """Run a sample backtest"""
    print("="*50)
    print("BACKTEST ENGINE")
    print("="*50)
    
    print("\n1. Generating sample data...")
    data = generate_sample_data(500)
    print(f"   Generated {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    
    print("\n2. Creating strategy...")
    strategy = SimpleStrategy()
    print(f"   Strategy: {strategy.name}")
    
    print("\n3. Initializing backtest engine...")
    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=100000,
        commission=0.001,
        slippage=0.001
    )
    print(f"   Initial capital: ${engine.portfolio.initial_capital:,.2f}")
    print(f"   Commission: {engine.commission*100}%")
    print(f"   Slippage: {engine.slippage*100}%")
    
    print("\n4. Running backtest...")
    results = engine.run()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    # Format and display results
    for key, value in results.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'rate' in key.lower() or 'drawdown' in key.lower():
                print(f"{key:20s}: {value:.2f}%")
            elif 'sharpe' in key.lower():
                print(f"{key:20s}: {value:.3f}")
            elif 'pnl' in key.lower() or 'equity' in key.lower() or 'capital' in key.lower():
                print(f"{key:20s}: ${value:,.2f}")
            else:
                print(f"{key:20s}: {value:.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Print trade summary
    if len(engine.portfolio.trades) > 0:
        print(f"\nTRADE SUMMARY")
        print("-" * 50)
        print(f"Total P&L: ${results.get('total_pnl', 0):,.2f}")
        
        if results.get('win_rate', 0) > 0:
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Avg Win: ${results['avg_win']:.2f}")
            print(f"Avg Loss: ${results['avg_loss']:.2f}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\nLast 5 trades:")
        for i, trade in enumerate(engine.portfolio.trades[-5:]):
            print(f"  {i+1}. {trade.symbol} - {trade.quantity} shares - " +
                  f"Entry: ${trade.entry_price:.2f} - " +
                  f"Exit: ${trade.exit_price:.2f} - " +
                  f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1f}%)")
    
    return engine, results


if __name__ == "__main__":
    engine, results = run_backtest()
    
    # Optional: Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print("\n5. Generating plots...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Price and trades
        data = engine.data_handler.data['default']
        ax1.plot(data.index, data['close'], label='Close Price', alpha=0.7)
        
        # Mark buy and sell points
        buys = []
        sells = []
        for trade in engine.portfolio.trades:
            if trade.exit_time:  # Complete trade
                buys.append((trade.entry_time, trade.entry_price))
                sells.append((trade.exit_time, trade.exit_price))
        
        if buys:
            buy_times, buy_prices = zip(*buys)
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy')
        
        if sells:
            sell_times, sell_prices = zip(*sells)
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        ax1.set_title('Price Chart with Trades')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        ax2.plot(engine.portfolio.timestamps, engine.portfolio.equity_curve, 
                label='Portfolio Equity', color='blue')
        ax2.axhline(y=engine.portfolio.initial_capital, color='red', 
                   linestyle='--', label='Initial Capital')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nNote: Install matplotlib to see plots: pip install matplotlib")
