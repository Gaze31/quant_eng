import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA STRUCTURES
# ============================================

class OrderType(Enum):
    """Order types supported"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"  # Immediate-or-Cancel
    FOK = "FOK"  # Fill-or-Kill

class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Individual order in the book"""
    order_id: int
    side: OrderSide
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    timestamp: float = None
    visible_quantity: int = None  # For iceberg orders
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.visible_quantity is None:
            self.visible_quantity = self.quantity
    
    @property
    def is_iceberg(self) -> bool:
        return self.visible_quantity < self.quantity

@dataclass
class Trade:
    """Executed trade"""
    trade_id: int
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: float

class OrderBook:
    """Main order book implementation"""
    
    def __init__(self, tick_size: float = 0.01, lot_size: int = 1):
        """
        Initialize order book
        
        Parameters:
        -----------
        tick_size : float
            Minimum price increment
        lot_size : int
            Minimum quantity increment
        """
        self.tick_size = tick_size
        self.lot_size = lot_size
        
        # Order books: price -> list of orders (FIFO queue)
        self.bids = {}  # Buy orders (highest price first)
        self.asks = {}  # Sell orders (lowest price first)
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.next_order_id = 1
        self.next_trade_id = 1
        
        # Trade history
        self.trades = []
        self.trade_history = deque(maxlen=1000)
        
        # Statistics
        self.best_bid = 0
        self.best_ask = float('inf')
        self.mid_price = 0
        self.spread = 0
        
        # Price levels for quick access
        self.bid_prices = []  # Sorted descending
        self.ask_prices = []  # Sorted ascending
        
        # Performance metrics
        self.total_volume = 0
        self.total_trades = 0
        
    def add_order(self, side: OrderSide, price: float, quantity: int, 
                  order_type: OrderType = OrderType.LIMIT) -> int:
        """
        Add new order to the book
        
        Returns:
        --------
        order_id : int
        """
        # Round to tick/lot sizes
        price = round(price / self.tick_size) * self.tick_size
        quantity = (quantity // self.lot_size) * self.lot_size
        
        if quantity <= 0:
            return -1
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
            timestamp=time.time()
        )
        
        self.orders[order_id] = order
        
        # Handle different order types
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)
        elif order_type in [OrderType.IOC, OrderType.FOK]:
            self._execute_immediate_order(order)
        else:  # LIMIT order
            self._add_limit_order(order)
        
        return order_id
    
    def _add_limit_order(self, order: Order):
        """Add limit order to the book"""
        # Try to match against opposite side first
        if order.side == OrderSide.BUY:
            matched = self._match_order(order, self.asks, ascending=True)
        else:
            matched = self._match_order(order, self.bids, ascending=False)
        
        # If not fully filled and not IOC/FOK, add remaining to book
        if order.quantity > 0 and order.order_type == OrderType.LIMIT:
            book = self.bids if order.side == OrderSide.BUY else self.asks
            if order.price not in book:
                book[order.price] = deque()
                self._update_price_levels()
            
            book[order.price].append(order)
    
    def _match_order(self, order: Order, opposite_book: Dict, ascending: bool) -> bool:
        """Match order against opposite book"""
        prices = sorted(opposite_book.keys(), reverse=not ascending)
        
        for price in prices:
            if (order.side == OrderSide.BUY and price > order.price) or \
               (order.side == OrderSide.SELL and price < order.price):
                continue
            
            while opposite_book[price] and order.quantity > 0:
                opposite_order = opposite_book[price][0]
                
                # Determine match quantity
                match_qty = min(order.quantity, opposite_order.quantity)
                
                # Execute trade
                self._execute_trade(order, opposite_order, price, match_qty)
                
                # Update quantities
                order.quantity -= match_qty
                opposite_order.quantity -= match_qty
                
                # Remove fully filled orders
                if opposite_order.quantity == 0:
                    opposite_book[price].popleft()
                    if not opposite_book[price]:
                        del opposite_book[price]
                        self._update_price_levels()
            
            if order.quantity == 0:
                break
        
        return order.quantity == 0
    
    def _execute_trade(self, buy_order: Order, sell_order: Order, 
                       price: float, quantity: int):
        """Execute a trade between two orders"""
        trade = Trade(
            trade_id=self.next_trade_id,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            price=price,
            quantity=quantity,
            timestamp=time.time()
        )
        
        self.next_trade_id += 1
        self.trades.append(trade)
        self.trade_history.append(trade)
        self.total_volume += quantity
        self.total_trades += 1
    
    def _execute_market_order(self, order: Order):
        """Execute market order"""
        if order.side == OrderSide.BUY:
            self._match_order(order, self.asks, ascending=True)
        else:
            self._match_order(order, self.bids, ascending=False)
        
        # Market orders are fully executed or fail
        if order.quantity > 0:
            print(f"Warning: Market order {order.order_id} only partially filled")
    
    def _execute_immediate_order(self, order: Order):
        """Execute IOC or FOK order"""
        original_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            self._match_order(order, self.asks, ascending=True)
        else:
            self._match_order(order, self.bids, ascending=False)
        
        # FOK orders must be fully filled
        if order.order_type == OrderType.FOK and order.quantity > 0:
            # Cancel the trade
            order.quantity = original_qty
    
    def _update_price_levels(self):
        """Update sorted price levels"""
        self.bid_prices = sorted(self.bids.keys(), reverse=True)
        self.ask_prices = sorted(self.asks.keys())
        
        self.best_bid = self.bid_prices[0] if self.bid_prices else 0
        self.best_ask = self.ask_prices[0] if self.ask_prices else float('inf')
        self.mid_price = (self.best_bid + self.best_ask) / 2 if self.best_bid and self.best_ask < float('inf') else 0
        self.spread = self.best_ask - self.best_bid if self.best_ask < float('inf') and self.best_bid > 0 else 0
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        book = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price in book:
            # Find and remove order
            for i, o in enumerate(book[order.price]):
                if o.order_id == order_id:
                    del book[order.price][i]
                    if not book[order.price]:
                        del book[order.price]
                        self._update_price_levels()
                    del self.orders[order_id]
                    return True
        
        return False
    
    def get_order_book_snapshot(self, levels: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get current order book snapshot"""
        bids_data = []
        for i, price in enumerate(self.bid_prices[:levels]):
            total_qty = sum(o.quantity for o in self.bids[price])
            bids_data.append({
                'price': price,
                'quantity': total_qty,
                'orders': len(self.bids[price])
            })
        
        asks_data = []
        for i, price in enumerate(self.ask_prices[:levels]):
            total_qty = sum(o.quantity for o in self.asks[price])
            asks_data.append({
                'price': price,
                'quantity': total_qty,
                'orders': len(self.asks[price])
            })
        
        bids_df = pd.DataFrame(bids_data)
        asks_df = pd.DataFrame(asks_data)
        
        return bids_df, asks_df
    
    def get_market_depth(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get market depth for visualization"""
        bid_prices = []
        bid_quantities = []
        ask_prices = []
        ask_quantities = []
        
        for price in self.bid_prices[:20]:
            total_qty = sum(o.quantity for o in self.bids[price])
            bid_prices.append(price)
            bid_quantities.append(total_qty)
        
        for price in self.ask_prices[:20]:
            total_qty = sum(o.quantity for o in self.asks[price])
            ask_prices.append(price)
            ask_quantities.append(total_qty)
        
        return (np.array(bid_prices), np.array(bid_quantities),
                np.array(ask_prices), np.array(ask_quantities))
    
    def get_stats(self) -> Dict:
        """Get order book statistics"""
        return {
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'total_orders': len(self.orders),
            'total_trades': self.total_trades,
            'total_volume': self.total_volume
        }

# ============================================
# MARKET MAKER AGENT
# ============================================

class MarketMaker:
    """Automated market maker agent"""
    
    def __init__(self, order_book: OrderBook, symbol: str = "AAPL"):
        self.order_book = order_book
        self.symbol = symbol
        self.inventory = 0
        self.cash = 1000000  # Starting cash
        self.orders_placed = []
        
        # Parameters
        self.position_limit = 1000
        inventory_scale = 0.1
        self.base_spread = 0.10  # $0.10 spread
        self.order_size = 100  # Base order size
        
    def update_quotes(self):
        """Update market making quotes"""
        if self.order_book.mid_price == 0:
            return
        
        # Adjust spread based on inventory
        inventory_adjustment = (self.inventory / self.position_limit) * self.base_spread
        
        # Calculate bid and ask prices
        bid_price = self.order_book.mid_price - self.base_spread/2 - inventory_adjustment
        ask_price = self.order_book.mid_price + self.base_spread/2 - inventory_adjustment
        
        # Round to tick size
        bid_price = round(bid_price / self.order_book.tick_size) * self.order_book.tick_size
        ask_price = round(ask_price / self.order_book.tick_size) * self.order_book.tick_size
        
        # Adjust size based on distance from best prices
        bid_size = self.order_size
        ask_size = self.order_size
        
        # Place orders
        if bid_price < ask_price:
            bid_id = self.order_book.add_order(OrderSide.BUY, bid_price, bid_size)
            ask_id = self.order_book.add_order(OrderSide.SELL, ask_price, ask_size)
            
            self.orders_placed.extend([bid_id, ask_id])
    
    def on_trade(self, trade: Trade):
        """Handle trade execution"""
        if trade.price < self.order_book.mid_price:
            # We bought
            self.inventory += trade.quantity
            self.cash -= trade.price * trade.quantity
        else:
            # We sold
            self.inventory -= trade.quantity
            self.cash += trade.price * trade.quantity
    
    def get_pnl(self) -> float:
        """Calculate P&L"""
        mark_to_market = self.inventory * self.order_book.mid_price
        return self.cash + mark_to_market - 1000000  # Subtract initial cash

# ============================================
# ORDER BOOK VISUALIZER
# ============================================

class OrderBookVisualizer:
    """Real-time order book visualization"""
    
    def __init__(self, order_book: OrderBook, figsize=(15, 10)):
        self.order_book = order_book
        self.figsize = figsize
        
        # Setup figure
        self.fig = plt.figure(figsize=figsize)
        self.setup_plots()
        
    def setup_plots(self):
        """Setup subplots"""
        # Main order book depth chart
        self.ax1 = plt.subplot(2, 3, (1, 2))
        self.ax1.set_title('Order Book Depth')
        self.ax1.set_xlabel('Price')
        self.ax1.set_ylabel('Quantity')
        self.ax1.grid(True, alpha=0.3)
        
        # Price-time series
        self.ax2 = plt.subplot(2, 3, 3)
        self.ax2.set_title('Price History')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Price')
        self.ax2.grid(True, alpha=0.3)
        
        # Trade history
        self.ax3 = plt.subplot(2, 3, (4, 5))
        self.ax3.set_title('Trade History')
        self.ax3.set_xlabel('Trade #')
        self.ax3.set_ylabel('Price')
        self.ax3.grid(True, alpha=0.3)
        
        # Stats
        self.ax4 = plt.subplot(2, 3, 6)
        self.ax4.set_title('Statistics')
        self.ax4.axis('off')
        
        # Store data for animation
        self.price_history = deque(maxlen=100)
        self.trade_prices = []
        self.trade_sizes = []
        
    def update(self, frame):
        """Update animation frame"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Reset titles and labels
        self.ax1.set_title('Order Book Depth')
        self.ax1.set_xlabel('Price')
        self.ax1.set_ylabel('Quantity')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Price History')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Price')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Trade History')
        self.ax3.set_xlabel('Trade #')
        self.ax3.set_ylabel('Price')
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 1: Order book depth
        bid_prices, bid_qty, ask_prices, ask_qty = self.order_book.get_market_depth()
        
        if len(bid_prices) > 0:
            self.ax1.bar(bid_prices, bid_qty, width=self.order_book.tick_size*3, 
                        color='green', alpha=0.6, label='Bids')
        if len(ask_prices) > 0:
            self.ax1.bar(ask_prices, ask_qty, width=self.order_book.tick_size*3, 
                        color='red', alpha=0.6, label='Asks')
        
        self.ax1.legend()
        
        # Add mid price line
        if self.order_book.mid_price > 0:
            self.ax1.axvline(x=self.order_book.mid_price, color='blue', 
                           linestyle='--', alpha=0.5, label='Mid Price')
        
        # Plot 2: Price history
        if self.order_book.mid_price > 0:
            self.price_history.append(self.order_book.mid_price)
        
        if len(self.price_history) > 1:
            self.ax2.plot(list(self.price_history), 'b-', linewidth=1)
        
        # Plot 3: Trade history
        trades = list(self.order_book.trade_history)
        if len(trades) > 0:
            trade_prices = [t.price for t in trades]
            trade_sizes = [t.quantity for t in trades]
            self.ax3.scatter(range(len(trade_prices)), trade_prices, 
                           s=np.array(trade_sizes)/10, alpha=0.5, c='purple')
        
        # Plot 4: Statistics
        stats = self.order_book.get_stats()
        stats_text = f"""
        Best Bid: ${stats['best_bid']:.2f}
        Best Ask: ${stats['best_ask']:.2f}
        Mid Price: ${stats['mid_price']:.2f}
        Spread: ${stats['spread']:.3f}
        
        Bid Levels: {stats['bid_levels']}
        Ask Levels: {stats['ask_levels']}
        Total Orders: {stats['total_orders']}
        
        Trades: {stats['total_trades']}
        Volume: {stats['total_volume']:,}
        """
        
        self.ax4.text(0.1, 0.9, stats_text, transform=self.ax4.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.ax4.axis('off')
        
        plt.tight_layout()

# ============================================
# SIMULATION ENGINE
# ============================================

class OrderBookSimulator:
    """Main simulation engine"""
    
    def __init__(self, initial_price: float = 100.0, tick_size: float = 0.01):
        self.order_book = OrderBook(tick_size=tick_size)
        self.initial_price = initial_price
        self.market_maker = MarketMaker(self.order_book)
        
        # Simulation parameters
        self.current_time = 0
        self.simulation_speed = 1.0
        
        # Statistics
        self.price_history = []
        self.volume_history = []
        
        # Initialize with some orders
        self._initialize_book()
        
    def _initialize_book(self):
        """Initialize order book with some resting orders"""
        # Add some initial liquidity
        np.random.seed(42)
        
        # Add bids at various prices
        for i in range(5):
            price = self.initial_price - 0.05 - i*0.02
            quantity = np.random.randint(100, 500)
            self.order_book.add_order(OrderSide.BUY, price, quantity)
        
        # Add asks at various prices
        for i in range(5):
            price = self.initial_price + 0.05 + i*0.02
            quantity = np.random.randint(100, 500)
            self.order_book.add_order(OrderSide.SELL, price, quantity)
    
    def step(self, num_orders: int = 10):
        """Run one simulation step"""
        for _ in range(num_orders):
            self._generate_random_order()
        
        # Market maker updates periodically
        if np.random.random() < 0.3:  # 30% chance per step
            self.market_maker.update_quotes()
        
        # Record statistics
        self.price_history.append(self.order_book.mid_price)
        self.volume_history.append(self.order_book.total_volume)
        
        self.current_time += 1
    
    def _generate_random_order(self):
        """Generate random order for simulation"""
        side = np.random.choice([OrderSide.BUY, OrderSide.SELL])
        
        # Order type distribution
        order_type = np.random.choice(
            [OrderType.LIMIT, OrderType.MARKET, OrderType.IOC],
            p=[0.7, 0.2, 0.1]
        )
        
        # Generate random price around mid
        if self.order_book.mid_price > 0:
            mid = self.order_book.mid_price
            price = mid + np.random.normal(0, 0.05)
        else:
            price = self.initial_price + np.random.normal(0, 0.05)
        
        # Random quantity
        quantity = np.random.randint(10, 200)
        
        # Add order
        self.order_book.add_order(side, price, quantity, order_type)
    
    def run_simulation(self, steps: int = 1000, visualize: bool = True):
        """Run complete simulation"""
        if visualize:
            visualizer = OrderBookVisualizer(self.order_book)
            anim = FuncAnimation(visualizer.fig, visualizer.update, 
                               interval=100, cache_frame_data=False)
            plt.show()
        else:
            for i in range(steps):
                self.step()
                
                if i % 100 == 0:
                    stats = self.order_book.get_stats()
                    print(f"Step {i}: Mid=${stats['mid_price']:.2f}, "
                          f"Spread=${stats['spread']:.3f}, "
                          f"Trades={stats['total_trades']}")

# ============================================
# ADVANCED FEATURES
# ============================================

class IcebergOrder(Order):
    """Iceberg order with hidden quantity"""
    def __init__(self, *args, peak_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_size = peak_size
        self.hidden_quantity = self.quantity - self.peak_size
        self.visible_quantity = min(self.peak_size, self.quantity)
    
    def refresh(self):
        """Refresh visible quantity"""
        if self.hidden_quantity > 0:
            refresh_qty = min(self.peak_size, self.hidden_quantity)
            self.visible_quantity += refresh_qty
            self.hidden_quantity -= refresh_qty

class TWAPOrder(Order):
    """Time-Weighted Average Price order"""
    def __init__(self, *args, duration: int = 60, slices: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration = duration
        self.slices = slices
        self.slice_size = self.quantity // slices
        self.slices_executed = 0

class OrderBookAnalyzer:
    """Analyze order book dynamics"""
    
    def __init__(self, order_book: OrderBook):
        self.order_book = order_book
    
    def calculate_imbalance(self) -> float:
        """Calculate order book imbalance"""
        bid_volume = sum(sum(o.quantity for o in orders) 
                        for orders in self.order_book.bids.values())
        ask_volume = sum(sum(o.quantity for o in orders) 
                        for orders in self.order_book.asks.values())
        
        if bid_volume + ask_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    def calculate_microprice(self) -> float:
        """Calculate microprice (weighted mid price)"""
        best_bid = self.order_book.best_bid
        best_ask = self.order_book.best_ask
        
        if best_bid == 0 or best_ask == float('inf'):
            return 0
        
        bid_qty = sum(o.quantity for o in self.order_book.bids.get(best_bid, []))
        ask_qty = sum(o.quantity for o in self.order_book.asks.get(best_ask, []))
        
        if bid_qty + ask_qty == 0:
            return (best_bid + best_ask) / 2
        
        return (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)
    
    def get_depth_profile(self, levels: int = 10) -> pd.DataFrame:
        """Get depth profile for analysis"""
        bids_df, asks_df = self.order_book.get_order_book_snapshot(levels)
        
        # Calculate cumulative depth
        bids_df['cumulative_quantity'] = bids_df['quantity'].cumsum()
        asks_df['cumulative_quantity'] = asks_df['quantity'].cumsum()
        
        # Calculate weighted average prices
        bids_df['weighted_price'] = bids_df['price'] * bids_df['quantity']
        asks_df['weighted_price'] = asks_df['price'] * asks_df['quantity']
        
        return bids_df, asks_df

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2 ORDER BOOK SIMULATOR")
    print("=" * 60)
    
    # Create simulator
    simulator = OrderBookSimulator(initial_price=100.0, tick_size=0.01)
    
    # Run simulation with visualization
    print("\nStarting simulation with visualization...")
    print("Close the plot window to stop simulation.\n")
    
    try:
        simulator.run_simulation(steps=1000, visualize=True)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    stats = simulator.order_book.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Analyze order book
    analyzer = OrderBookAnalyzer(simulator.order_book)
    print(f"\nOrder Book Imbalance: {analyzer.calculate_imbalance():.3f}")
    print(f"Microprice: ${analyzer.calculate_microprice():.2f}")
    
    # Get depth profile
    bids_df, asks_df = analyzer.get_depth_profile(levels=5)
    
    print("\nTop 5 Bid Levels:")
    print(bids_df.to_string(index=False))
    
    print("\nTop 5 Ask Levels:")
    print(asks_df.to_string(index=False))
    
    # Market maker P&L
    print(f"\nMarket Maker P&L: ${simulator.market_maker.get_pnl():,.2f}")
    print(f"Market Maker Inventory: {simulator.market_maker.inventory} shares")

# ============================================
# INTERACTIVE TESTING
# ============================================

def interactive_order_entry():
    """Interactive order entry for testing"""
    book = OrderBook(tick_size=0.01)
    
    print("\n" + "=" * 60)
    print("INTERACTIVE ORDER BOOK TESTER")
    print("=" * 60)
    print("Commands:")
    print("  b <price> <qty> - Place buy limit order")
    print("  s <price> <qty> - Place sell limit order")
    print("  m b <qty>      - Place buy market order")
    print("  m s <qty>      - Place sell market order")
    print("  c <order_id>    - Cancel order")
    print("  q              - Quit")
    print("-" * 60)
    
    while True:
        # Show current book
        bids_df, asks_df = book.get_order_book_snapshot(levels=3)
        print(f"\nBest Bid: ${book.best_bid:.2f} | Best Ask: ${book.best_ask:.2f} | Spread: ${book.spread:.3f}")
        print(f"Bids: {bids_df.to_string(index=False)}")
        print(f"Asks: {asks_df.to_string(index=False)}")
        
        # Get user input
        cmd = input("\nEnter command: ").strip().split()
        
        if not cmd:
            continue
        
        if cmd[0].lower() == 'q':
            break
        
        try:
            if cmd[0].lower() == 'b' and len(cmd) == 3:
                price = float(cmd[1])
                qty = int(cmd[2])
                order_id = book.add_order(OrderSide.BUY, price, qty)
                print(f"Placed buy order {order_id}")
            
            elif cmd[0].lower() == 's' and len(cmd) == 3:
                price = float(cmd[1])
                qty = int(cmd[2])
                order_id = book.add_order(OrderSide.SELL, price, qty)
                print(f"Placed sell order {order_id}")
            
            elif cmd[0].lower() == 'm' and len(cmd) == 3:
                side = cmd[1].lower()
                qty = int(cmd[2])
                if side == 'b':
                    order_id = book.add_order(OrderSide.BUY, 0, qty, OrderType.MARKET)
                else:
                    order_id = book.add_order(OrderSide.SELL, 0, qty, OrderType.MARKET)
                print(f"Placed market order {order_id}")
            
            elif cmd[0].lower() == 'c' and len(cmd) == 2:
                order_id = int(cmd[1])
                if book.cancel_order(order_id):
                    print(f"Cancelled order {order_id}")
                else:
                    print(f"Order {order_id} not found")
            
        except Exception as e:
            print(f"Error: {e}")

# Uncomment to run interactive tester
# interactive_order_entry()