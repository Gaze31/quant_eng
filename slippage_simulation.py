import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SLIPPAGE MODELS
# ============================================

@dataclass
class MarketState:
    """Current market conditions"""
    mid_price: float
    spread: float
    bid_depth: float      # volume at best bid
    ask_depth: float      # volume at best ask
    volatility: float     # 1-min volatility
    volume_30min: float   # last 30 min volume
    momentum: float       # short-term price trend

class SlippageSimulator:
    """
    Comprehensive slippage simulator with multiple models
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.trades = []
        self.slippage_history = []
        
    # ============================================
    # 1. CONSTANT SLIPPAGE MODEL
    # ============================================
    def constant_slippage(self, 
                          order_size: float,
                          side: str,
                          base_slippage: float = 0.001) -> float:
        """
        Simplest model: fixed slippage regardless of size
        """
        sign = 1 if side.lower() == 'buy' else -1
        slippage = base_slippage * sign
        execution_price = 100 * (1 + slippage)  # assume price=100
        return execution_price, slippage * 100  # return % slippage
    
    # ============================================
    # 2. SPREAD-ONLY SLIPPAGE
    # ============================================
    def spread_slippage(self,
                       order_size: float,
                       side: str,
                       spread: float = 0.02) -> Tuple[float, float]:
        """
        Slippage from crossing the spread only
        """
        half_spread = spread / 2
        sign = 1 if side.lower() == 'buy' else -1
        
        # Buys pay ask (mid + half-spread), sells receive bid (mid - half-spread)
        execution_price = 100 + sign * half_spread
        slippage_pct = (sign * half_spread / 100) * 100  # in percent
        
        return execution_price, slippage_pct
    
    # ============================================
    # 3. LINEAR IMPACT MODEL
    # ============================================
    def linear_impact(self,
                     order_size: float,
                     side: str,
                     mid_price: float = 100.0,
                     spread: float = 0.02,
                     impact_coeff: float = 0.0001) -> Tuple[float, float]:
        """
        Linear price impact: ΔP = η × Q
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        # Spread cost
        half_spread = spread / 2
        spread_cost = half_spread * sign
        
        # Impact cost (linear in size)
        impact_cost = impact_coeff * order_size * sign
        
        total_impact = spread_cost + impact_cost
        execution_price = mid_price * (1 + total_impact / mid_price)
        slippage_pct = (total_impact / mid_price) * 100
        
        return execution_price, slippage_pct
    
    # ============================================
    # 4. SQUARE ROOT IMPACT (EMPIRICAL)
    # ============================================
    def square_root_impact(self,
                          order_size: float,
                          side: str,
                          mid_price: float = 100.0,
                          spread: float = 0.02,
                          volatility: float = 0.02,  # daily vol
                          adv: float = 1_000_000,    # avg daily volume
                          gamma: float = 0.3) -> Tuple[float, float]:
        """
        Square root impact law: ΔP/P = γ·σ·√(Q/ADV)
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        # Spread cost
        half_spread = spread / 2
        spread_cost = half_spread * sign / mid_price
        
        # Square root impact
        impact = gamma * volatility * np.sqrt(order_size / adv)
        total_impact = spread_cost + impact * sign
        
        execution_price = mid_price * (1 + total_impact)
        slippage_pct = total_impact * 100
        
        return execution_price, slippage_pct
    
    # ============================================
    # 5. ALMGREN-CHRISS IMPACT MODEL
    # ============================================
    def almgren_chriss_impact(self,
                            order_size: float,
                            side: str,
                            mid_price: float = 100.0,
                            spread: float = 0.02,
                            gamma: float = 2.5e-7,   # permanent impact
                            eta: float = 2.5e-6,      # temporary impact
                            trading_rate: float = None) -> Tuple[float, float]:
        """
        Almgren-Chriss impact decomposition
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        if trading_rate is None:
            trading_rate = order_size / 60  # assume 60 min execution
        
        # Spread cost
        half_spread = spread / 2
        spread_cost = half_spread * sign / mid_price
        
        # Permanent impact (affects all future trades)
        perm_impact = gamma * order_size * sign / mid_price
        
        # Temporary impact (one-time cost)
        temp_impact = eta * trading_rate * sign / mid_price
        
        total_impact = spread_cost + perm_impact + temp_impact
        execution_price = mid_price * (1 + total_impact)
        slippage_pct = total_impact * 100
        
        return execution_price, slippage_pct
    
    # ============================================
    # 6. ORDER BOOK WALK MODEL
    # ============================================
    def order_book_walk(self,
                       order_size: float,
                       side: str,
                       lob: Dict[float, float]) -> Tuple[float, float, List]:
        """
        Simulate walking through actual limit order book
        lob = {price: quantity} for relevant side
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        if side.lower() == 'buy':
            # Sort asks ascending (best ask first)
            levels = sorted(lob.items())
        else:
            # Sort bids descending (best bid first)
            levels = sorted(lob.items(), reverse=True)
        
        remaining = order_size
        total_cost = 0
        execution_path = []
        weighted_avg_price = 0
        
        for price, qty in levels:
            if remaining <= 0:
                break
            
            fill = min(remaining, qty)
            total_cost += fill * price
            remaining -= fill
            execution_path.append((price, fill))
        
        if remaining > 0:
            print(f"Warning: Only filled {order_size - remaining}/{order_size}")
        
        avg_price = total_cost / (order_size - remaining) if remaining < order_size else 0
        slippage_pct = ((avg_price - 100) / 100) * 100 * sign
        
        return avg_price, slippage_pct, execution_path
    
    # ============================================
    # 7. KXYLE'S LAMBDA MODEL
    # ============================================
    def kyle_lambda(self,
                   order_size: float,
                   side: str,
                   mid_price: float = 100.0,
                   lambda_coeff: float = 0.001) -> Tuple[float, float]:
        """
        Kyle's lambda: ΔP = λ × Q
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        price_impact = lambda_coeff * order_size * sign
        execution_price = mid_price + price_impact
        slippage_pct = (price_impact / mid_price) * 100
        
        return execution_price, slippage_pct
    
    # ============================================
    # 8. STOCHASTIC SLIPPAGE (WITH VOLATILITY)
    # ============================================
    def stochastic_slippage(self,
                           order_size: float,
                           side: str,
                           mid_price: float = 100.0,
                           spread: float = 0.02,
                           volatility: float = 0.02,
                           execution_time: float = 60,  # seconds
                           n_steps: int = 10) -> Tuple[float, float, List]:
        """
        Simulate slippage with random price moves during execution
        """
        sign = 1 if side.lower() == 'buy' else -1
        half_spread = spread / 2
        
        dt = execution_time / n_steps
        order_per_step = order_size / n_steps
        
        prices = []
        total_cost = 0
        
        for i in range(n_steps):
            # Random price move during execution
            price_move = np.random.normal(0, volatility * np.sqrt(dt/252/6.5/3600))
            current_mid = mid_price * (1 + price_move * (i+1))
            
            # Execution price includes spread
            exec_price = current_mid + sign * half_spread
            
            # Add impact (simplified)
            impact = 0.0001 * order_per_step * sign
            exec_price += impact
            
            total_cost += exec_price * order_per_step
            prices.append(exec_price)
        
        avg_price = total_cost / order_size
        expected_price = mid_price + sign * half_spread
        slippage_pct = ((avg_price - expected_price) / mid_price) * 100
        
        return avg_price, slippage_pct, prices
    
    # ============================================
    # 9. REALISTIC COMBINED MODEL
    # ============================================
    def realistic_slippage(self,
                          order_size: float,
                          side: str,
                          market: MarketState) -> Dict:
        """
        Combine multiple factors for realistic slippage
        """
        sign = 1 if side.lower() == 'buy' else -1
        
        # 1. Spread cost
        half_spread = market.spread / 2
        spread_cost = half_spread * sign
        
        # 2. Market impact (square root law)
        volume_ratio = order_size / market.volume_30min
        impact = 0.1 * market.volatility * np.sqrt(volume_ratio) * sign
        
        # 3. Momentum effect
        momentum_effect = market.momentum * sign * 0.5
        
        # 4. Depth penalty (if order > available depth)
        available_depth = market.ask_depth if side == 'buy' else market.bid_depth
        if order_size > available_depth:
            depth_penalty = 0.1 * np.log(order_size / available_depth) * sign
        else:
            depth_penalty = 0
        
        # 5. Random noise
        noise = np.random.normal(0, 0.0001) * sign
        
        total_impact = (spread_cost + impact + momentum_effect + 
                       depth_penalty + noise) / market.mid_price
        
        execution_price = market.mid_price * (1 + total_impact)
        slippage_bps = total_impact * 10000  # convert to basis points
        
        return {
            'execution_price': execution_price,
            'slippage_bps': slippage_bps,
            'components': {
                'spread': spread_cost / market.mid_price * 10000,
                'impact': impact / market.mid_price * 10000,
                'momentum': momentum_effect / market.mid_price * 10000,
                'depth_penalty': depth_penalty / market.mid_price * 10000,
                'noise': noise / market.mid_price * 10000
            }
        }

# ============================================
# SLIPPAGE VISUALIZER
# ============================================

class SlippageVisualizer:
    """Visualize slippage patterns"""
    
    def __init__(self):
        self.colors = {'buy': 'green', 'sell': 'red'}
        
    def plot_slippage_curves(self, simulator: SlippageSimulator):
        """Plot slippage vs order size for different models"""
        sizes = np.logspace(1, 6, 50)  # 10 to 1,000,000 shares
        
        models = {
            'Constant': [],
            'Spread Only': [],
            'Linear Impact': [],
            'Square Root': [],
            'Kyle Lambda': [],
            'Almgren-Chriss': []
        }
        
        for size in sizes:
            # Constant
            _, slip = simulator.constant_slippage(size, 'buy')
            models['Constant'].append(abs(slip))
            
            # Spread Only
            _, slip = simulator.spread_slippage(size, 'buy', spread=0.02)
            models['Spread Only'].append(abs(slip))
            
            # Linear Impact
            _, slip = simulator.linear_impact(size, 'buy', impact_coeff=0.00001)
            models['Linear Impact'].append(abs(slip))
            
            # Square Root
            _, slip = simulator.square_root_impact(size, 'buy', adv=2_000_000)
            models['Square Root'].append(abs(slip))
            
            # Kyle Lambda
            _, slip = simulator.kyle_lambda(size, 'buy', lambda_coeff=0.000005)
            models['Kyle Lambda'].append(abs(slip))
            
            # Almgren-Chriss
            _, slip = simulator.almgren_chriss_impact(size, 'buy')
            models['Almgren-Chriss'].append(abs(slip))
        
        plt.figure(figsize=(12, 8))
        for name, values in models.items():
            plt.loglog(sizes, values, linewidth=2, label=name)
        
        plt.xlabel('Order Size (shares)')
        plt.ylabel('Slippage (%)')
        plt.title('Slippage Models Comparison (Log-Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_order_book_walk(self, simulator: SlippageSimulator):
        """Visualize walking through order book"""
        # Create sample LOB
        lob = {100.01: 1000, 100.02: 2000, 100.03: 3000, 
               100.04: 4000, 100.05: 5000}
        
        sizes = [1000, 3000, 6000, 10000, 15000]
        
        plt.figure(figsize=(12, 6))
        
        for size in sizes:
            price, slip, path = simulator.order_book_walk(size, 'buy', lob)
            
            # Plot execution path
            prices = [p for p, _ in path]
            cum_qty = 0
            x_pos = []
            for _, qty in path:
                cum_qty += qty
                x_pos.append(cum_qty)
            
            plt.plot(x_pos, prices, 'o-', label=f'Size={size}', 
                    markersize=8, linewidth=2)
        
        plt.xlabel('Cumulative Quantity Executed')
        plt.ylabel('Execution Price')
        plt.title('Order Book Walk: Price Impact by Order Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_slippage_distribution(self, simulator: SlippageSimulator, n_sims: int = 1000):
        """Plot distribution of stochastic slippage"""
        slippages = []
        
        for _ in range(n_sims):
            _, slip, _ = simulator.stochastic_slippage(
                order_size=10000,
                side='buy',
                volatility=0.02,
                execution_time=60
            )
            slippages.append(slip)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(slippages, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Slippage (%)')
        plt.ylabel('Frequency')
        plt.title('Slippage Distribution')
        plt.axvline(np.mean(slippages), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(slippages):.4f}%')
        plt.axvline(np.percentile(slippages, 95), color='orange', 
                   linestyle='--', label=f'95%: {np.percentile(slippages, 95):.4f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(slippages)
        plt.ylabel('Slippage (%)')
        plt.title('Slippage Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_slippage_components(self, simulator: SlippageSimulator):
        """Plot breakdown of slippage components"""
        market = MarketState(
            mid_price=100.0,
            spread=0.02,
            bid_depth=5000,
            ask_depth=5000,
            volatility=0.02,
            volume_30min=1_000_000,
            momentum=0.001
        )
        
        sizes = [1000, 10000, 100000, 500000]
        results = []
        
        for size in sizes:
            result = simulator.realistic_slippage(size, 'buy', market)
            results.append({
                'size': size,
                **result['components']
            })
        
        df = pd.DataFrame(results)
        df.set_index('size', inplace=True)
        
        ax = df.plot(kind='bar', stacked=True, figsize=(12, 6),
                    color=['blue', 'green', 'orange', 'red', 'gray'])
        plt.xlabel('Order Size')
        plt.ylabel('Slippage (bps)')
        plt.title('Slippage Component Breakdown by Order Size')
        plt.legend(title='Component')
        plt.grid(True, alpha=0.3, axis='y')
        plt.show()

# ============================================
# SLIPPAGE ANALYZER
# ============================================

class SlippageAnalyzer:
    """Analyze and compare slippage models"""
    
    def __init__(self, simulator: SlippageSimulator):
        self.simulator = simulator
        
    def compare_models(self, order_size: float = 10000):
        """Compare all models for same order size"""
        models = {
            'Constant': self.simulator.constant_slippage(order_size, 'buy'),
            'Spread Only': self.simulator.spread_slippage(order_size, 'buy'),
            'Linear Impact': self.simulator.linear_impact(order_size, 'buy'),
            'Square Root': self.simulator.square_root_impact(order_size, 'buy'),
            'Kyle Lambda': self.simulator.kyle_lambda(order_size, 'buy'),
            'Almgren-Chriss': self.simulator.almgren_chriss_impact(order_size, 'buy')
        }
        
        results = []
        for name, (price, slip) in models.items():
            results.append({
                'Model': name,
                'Execution Price': price,
                'Slippage (%)': slip
            })
        
        df = pd.DataFrame(results)
        print("\nModel Comparison (Order Size = {:,} shares):".format(order_size))
        print("=" * 70)
        print(df.to_string(index=False))
        return df
    
    def analyze_volume_impact(self, sizes: List[float]):
        """Analyze how volume affects slippage"""
        results = []
        
        for size in sizes:
            for side in ['buy', 'sell']:
                _, slip_sq = self.simulator.square_root_impact(size, side, adv=2_000_000)
                _, slip_lin = self.simulator.linear_impact(size, side, impact_coeff=0.00001)
                _, slip_kyle = self.simulator.kyle_lambda(size, side, lambda_coeff=0.000005)
                
                results.append({
                    'Size': size,
                    'Side': side,
                    'Square Root': abs(slip_sq),
                    'Linear': abs(slip_lin),
                    'Kyle': abs(slip_kyle)
                })
        
        df = pd.DataFrame(results)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, side in zip(axes, ['buy', 'sell']):
            side_data = df[df['Side'] == side]
            ax.plot(side_data['Size'], side_data['Square Root'], 'o-', label='Square Root')
            ax.plot(side_data['Size'], side_data['Linear'], 's-', label='Linear')
            ax.plot(side_data['Size'], side_data['Kyle'], '^-', label='Kyle')
            ax.set_xlabel('Order Size')
            ax.set_ylabel('Slippage (%)')
            ax.set_title(f'{side.capitalize()} Side')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        return df
    
    def calculate_slippage_cost(self, 
                               order_size: float,
                               side: str,
                               n_trades: int = 100,
                               model: str = 'square_root') -> Dict:
        """Calculate total slippage cost for multiple trades"""
        total_slippage = 0
        total_cost = 0
        
        for _ in range(n_trades):
            if model == 'square_root':
                price, slip = self.simulator.square_root_impact(order_size, side)
            elif model == 'linear':
                price, slip = self.simulator.linear_impact(order_size, side)
            elif model == 'kyle':
                price, slip = self.simulator.kyle_lambda(order_size, side)
            else:
                price, slip = self.simulator.realistic_slippage(
                    order_size, side, MarketState(100, 0.02, 5000, 5000, 0.02, 1e6, 0)
                )['execution_price'], 0
            
            total_slippage += abs(slip) * order_size * price / 100
            total_cost += order_size * price
        
        return {
            'total_slippage_cost': total_slippage,
            'total_traded_value': total_cost,
            'avg_slippage_pct': total_slippage / total_cost * 100,
            'avg_slippage_per_trade': total_slippage / n_trades
        }

# ============================================
# REAL-TIME SLIPPAGE MONITOR
# ============================================

class SlippageMonitor:
    """Monitor slippage in real-time trading"""
    
    def __init__(self, simulator: SlippageSimulator):
        self.simulator = simulator
        self.trades = []
        self.slippages = []
        
    def execute_trade(self, 
                     order_size: float,
                     side: str,
                     expected_price: float,
                     market_condition: str = 'normal'):
        """Execute a trade and record slippage"""
        
        # Adjust impact based on market condition
        if market_condition == 'normal':
            impact_coeff = 0.0001
            volatility = 0.02
        elif market_condition == 'illiquid':
            impact_coeff = 0.0005
            volatility = 0.03
        elif market_condition == 'stressed':
            impact_coeff = 0.001
            volatility = 0.05
        else:  # crisis
            impact_coeff = 0.002
            volatility = 0.10
        
        # Simulate execution with stochastic elements
        exec_price, slip, _ = self.simulator.stochastic_slippage(
            order_size=order_size,
            side=side,
            mid_price=expected_price,
            spread=0.02 * (1 + volatility * 10),
            volatility=volatility,
            execution_time=np.random.uniform(10, 120)
        )
        
        trade = {
            'timestamp': pd.Timestamp.now(),
            'size': order_size,
            'side': side,
            'expected_price': expected_price,
            'execution_price': exec_price,
            'slippage_pct': slip,
            'slippage_cost': abs(slip/100) * order_size * exec_price,
            'market_condition': market_condition
        }
        
        self.trades.append(trade)
        self.slippages.append(slip)
        
        return trade
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of slippage"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        return {
            'total_trades': len(df),
            'total_slippage_cost': df['slippage_cost'].sum(),
            'avg_slippage_pct': df['slippage_pct'].mean(),
            'median_slippage_pct': df['slippage_pct'].median(),
            'std_slippage_pct': df['slippage_pct'].std(),
            'p95_slippage': df['slippage_pct'].quantile(0.95),
            'max_slippage': df['slippage_pct'].max(),
            'min_slippage': df['slippage_pct'].min(),
            'positive_slippage_trades': (df['slippage_pct'] > 0).sum(),
            'negative_slippage_trades': (df['slippage_pct'] < 0).sum()
        }
    
    def plot_monitor_results(self):
        """Plot monitoring results"""
        if not self.trades:
            print("No trades to plot")
            return
        
        df = pd.DataFrame(self.trades)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Slippage over time
        axes[0, 0].plot(df['timestamp'], df['slippage_pct'], 'o-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Slippage (%)')
        axes[0, 0].set_title('Slippage Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Slippage distribution by market condition
        conditions = df['market_condition'].unique()
        for cond in conditions:
            data = df[df['market_condition'] == cond]['slippage_pct']
            axes[0, 1].hist(data, bins=20, alpha=0.5, label=cond, density=True)
        axes[0, 1].set_xlabel('Slippage (%)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Slippage Distribution by Market Condition')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Slippage vs Order Size
        axes[1, 0].scatter(df['size'], df['slippage_pct'], 
                          c=df['slippage_cost'], alpha=0.6, cmap='RdYlBu_r')
        axes[1, 0].set_xlabel('Order Size')
        axes[1, 0].set_ylabel('Slippage (%)')
        axes[1, 0].set_title('Slippage vs Order Size')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Cost ($)')
        
        # 4. Cumulative slippage cost
        df['cumulative_cost'] = df['slippage_cost'].cumsum()
        axes[1, 1].fill_between(df['timestamp'], 0, df['cumulative_cost'], alpha=0.5)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative Slippage Cost ($)')
        axes[1, 1].set_title('Cumulative Slippage Cost')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("=" * 60)
    print("SLIPPAGE SIMULATION - COMPLETE ANALYSIS")
    print("=" * 60)
    
    # Initialize simulator
    sim = SlippageSimulator(seed=42)
    viz = SlippageVisualizer()
    analyzer = SlippageAnalyzer(sim)
    
    # 1. Compare slippage curves
    print("\n📊 1. Comparing Slippage Models...")
    viz.plot_slippage_curves(sim)
    
    # 2. Compare specific order size
    print("\n📊 2. Model Comparison for 10,000 shares...")
    df_comparison = analyzer.compare_models(order_size=10000)
    
    # 3. Order book walk simulation
    print("\n📊 3. Simulating Order Book Walk...")
    viz.plot_order_book_walk(sim)
    
    # 4. Stochastic slippage distribution
    print("\n📊 4. Stochastic Slippage Distribution...")
    viz.plot_slippage_distribution(sim, n_sims=1000)
    
    # 5. Slippage components breakdown
    print("\n📊 5. Slippage Component Analysis...")
    viz.plot_slippage_components(sim)
    
    # 6. Volume impact analysis
    print("\n📊 6. Volume Impact Analysis...")
    sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    df_volume = analyzer.analyze_volume_impact(sizes)
    
    # 7. Real-time monitoring simulation
    print("\n📊 7. Real-time Slippage Monitoring...")
    monitor = SlippageMonitor(sim)
    
    # Simulate 100 trades under different conditions
    conditions = ['normal'] * 40 + ['illiquid'] * 30 + ['stressed'] * 20 + ['crisis'] * 10
    np.random.shuffle(conditions)
    
    for i, cond in enumerate(conditions):
        size = np.random.choice([1000, 5000, 10000, 25000])
        side = np.random.choice(['buy', 'sell'])
        expected = 100 * (1 + np.random.normal(0, 0.001))
        
        monitor.execute_trade(size, side, expected, cond)
    
    # Show summary statistics
    stats = monitor.get_summary_stats()
    print("\n📊 Monitoring Summary:")
    print("=" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot monitoring results
    monitor.plot_monitor_results()
    
    # 8. Calculate total slippage cost
    print("\n📊 8. Slippage Cost Analysis...")
    for model in ['square_root', 'linear', 'kyle']:
        cost = analyzer.calculate_slippage_cost(
            order_size=10000,
            side='buy',
            n_trades=100,
            model=model
        )
        print(f"\n{model.upper()} Model:")
        print(f"  Total Slippage Cost: ${cost['total_slippage_cost']:,.2f}")
        print(f"  Avg Slippage per Trade: {cost['avg_slippage_pct']:.4f}%")
    
    print("\n" + "=" * 60)
    print("SLIPPAGE SIMULATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()