import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SPREAD MODELS
# ============================================

class SpreadModel(Enum):
    """Different spread models for simulation"""
    CONSTANT = "Constant Spread"
    INVENTORY = "Inventory-Based Spread"
    VOLATILITY = "Volatility-Based Spread"
    ROLL = "Roll's Spread Estimator"
    HASBROUCK = "Hasbrouck's Model"
    MADHAVAN = "Madhavan's Model"
    REALISTIC = "Realistic Market Spread"

@dataclass
class SpreadComponents:
    """Components that make up the spread"""
    fixed_cost: float = 0.01      # Fixed exchange/liquidity fees
    inventory_cost: float = 0.0    # Cost of holding inventory
    adverse_selection: float = 0.0  # Cost of adverse selection
    order_processing: float = 0.005  # Order processing costs
    competition_factor: float = 1.0  # Competition multiplier

class BidAskSpreadSimulator:
    """
    Comprehensive bid-ask spread simulator with multiple models
    """
    
    def __init__(self, 
                 initial_price: float = 100.0,
                 tick_size: float = 0.01,
                 min_spread: float = 0.01,
                 max_spread: float = 1.0):
        
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.min_spread = min_spread
        self.max_spread = max_spread
        
        # Price process
        self.current_price = initial_price
        self.price_history = [initial_price]
        
        # Spread history
        self.spread_history = []
        self.bid_history = []
        self.ask_history = []
        
        # Market conditions
        self.volatility = 0.02  # 2% annualized volatility
        self.volume = 1000000    # Daily volume
        self.inventory = 0        # Market maker inventory
        self.num_market_makers = 5  # Competition level
        
        # Time
        self.current_time = 0
        self.timestamps = []
        
    # ============================================
    # SPREAD MODELS
    # ============================================
    
    def constant_spread(self, spread_value: float = 0.05) -> Tuple[float, float, float]:
        """Constant spread model"""
        bid = self.current_price - spread_value / 2
        ask = self.current_price + spread_value / 2
        return bid, ask, spread_value
    
    def inventory_spread(self, 
                         base_spread: float = 0.05,
                         inventory_scale: float = 0.1,
                         inventory_limit: float = 1000) -> Tuple[float, float, float]:
        """
        Inventory-based spread model
        Spread widens as inventory deviates from target
        """
        inventory_ratio = self.inventory / inventory_limit
        spread = base_spread * (1 + abs(inventory_ratio) * inventory_scale)
        
        # Asymmetric adjustment based on inventory sign
        if self.inventory > 0:
            # Too much inventory - lower asks, lower bids more
            bid = self.current_price - spread * (1 + inventory_ratio)
            ask = self.current_price + spread * (1 - inventory_ratio)
        else:
            # Need inventory - raise bids, raise asks less
            bid = self.current_price - spread * (1 - abs(inventory_ratio))
            ask = self.current_price + spread * (1 + abs(inventory_ratio))
        
        return bid, ask, spread
    
    def volatility_spread(self,
                          base_spread: float = 0.03,
                          vol_multiplier: float = 2.0,
                          lookback: int = 20) -> Tuple[float, float, float]:
        """
        Volatility-based spread model
        Spread increases with recent volatility
        """
        if len(self.price_history) > lookback:
            returns = np.diff(self.price_history[-lookback:]) / self.price_history[-lookback:-1]
            recent_vol = np.std(returns) * np.sqrt(252 * 390)  # Annualized
        else:
            recent_vol = self.volatility
        
        spread = base_spread + vol_multiplier * recent_vol
        spread = np.clip(spread, self.min_spread, self.max_spread)
        
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        return bid, ask, spread
    
    def roll_spread(self, 
                    lookback: int = 20,
                    min_spread: float = 0.01) -> Tuple[float, float, float]:
        """
        Roll's spread estimator (1984)
        Spread = 2 * sqrt(-covariance of price changes)
        """
        if len(self.price_history) > lookback + 1:
            price_changes = np.diff(self.price_history[-lookback:])
            cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            
            if cov < 0:
                spread = 2 * np.sqrt(-cov)
            else:
                spread = min_spread
        else:
            spread = min_spread
        
        spread = np.clip(spread, self.min_spread, self.max_spread)
        
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        return bid, ask, spread
    
    def hasbrouck_spread(self,
                         trade_indicator: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Hasbrouck's (2004) Gibbs sampling based spread estimator
        Simplified version based on trade direction
        """
        if trade_indicator is None:
            # Simulate trade direction (-1: sell, 0: none, 1: buy)
            trade_indicator = np.random.choice([-1, 0, 1], size=1, p=[0.3, 0.4, 0.3])[0]
        
        # Effective spread based on trade direction
        if trade_indicator > 0:
            # Buy trade - use ask
            spread = self.current_price * 0.001  # 10 bps
        elif trade_indicator < 0:
            # Sell trade - use bid
            spread = self.current_price * 0.001  # 10 bps
        else:
            # No trade - use quoted spread
            spread = self.current_price * 0.002  # 20 bps
        
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        return bid, ask, spread
    
    def madhavan_spread(self,
                        lambda_param: float = 0.5,
                        phi_param: float = 0.2) -> Tuple[float, float, float]:
        """
        Madhavan, Richardson, Roomans (1997) spread model
        Spread = lambda * sigma^2 + phi * inventory_cost
        """
        # Information asymmetry component
        info_component = lambda_param * self.volatility**2
        
        # Inventory component
        inventory_component = phi_param * abs(self.inventory) / 1000
        
        spread = info_component + inventory_component
        spread = max(spread, self.min_spread)
        
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        return bid, ask, spread
    
    def realistic_market_spread(self,
                                components: Optional[SpreadComponents] = None) -> Tuple[float, float, float]:
        """
        Realistic spread combining multiple factors
        """
        if components is None:
            components = SpreadComponents()
        
        # Update components based on market conditions
        components.inventory_cost = abs(self.inventory) * 0.0001
        components.adverse_selection = self.volatility * 0.5
        components.competition_factor = 1.0 / np.sqrt(self.num_market_makers)
        
        # Calculate total spread
        total_spread = (
            components.fixed_cost +
            components.inventory_cost +
            components.adverse_selection +
            components.order_processing
        ) * components.competition_factor
        
        total_spread = np.clip(total_spread, self.min_spread, self.max_spread)
        
        # Random noise
        total_spread *= (1 + np.random.normal(0, 0.1))
        
        bid = self.current_price - total_spread / 2
        ask = self.current_price + total_spread / 2
        
        return bid, ask, total_spread
    
    # ============================================
    # SIMULATION ENGINE
    # ============================================
    
    def update_market_conditions(self):
        """Update market conditions for next step"""
        # Random walk for price
        price_change = np.random.normal(0, self.volatility * self.current_price / np.sqrt(252 * 390))
        self.current_price += price_change
        self.current_price = max(self.current_price, 0.01)  # Prevent negative prices
        self.price_history.append(self.current_price)
        
        # Update volatility (stochastic volatility)
        self.volatility *= np.exp(np.random.normal(0, 0.01))
        self.volatility = np.clip(self.volatility, 0.005, 0.05)
        
        # Update inventory (random trades)
        trade_direction = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        trade_size = np.random.randint(0, 100)
        self.inventory += trade_direction * trade_size
        
        # Update competition
        self.num_market_makers += np.random.choice([-1, 0, 1], p=[0.05, 0.9, 0.05])
        self.num_market_makers = max(1, self.num_market_makers)
        
        self.current_time += 1
        self.timestamps.append(pd.Timestamp.now())
    
    def simulate_step(self, model: SpreadModel, **kwargs) -> Dict:
        """
        Simulate one step with specified model
        """
        # Update market conditions
        self.update_market_conditions()
        
        # Calculate spread based on model
        if model == SpreadModel.CONSTANT:
            bid, ask, spread = self.constant_spread(**kwargs)
        elif model == SpreadModel.INVENTORY:
            bid, ask, spread = self.inventory_spread(**kwargs)
        elif model == SpreadModel.VOLATILITY:
            bid, ask, spread = self.volatility_spread(**kwargs)
        elif model == SpreadModel.ROLL:
            bid, ask, spread = self.roll_spread(**kwargs)
        elif model == SpreadModel.HASBROUCK:
            bid, ask, spread = self.hasbrouck_spread(**kwargs)
        elif model == SpreadModel.MADHAVAN:
            bid, ask, spread = self.madhavan_spread(**kwargs)
        elif model == SpreadModel.REALISTIC:
            bid, ask, spread = self.realistic_market_spread(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Round to tick size
        bid = round(bid / self.tick_size) * self.tick_size
        ask = round(ask / self.tick_size) * self.tick_size
        spread = ask - bid
        
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        self.spread_history.append(spread)
        
        return {
            'time': self.current_time,
            'price': self.current_price,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'volatility': self.volatility,
            'inventory': self.inventory,
            'num_market_makers': self.num_market_makers
        }
    
    def simulate(self, 
                 model: SpreadModel,
                 n_steps: int = 1000,
                 **kwargs) -> pd.DataFrame:
        """
        Run full simulation
        """
        results = []
        
        for _ in range(n_steps):
            step_result = self.simulate_step(model, **kwargs)
            results.append(step_result)
        
        df = pd.DataFrame(results)
        
        # Add derived metrics
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000  # Spread in basis points
        df['relative_spread'] = df['spread'] / df['mid_price']
        
        return df
    
    def reset(self):
        """Reset simulator to initial state"""
        self.current_price = self.initial_price
        self.price_history = [self.initial_price]
        self.spread_history = []
        self.bid_history = []
        self.ask_history = []
        self.volatility = 0.02
        self.volume = 1000000
        self.inventory = 0
        self.num_market_makers = 5
        self.current_time = 0
        self.timestamps = []

# ============================================
# SPREAD VISUALIZER
# ============================================

class SpreadVisualizer:
    """Visualize bid-ask spread dynamics"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'bid': 'green',
            'ask': 'red',
            'mid': 'blue',
            'spread': 'purple'
        }
        
    def plot_spread_evolution(self, df: pd.DataFrame, title: str = "Spread Evolution"):
        """Plot spread over time"""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # Price and quotes
        ax1 = axes[0]
        ax1.plot(df.index, df['bid'], label='Bid', color=self.colors['bid'], alpha=0.7)
        ax1.plot(df.index, df['ask'], label='Ask', color=self.colors['ask'], alpha=0.7)
        ax1.plot(df.index, df['mid_price'], label='Mid', color=self.colors['mid'], linestyle='--')
        ax1.set_ylabel('Price')
        ax1.set_title(f'{title} - Price and Quotes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spread
        ax2 = axes[1]
        ax2.fill_between(df.index, 0, df['spread'], color=self.colors['spread'], alpha=0.5)
        ax2.set_ylabel('Spread ($)')
        ax2.set_title('Absolute Spread')
        ax2.grid(True, alpha=0.3)
        
        # Spread in basis points
        ax3 = axes[2]
        ax3.plot(df.index, df['spread_bps'], color=self.colors['spread'])
        ax3.set_ylabel('Spread (bps)')
        ax3.set_xlabel('Time Step')
        ax3.set_title('Spread in Basis Points')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_spread_distribution(self, dfs: Dict[str, pd.DataFrame]):
        """Compare spread distributions across models"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Histogram of spreads
        ax1 = axes[0, 0]
        for name, df in dfs.items():
            ax1.hist(df['spread_bps'], bins=50, alpha=0.5, label=name, density=True)
        ax1.set_xlabel('Spread (bps)')
        ax1.set_ylabel('Density')
        ax1.set_title('Spread Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[0, 1]
        box_data = [df['spread_bps'] for df in dfs.values()]
        ax2.boxplot(box_data, labels=dfs.keys())
        ax2.set_ylabel('Spread (bps)')
        ax2.set_title('Spread Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Time series comparison
        ax3 = axes[1, 0]
        for name, df in dfs.items():
            ax3.plot(df['spread_bps'].values[:500], label=name, alpha=0.7)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Spread (bps)')
        ax3.set_title('Spread Comparison (First 500 steps)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_data = []
        for name, df in dfs.items():
            stats_data.append([
                name,
                f"{df['spread'].mean():.4f}",
                f"{df['spread'].std():.4f}",
                f"{df['spread_bps'].mean():.2f}",
                f"{df['spread_bps'].std():.2f}"
            ])
        
        table = ax4.table(cellText=stats_data,
                         colLabels=['Model', 'Mean Spread', 'Std Spread', 'Mean bps', 'Std bps'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.show()
    
    def plot_market_impact(self, df: pd.DataFrame):
        """Plot relationship between spread and market conditions"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Spread vs Volatility
        ax1 = axes[0, 0]
        ax1.scatter(df['volatility'] * 100, df['spread_bps'], alpha=0.5, s=10)
        ax1.set_xlabel('Volatility (%)')
        ax1.set_ylabel('Spread (bps)')
        ax1.set_title('Spread vs Volatility')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['volatility'] * 100, df['spread_bps'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['volatility'].min() * 100, df['volatility'].max() * 100, 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
        ax1.legend()
        
        # Spread vs Inventory
        ax2 = axes[0, 1]
        ax2.scatter(df['inventory'], df['spread_bps'], alpha=0.5, s=10)
        ax2.set_xlabel('Inventory')
        ax2.set_ylabel('Spread (bps)')
        ax2.set_title('Spread vs Inventory')
        ax2.grid(True, alpha=0.3)
        
        # Spread vs Market Makers
        ax3 = axes[1, 0]
        unique_mms = sorted(df['num_market_makers'].unique())
        spread_by_mm = [df[df['num_market_makers'] == mm]['spread_bps'].mean() for mm in unique_mms]
        ax3.plot(unique_mms, spread_by_mm, 'bo-')
        ax3.set_xlabel('Number of Market Makers')
        ax3.set_ylabel('Avg Spread (bps)')
        ax3.set_title('Spread vs Competition')
        ax3.grid(True, alpha=0.3)
        
        # Autocorrelation of spreads
        ax4 = axes[1, 1]
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(df['spread_bps'].dropna(), ax=ax4)
        ax4.set_title('Spread Autocorrelation')
        ax4.set_xlim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def animate_spread(self, df: pd.DataFrame, interval: int = 50):
        """Create animated spread visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def animate(i):
            ax1.clear()
            ax2.clear()
            
            # Plot up to current frame
            frame_data = df.iloc[:i+1]
            
            # Price chart
            ax1.plot(frame_data.index, frame_data['bid'], 
                    color=self.colors['bid'], label='Bid', linewidth=1)
            ax1.plot(frame_data.index, frame_data['ask'], 
                    color=self.colors['ask'], label='Ask', linewidth=1)
            ax1.plot(frame_data.index, frame_data['mid_price'], 
                    color=self.colors['mid'], linestyle='--', label='Mid', linewidth=1)
            ax1.set_ylabel('Price')
            ax1.set_title(f'Bid-Ask Spread Animation (Step {i})')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Spread chart
            ax2.fill_between(frame_data.index, 0, frame_data['spread'], 
                            color=self.colors['spread'], alpha=0.5)
            ax2.set_ylabel('Spread ($)')
            ax2.set_xlabel('Time Step')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, animate, frames=len(df), interval=interval, repeat=False)
        plt.show()
        return anim

# ============================================
# SPREAD ANALYSIS TOOLS
# ============================================

class SpreadAnalyzer:
    """Analyze spread dynamics and properties"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_spread_components(self) -> Dict:
        """Decompose spread into components"""
        # Realized spread (using Roll's model)
        price_changes = np.diff(self.df['mid_price'])
        if len(price_changes) > 1:
            cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
            if cov < 0:
                roll_spread = 2 * np.sqrt(-cov)
            else:
                roll_spread = 0
        else:
            roll_spread = 0
        
        # Effective spread
        effective_spread = self.df['spread'].mean()
        
        # Adverse selection component (simplified)
        adverse_selection = effective_spread - roll_spread
        
        return {
            'effective_spread': effective_spread,
            'roll_implied_spread': roll_spread,
            'adverse_selection': max(0, adverse_selection),
            'order_processing': max(0, roll_spread),
            'spread_efficiency': (roll_spread / effective_spread) if effective_spread > 0 else 0
        }
    
    def calculate_liquidity_metrics(self) -> Dict:
        """Calculate various liquidity metrics"""
        metrics = {}
        
        # Amihud illiquidity measure
        returns = self.df['mid_price'].pct_change().dropna()
        volume = self.df['volume'] if 'volume' in self.df.columns else [1] * len(returns)
        amihud = (abs(returns) / volume).mean() if len(returns) > 0 else 0
        metrics['amihud_illiquidity'] = amihud
        
        # Quoted spread
        metrics['avg_quoted_spread'] = self.df['spread'].mean()
        metrics['median_quoted_spread'] = self.df['spread'].median()
        metrics['min_spread'] = self.df['spread'].min()
        metrics['max_spread'] = self.df['spread'].max()
        
        # Relative spread
        metrics['avg_relative_spread_bps'] = self.df['spread_bps'].mean()
        metrics['relative_spread_volatility'] = self.df['spread_bps'].std()
        
        # Spread percentiles
        metrics['spread_percentiles'] = {
            '1%': self.df['spread'].quantile(0.01),
            '5%': self.df['spread'].quantile(0.05),
            '25%': self.df['spread'].quantile(0.25),
            '75%': self.df['spread'].quantile(0.75),
            '95%': self.df['spread'].quantile(0.95),
            '99%': self.df['spread'].quantile(0.99)
        }
        
        return metrics
    
    def calculate_market_quality(self) -> Dict:
        """Calculate market quality metrics"""
        quality = {}
        
        # Bid-ask bounce
        returns = self.df['mid_price'].pct_change().dropna()
        quality['bid_ask_bounce'] = returns.autocorr()
        
        # Price impact (simplified)
        spread_changes = self.df['spread'].diff().dropna()
        price_changes = self.df['mid_price'].diff().dropna()
        
        if len(spread_changes) > 0 and len(price_changes) > 0:
            correlation = np.corrcoef(spread_changes, price_changes[:len(spread_changes)])[0, 1]
            quality['spread_price_correlation'] = correlation
        else:
            quality['spread_price_correlation'] = 0
        
        # Spread persistence
        quality['spread_half_life'] = self._calculate_half_life(self.df['spread'])
        
        return quality
    
    def _calculate_half_life(self, series: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        series = series.dropna()
        if len(series) < 2:
            return np.nan
        
        # Fit AR(1) model
        y = series.values[1:]
        x = series.values[:-1]
        
        x = x.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x, y)
        
        theta = model.coef_[0]
        
        if abs(theta) >= 1:
            return np.inf
        
        half_life = -np.log(2) / np.log(abs(theta))
        return half_life

# ============================================
# MAIN SIMULATION AND COMPARISON
# ============================================

def run_spread_comparison(n_steps: int = 1000):
    """Run comparison of all spread models"""
    
    print("=" * 60)
    print("BID-ASK SPREAD SIMULATION COMPARISON")
    print("=" * 60)
    
    # Initialize simulator
    simulator = BidAskSpreadSimulator(initial_price=100.0, tick_size=0.01)
    
    # Define models to test
    models = [
        SpreadModel.CONSTANT,
        SpreadModel.INVENTORY,
        SpreadModel.VOLATILITY,
        SpreadModel.ROLL,
        SpreadModel.HASBROUCK,
        SpreadModel.MADHAVAN,
        SpreadModel.REALISTIC
    ]
    
    # Store results
    results = {}
    
    # Run simulations
    for model in models:
        print(f"\nSimulating {model.value}...")
        simulator.reset()
        
        # Model-specific parameters
        if model == SpreadModel.CONSTANT:
            df = simulator.simulate(model, n_steps=n_steps, spread_value=0.05)
        elif model == SpreadModel.INVENTORY:
            df = simulator.simulate(model, n_steps=n_steps, base_spread=0.03)
        elif model == SpreadModel.VOLATILITY:
            df = simulator.simulate(model, n_steps=n_steps, base_spread=0.02)
        else:
            df = simulator.simulate(model, n_steps=n_steps)
        
        results[model.value] = df
        
        # Print basic stats
        print(f"  Avg Spread: ${df['spread'].mean():.4f}")
        print(f"  Avg Spread (bps): {df['spread_bps'].mean():.2f}")
        print(f"  Spread Std: ${df['spread'].std():.4f}")
    
    return results

# ============================================
# ADVANCED SPREAD MODELS
# ============================================

class DynamicSpreadModel:
    """Dynamic spread model with regime switching"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_probs = np.ones(n_regimes) / n_regimes
        self.transition_matrix = self._create_transition_matrix()
        
    def _create_transition_matrix(self) -> np.ndarray:
        """Create regime transition matrix"""
        # High persistence in regimes
        P = np.ones((self.n_regimes, self.n_regimes)) * 0.1
        np.fill_diagonal(P, 0.8)
        # Normalize rows
        P = P / P.sum(axis=1, keepdims=True)
        return P
    
    def next_regime(self, current_regime: int) -> int:
        """Sample next regime"""
        return np.random.choice(self.n_regimes, p=self.transition_matrix[current_regime])
    
    def get_spread_params(self, regime: int) -> Dict:
        """Get spread parameters for regime"""
        if regime == 0:  # Low spread regime
            return {
                'base_spread': 0.02,
                'vol_multiplier': 1.0,
                'inventory_scale': 0.05
            }
        elif regime == 1:  # Normal regime
            return {
                'base_spread': 0.05,
                'vol_multiplier': 2.0,
                'inventory_scale': 0.1
            }
        else:  # High spread regime
            return {
                'base_spread': 0.10,
                'vol_multiplier': 3.0,
                'inventory_scale': 0.2
            }

class IntradaySpreadPattern:
    """Model intraday spread patterns"""
    
    def __init__(self):
        # U-shaped pattern typical in markets
        self.pattern = self._create_u_shape()
        
    def _create_u_shape(self, n_minutes: int = 390) -> np.ndarray:
        """Create U-shaped intraday pattern"""
        x = np.linspace(0, 1, n_minutes)
        # U-shaped function: high at open/close, low in middle
        pattern = 1 + 2 * np.exp(-((x - 0.5) ** 2) / 0.05)
        pattern = pattern / pattern.mean()  # Normalize
        return pattern
    
    def get_spread_multiplier(self, minute_of_day: int) -> float:
        """Get spread multiplier for given minute"""
        return self.pattern[minute_of_day % len(self.pattern)]

# ============================================
# INTERACTIVE SPREAD TESTER
# ============================================

def interactive_spread_tester():
    """Interactive command-line spread tester"""
    
    simulator = BidAskSpreadSimulator()
    visualizer = SpreadVisualizer()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE BID-ASK SPREAD TESTER")
    print("=" * 60)
    print("\nAvailable Models:")
    for model in SpreadModel:
        print(f"  {model.value}")
    
    print("\nCommands:")
    print("  run <model> <steps> - Run simulation")
    print("  compare <steps>     - Compare all models")
    print("  analyze <model>     - Analyze specific model")
    print("  list               - List available models")
    print("  quit               - Exit")
    
    while True:
        cmd = input("\nEnter command: ").strip().split()
        
        if not cmd:
            continue
        
        if cmd[0].lower() == 'quit':
            break
        
        elif cmd[0].lower() == 'list':
            print("\nAvailable Models:")
            for model in SpreadModel:
                print(f"  {model.value}")
        
        elif cmd[0].lower() == 'run' and len(cmd) >= 2:
            model_name = ' '.join(cmd[1:-1]) if len(cmd) > 2 else cmd[1]
            steps = int(cmd[-1]) if len(cmd) > 2 else 500
            
            # Find model
            selected_model = None
            for model in SpreadModel:
                if model.value.lower() == model_name.lower():
                    selected_model = model
                    break
            
            if selected_model:
                print(f"\nRunning {selected_model.value} for {steps} steps...")
                simulator.reset()
                df = simulator.simulate(selected_model, n_steps=steps)
                visualizer.plot_spread_evolution(df, title=selected_model.value)
                
                # Show statistics
                analyzer = SpreadAnalyzer(df)
                metrics = analyzer.calculate_liquidity_metrics()
                print(f"\nStatistics for {selected_model.value}:")
                print(f"  Mean Spread: ${metrics['avg_quoted_spread']:.4f}")
                print(f"  Mean Spread (bps): {metrics['avg_relative_spread_bps']:.2f}")
                print(f"  Spread Range: ${metrics['min_spread']:.4f} - ${metrics['max_spread']:.4f}")
            else:
                print(f"Model '{model_name}' not found")
        
        elif cmd[0].lower() == 'compare' and len(cmd) >= 2:
            steps = int(cmd[1])
            results = run_spread_comparison(n_steps=steps)
            
            # Create DataFrames for comparison
            dfs = {name: df for name, df in results.items()}
            visualizer.plot_spread_distribution(dfs)
        
        elif cmd[0].lower() == 'analyze' and len(cmd) >= 2:
            model_name = ' '.join(cmd[1:])
            
            # Find model
            selected_model = None
            for model in SpreadModel:
                if model.value.lower() == model_name.lower():
                    selected_model = model
                    break
            
            if selected_model:
                print(f"\nAnalyzing {selected_model.value}...")
                simulator.reset()
                df = simulator.simulate(selected_model, n_steps=1000)
                
                analyzer = SpreadAnalyzer(df)
                
                # Get various analyses
                components = analyzer.calculate_spread_components()
                liquidity = analyzer.calculate_liquidity_metrics()
                quality = analyzer.calculate_market_quality()
                
                print("\nSpread Components:")
                for key, value in components.items():
                    print(f"  {key}: {value:.6f}")
                
                print("\nLiquidity Metrics:")
                for key, value in liquidity.items():
                    if key != 'spread_percentiles':
                        print(f"  {key}: {value:.6f}")
                
                print("\nSpread Percentiles:")
                for p, v in liquidity['spread_percentiles'].items():
                    print(f"  {p}: ${v:.4f}")
                
                print("\nMarket Quality:")
                for key, value in quality.items():
                    print(f"  {key}: {value:.6f}")
                
                # Plot market impact
                visualizer.plot_market_impact(df)
            else:
                print(f"Model '{model_name}' not found")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Run basic comparison
    print("Running basic spread comparison...")
    results = run_spread_comparison(n_steps=1000)
    
    # Create visualizer
    visualizer = SpreadVisualizer()
    
    # Plot first model as example
    first_model = list(results.keys())[0]
    print(f"\nPlotting {first_model} as example...")
    visualizer.plot_spread_evolution(results[first_model], title=first_model)
    
    # Compare all models
    print("\nComparing all spread models...")
    visualizer.plot_spread_distribution(results)
    
    # Analyze a specific model
    print("\nAnalyzing realistic market spread model...")
    simulator = BidAskSpreadSimulator()
    df = simulator.simulate(SpreadModel.REALISTIC, n_steps=1000)
    
    analyzer = SpreadAnalyzer(df)
    visualizer.plot_market_impact(df)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("FINAL SUMMARY STATISTICS")
    print("=" * 60)
    
    for name, df in results.items():
        print(f"\n{name}:")
        print(f"  Mean Spread: ${df['spread'].mean():.4f}")
        print(f"  Std Spread: ${df['spread'].std():.4f}")
        print(f"  Mean Spread (bps): {df['spread_bps'].mean():.2f}")
        print(f"  Spread Range: ${df['spread'].min():.4f} - ${df['spread'].max():.4f}")
    
    # Uncomment for interactive mode
    # interactive_spread_tester()