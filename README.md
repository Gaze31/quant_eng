# Algorithmic Trading Strategy Simulator

A Python-based backtesting framework that implements and compares four quantitative trading strategies on historical equity data. Built as part of a quantitative finance study to understand the real-world performance tradeoffs between technical, momentum, and machine learning approaches.

---

## Strategies Implemented

| Strategy | Approach | Key Indicators |
|---|---|---|
| Trend Following | MA Golden Cross (50/200) | SMA, EMA |
| Mean Reversion | Bollinger Bands + RSI | BB Width, RSI(14) |
| Momentum | Price Rate of Change | ROC, Volume Ratio |
| Machine Learning | Random Forest Classifier | 32 engineered features |

---

## Results — AAPL (2020-01-02 to 2023-12-29)

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|---|---|---|---|---|---|
| Trend Following | +8.32% | 0.09 | -42.28% | 51.0% | 804 |
| Mean Reversion | +2.69% | 0.21 | -5.11% | 32.4% | 71 |
| Momentum | -11.86% | -0.39 | -21.84% | 26.8% | 123 |
| Machine Learning | +959.09% | 3.96 | -36.36% | 55.6% | 804 |
| **Buy & Hold** | **+61.95%** | — | — | — | — |

---

## Key Findings

### 1. Trend Following — Overtrading Problem Identified and Fixed

Initial implementation used 20/50 MA crossover, generating 804 trades and producing **-37.11% return** due to transaction cost drag and whipsawing signals.

After switching to 50/200 (Golden Cross), the strategy stabilized:
- Return improved from **-37.11% → +8.32%**
- Sharpe improved from **-0.48 → +0.09**
- Max Drawdown improved from **-55.16% → -42.28%**

Still underperforms buy & hold — expected behavior for a single trending large-cap stock. Trend following strategies are better suited to diversified multi-asset portfolios.

### 2. Mean Reversion — Wrong Strategy for the Wrong Asset

Bollinger Bands + RSI mean reversion produced near-flat equity with only 71 trades over 4 years. The strategy is designed for range-bound assets. AAPL in 2020–2023 was in a sustained uptrend — mean reversion logic will always lose ground in that regime.

**Lesson:** Strategy selection must account for asset regime. Mean reversion on a trending stock is structurally broken regardless of parameter tuning.

### 3. Machine Learning — Classic Overfitting

Random Forest classifier showed:
- Train accuracy: **98.4%**
- Test accuracy: **62.9%**
- Reported return: **+959%**

The gap between train and test accuracy (35 points) is a clear signal of overfitting. The reported 959% return is not reliable — it is a product of the model memorizing training data, not learning generalizable patterns. This result would not survive out-of-sample or walk-forward validation.

Top features by importance: BB Width (14.3%), OBV (11.4%), MACD (11.3%), Price ROC (11.0%), ATR% (10.7%).

---

## Technical Indicators Implemented (from scratch)

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Average True Range (ATR)
- Stochastic Oscillator (%K, %D)
- On-Balance Volume (OBV)
- Volume Ratio

---

## Setup

```bash
git clone https://github.com/Gaze31/quant_eng.git
cd quant_eng
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python algo_trading.py
```

---

## Requirements

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## Known Limitations

- Single asset backtesting only — no portfolio-level risk management
- No walk-forward validation on ML strategy
- Transaction costs modeled as fixed 0.1% per trade — does not account for slippage
- No short selling implemented
- ML model requires out-of-sample validation before any real-world use

---

## Next Steps

- [ ] Walk-forward validation for ML strategy
- [ ] Multi-asset portfolio backtesting
- [ ] Regime detection (trending vs. mean-reverting) for dynamic strategy switching
- [ ] Add Sharpe-optimized position sizing
- [ ] Test on Indian equity markets (NSE data via yfinance)

---

## Author

**Sumedha Hundekar** — Finance graduate building quantitative trading systems in Python.  
Contact: velvetgazeze@gmail.com
