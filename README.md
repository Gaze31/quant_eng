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

## Credit Risk Neural Network

A feedforward neural network for predicting loan default probability, benchmarked against a Logistic Regression baseline. Trained on 32,574 real loan records with 14 engineered features.

---

### Architecture

```
Input (14 features)
    → Dense(64) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(32) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(16) + ReLU
    → Dense(1) + Sigmoid
Output: default probability (0 to 1)
```

---

### Dataset

- **Source:** Credit Risk Dataset (Kaggle)
- **Samples:** 32,574 (after cleaning)
- **Default rate:** 21.82% (class imbalance handled with weighted loss)
- **Features:** age, income, employment length, loan amount, interest rate, loan grade, home ownership, loan intent, debt-to-income ratio + 3 engineered features

---

### Results

| Model | AUC-ROC | Avg Precision | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8496 | 0.6848 | 79% |
| Neural Network | **0.8973** | **0.8307** | **90%** |
| **AUC Improvement** | **+5.61%** | **+21.3%** | — |

![Credit Risk Results](results/credit_risk_results.png)

---

### Key Findings

**1. Neural network meaningfully outperforms baseline**
AUC improved from 0.850 to 0.897 — a +5.61% gain. In credit risk, where model outputs directly affect lending decisions, this magnitude of improvement is operationally significant.

**2. Average Precision jump is the more important number**
Avg Precision improved from 0.685 to 0.831 — a +21.3% improvement. This matters more than AUC in imbalanced datasets because it measures precision across all recall thresholds, penalizing false positives heavily. A lender cares more about not approving bad loans than catching every good one.

**3. Default class precision: 0.93**
The model correctly identifies 93% of predicted defaults as actual defaults. Low false positive rate means fewer good borrowers wrongly rejected.

**4. Default class recall: 0.61**
The model misses 39% of actual defaults. This is the key tradeoff — the model is conservative, only flagging high-confidence defaults. Adjusting the classification threshold from 0.5 downward would improve recall at the cost of precision.

**5. Training converged cleanly**
No overfitting observed — validation loss decreased steadily across all 50 epochs, and validation AUC improved from 0.855 at epoch 1 to 0.908 at epoch 50.

---

### Setup

```bash
pip install torch scikit-learn pandas numpy matplotlib
python credit_risk_dl.py
```

Runs in approximately 5 minutes on CPU (MPS accelerated on Apple Silicon).

---

### Known Limitations

- Dataset is synthetic/public — not validated on real bank loan data
- No temporal validation — loans from different time periods may have different risk profiles
- Threshold of 0.5 is arbitrary — optimal threshold depends on the cost ratio of false positives vs false negatives in the specific lending context
- Model does not account for macroeconomic regime — trained in one market environment, may degrade in recession

---

Random Forest Strategy — Walk-Forward Validation
A proper out-of-sample validation of ML-based trading on AAPL (2018–2024). Built to address the overfitting problem identified in the original Random Forest strategy.
Methodology: 24-month rolling train window, 3-month test window, 18 folds. Zero data leakage. 46 features including RSI, MACD, Bollinger Bands, ATR, volume ratios, lagged returns, and calendar effects.
Results:
MetricValueMean Accuracy0.492 ± 0.062Mean AUC-ROC0.496 ± 0.064Total Return-6.60%Sharpe Ratio-4.549Win Rate42.06%

Key Finding: Walk-forward accuracy of 49.2% is indistinguishable from random. This directly contradicts the original in-sample ML result of 959% return — confirming it was overfitted. Technical indicators alone carry no predictive signal for next-day AAPL direction across 4 years of out-of-sample data.

Top feature by importance: overnight_gap — the gap between previous close and today's open. Despite being the most informative feature, it still cannot produce above-random accuracy out-of-sample.

Conclusion: Efficient market hypothesis holds for large-cap short-term prediction with standard technical features. Alpha requires either longer holding periods, alternative data sources, or fundamentally different signal construction.

# Stock Movement Classifier — XGBoost

Predicts next-day direction (up/down) for AAPL using walk-forward validation.

## What this project does differently
- No data leakage — uses walk-forward validation, not random train/test split
- Engineered features from returns and volume, not raw prices
- Honest evaluation — F1 score, not accuracy

## Results
| Fold | F1 | Precision | Recall |
|------|----|-----------|--------|
| 1 | 0.495 | 0.486 | 0.505 |
| 2 | 0.476 | 0.462 | 0.491 |
| 3 | 0.523 | 0.528 | 0.518 |
| 4 | 0.559 | 0.547 | 0.571 |
| 5 | 0.647 | 0.610 | 0.688 |
| **Mean** | **0.540** | **0.526** | **0.555** |

## Key finding
`volume_ratio` ranked #1 in both frequency and gain — volume precedes price movement,
consistent with institutional accumulation patterns before retail reaction.

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/stock_xgb
cd stock_xgb
pip install -r requirements.txt
python stock_xgb.py
```

## Features engineered
- Returns: 1d, 5d, 10d
- Volatility: 10d, 20d rolling std
- MA ratio: 5d/20d crossover signal
- RSI: 14-period
- Volume ratio: vs 20d average

## Model Comparison

| Model | Mean F1 | Notes |
|-------|---------|-------|
| XGBoost | 0.540 | Best on volatile regimes (Fold 1) |
| Random Forest | 0.541 | Consistent on trending markets |
| Bagging | 0.553 | Best mean F1, weakest on shocks |

**Finding:** No single model dominates. XGBoost shows stronger
resilience during market shocks (COVID crash period).
Bagging edges ahead on stable trending data.

| Stacking (XGB+RF+Bag → LR) | 0.604 | Best F1, but recall 
bias in bull markets — predicts UP aggressively in trending periods |
```

---

## Where to go from here

You've covered the three pillars of ensemble learning:
```
Boosting  ✅
Bagging   ✅
Stacking  ✅

# Optuna Quant Alpha Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-green.svg)](https://optuna.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-orange.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

A sophisticated quantitative finance alpha model that uses XGBoost with Optuna hyperparameter optimization to predict asset returns. The model implements walk-forward cross-validation with embargo periods to prevent look-ahead bias and generates synthetic market data with realistic factor structures.

## 📊 Key Features

- **Synthetic Market Data Generation**: Realistic equity return simulation with latent factor structure and idiosyncratic noise
- **Rich Feature Engineering**: 24 features including momentum, volatility, skewness, mean reversion, and cross-sectional signals
- **Walk-Forward Validation**: Time-series cross-validation with embargo periods to prevent look-ahead bias
- **Bayesian Hyperparameter Optimization**: Optuna with TPE sampler and Median pruning for efficient tuning
- **Comprehensive Performance Metrics**: Sharpe ratio, information coefficient, max drawdown, hit rate, and rank IC
- **Beautiful Visualizations**: 6-panel dashboard with optimization history, feature importance, and equity curves

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher

optuna-quant-alpha/
├── optuna_quant_project.py    # Main script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── optuna_results/            # Generated visualizations
    └── optuna_quant_results_*.png

============================================================
  OPTUNA QUANT ALPHA MODEL  —  Full Pipeline
============================================================

[1/5] Generating synthetic market data...
      1500 trading days | 50 assets
[2/5] Engineering features...
      Feature matrix: (1436, 24)  |  Target: (1436,)
[3/5] Running Optuna optimization...
[4/5] Optimization complete.

  ╔══════════════════════════════════════╗
  ║  BEST TRIAL RESULTS                  ║
  ╠══════════════════════════════════════╣
  ║  Trial #    : 0                       ║
  ║  Sharpe     : 0.7538                  ║
  ╠══════════════════════════════════════╣
  ║  BEST HYPERPARAMETERS                ║
  ╠══════════════════════════════════════╣
  ║  n_estimators      : 218               ║
  ║  max_depth         : 8                 ║
  ║  learning_rate     : 0.10013           ║
  ╚══════════════════════════════════════╝

[5/5] Final out-of-sample evaluation...

  OOS Sharpe     : 0.7450
  OOS Max DD     : -24.86%
  OOS Total Ret  : 15.14%
  OOS Hit Rate   : 54.2%
  OOS Rank IC    : 0.0000

# Feature Extraction Comparison for Handwritten Digit Recognition

This project demonstrates the impact of different feature extraction techniques on classification performance using the Digits dataset (8×8 images of handwritten digits). It compares:

- **Raw pixels** (flattened 64-dimensional vectors)
- **Histogram of Oriented Gradients (HOG)** – captures edge and shape information
- **Local Binary Patterns (LBP)** – captures local texture patterns

The extracted features are used to train a Support Vector Machine (SVM) classifier, and the accuracy of each approach is compared.

## 📁 Project Structure

This README is straightforward and suitable for a GitHub repository. You can adjust the repository name and your username accordingly.

# PCA vs t‑SNE: Dimensionality Reduction for Visualization

This project demonstrates two popular dimensionality reduction techniques – **PCA** (Principal Component Analysis) and **t‑SNE** (t‑Distributed Stochastic Neighbor Embedding) – applied to the Iris dataset. The results are visualized in 2D, highlighting the strengths of each method.

## 📊 Overview

- **PCA**: Linear, fast, preserves global variance.  
- **t‑SNE**: Non‑linear, excels at revealing local clusters (stochastic).  

The code:
1. Loads and standardizes the Iris dataset.
2. Reduces features to 2 dimensions using PCA and t‑SNE.
3. Plots both embeddings side‑by‑side with species‑colored points.

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit‑learn`

### Installation
```bash
git clone https://github.com/yourusername/pca-tsne-iris.git
cd pca-tsne-iris
pip install -r requirements.txt   # (optional: create a requirements.txt)
## Author

**Sumedha Hundekar** — Finance graduate building quantitative trading systems in Python.  
Contact: velvetgazeze@gmail.com
