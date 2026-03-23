# ============================================================
# rf_finance.py — Complete Random Forest Finance Pipeline
# Single file. No setup. Just run it.
#
# INSTALL:  pip install yfinance scikit-learn pandas numpy matplotlib python-dateutil
# RUN:      python rf_finance.py
# ============================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Save charts to file instead of popup
from pathlib import Path
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS — Change these to experiment
# ============================================================

TICKER           = "AAPL"       # Stock to analyze
TRAIN_START      = "2018-01-01"
TRAIN_END        = "2024-01-01"
PROB_THRESHOLD   = 0.55         # Only trade when model is this confident
TRANSACTION_COST = 0.001        # 0.1% per trade
POSITION_SIZE    = 0.10         # 10% of portfolio per trade
STOP_LOSS        = -0.05        # -5% stop loss
INITIAL_CAPITAL  = 100_000      # Starting portfolio value

RF_PARAMS = {
    "n_estimators"   : 300,
    "max_depth"      : 6,
    "min_samples_leaf": 50,
    "max_features"   : "sqrt",
    "class_weight"   : "balanced",
    "n_jobs"         : -1,
    "random_state"   : 42,
}

# ============================================================
# STEP 1: DOWNLOAD DATA
# ============================================================

def download_data(ticker, start, end):
    print(f"\n{'='*55}")
    print(f"  RF FINANCE PIPELINE | {ticker}")
    print(f"{'='*55}")
    print(f"\n[1/5] Downloading {ticker} data...")

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        print(f"    Downloaded {len(df)} trading days ({df.index.min().date()} → {df.index.max().date()})")
        return df
    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        print("    Make sure yfinance is installed: pip install yfinance")
        sys.exit(1)

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================

def build_features(df):
    print(f"\n[2/5] Engineering features...")
    df = df.copy()

    # Returns
    df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["return_1d"]    = df["Close"].pct_change(1)
    df["return_5d"]    = df["Close"].pct_change(5)
    df["return_21d"]   = df["Close"].pct_change(21)
    df["overnight_gap"]= (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Lagged returns (gives RF time context)
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"lag_{lag}d"] = df["log_return"].shift(lag)

    # Rolling stats
    for w in [5, 10, 21, 63]:
        df[f"roll_mean_{w}"] = df["log_return"].rolling(w).mean()
        df[f"roll_std_{w}"]  = df["log_return"].rolling(w).std()
        df[f"roll_skew_{w}"] = df["log_return"].rolling(w).skew()

    # Momentum
    for w in [5, 10, 21, 63]:
        df[f"dist_high_{w}"] = (df["Close"] - df["High"].rolling(w).max()) / df["High"].rolling(w).max()
        df[f"dist_low_{w}"]  = (df["Close"] - df["Low"].rolling(w).min())  / df["Low"].rolling(w).min()

    # RSI
    for period in [7, 14, 21]:
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        df[f"rsi_{period}"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # Bollinger Bands
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_pos"]   = (df["Close"] - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std) + 1e-10)
    df["bb_width"] = (4 * std) / sma

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_norm"] = tr.rolling(14).mean() / df["Close"]

    # Volume
    for w in [5, 10, 21]:
        df[f"vol_ratio_{w}"] = df["Volume"] / (df["Volume"].rolling(w).mean() + 1e-10)

    # Calendar
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)

    # TARGET: will price go UP tomorrow?
    df["future_return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["target"]        = (df["future_return"] > 0).astype(int)

    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns
                    if c not in {"Open","High","Low","Close","Volume","target","future_return"}]

    print(f"    {len(feature_cols)} features created | {len(df)} rows remaining")
    return df, feature_cols

# ============================================================
# STEP 3: WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validate(df, feature_cols):
    print(f"\n[3/5] Walk-Forward Validation (no data leakage)...")
    print(f"      Train window: 24 months | Test window: 3 months\n")

    results   = []
    min_date  = df.index.min()
    max_date  = df.index.max()
    train_start = min_date
    train_end   = min_date + relativedelta(months=24)
    test_start  = train_end
    test_end    = test_start + relativedelta(months=3)
    fold = 1

    while test_end <= max_date:
        tr_mask = (df.index >= train_start) & (df.index < train_end)
        te_mask = (df.index >= test_start)  & (df.index < test_end)

        X_tr = df.loc[tr_mask, feature_cols].values
        y_tr = df.loc[tr_mask, "target"].values
        X_te = df.loc[te_mask, feature_cols].values
        y_te = df.loc[te_mask, "target"].values

        if len(X_tr) < 100 or len(X_te) < 10:
            train_end = test_end
            test_start = train_end
            test_end = test_start + relativedelta(months=3)
            continue

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        print(f"    Fold {fold:02d} | {test_start.date()} → {test_end.date()} | Acc: {acc:.3f} | AUC: {auc:.3f}")

        results.append({
            "fold"       : fold,
            "test_start" : test_start.date(),
            "test_end"   : test_end.date(),
            "accuracy"   : acc,
            "precision"  : precision_score(y_te, y_pred, zero_division=0),
            "recall"     : recall_score(y_te, y_pred, zero_division=0),
            "f1"         : f1_score(y_te, y_pred, zero_division=0),
            "roc_auc"    : auc,
            "test_index" : df.loc[te_mask].index.tolist(),
            "y_true"     : y_te.tolist(),
            "y_pred"     : y_pred.tolist(),
            "y_prob"     : y_prob.tolist(),
            "feat_imp"   : dict(zip(feature_cols, clf.feature_importances_)),
        })

        train_end  = test_end
        test_start = train_end
        test_end   = test_start + relativedelta(months=3)
        fold += 1

    accs = [r["accuracy"] for r in results]
    aucs = [r["roc_auc"]  for r in results]
    print(f"\n    Mean Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"    Mean AUC      : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return results

# ============================================================
# STEP 4: BACKTEST
# ============================================================

def backtest(results, df):
    print(f"\n[4/5] Backtesting (with costs, stop loss, position sizing)...")

    trades  = []
    capital = INITIAL_CAPITAL
    peak    = INITIAL_CAPITAL
    equity  = []
    halted  = False

    for fold in results:
        for date, prob, actual_dir in zip(fold["test_index"], fold["y_prob"], fold["y_true"]):
            if prob > PROB_THRESHOLD:
                signal = 1
            elif prob < (1 - PROB_THRESHOLD):
                signal = -1
            else:
                continue

            if date not in df.index:
                continue

            actual_ret  = df.loc[date, "future_return"]
            if pd.isna(actual_ret):
                continue

            trade_ret   = signal * actual_ret
            trade_ret   = max(trade_ret, STOP_LOSS)
            trade_ret  -= 2 * TRANSACTION_COST
            pnl         = trade_ret * POSITION_SIZE

            if not halted:
                capital += capital * pnl
                peak     = max(peak, capital)
                drawdown = (capital - peak) / peak
                if drawdown < -0.20:
                    halted = True
                    print(f"    ⚠ Trading HALTED {date} — max drawdown breached")

            trades.append({
                "date": date, "signal": signal, "prob": prob,
                "trade_return": trade_ret, "pnl": pnl, "capital": capital,
                "drawdown": (capital - peak) / peak
            })

    trades_df = pd.DataFrame(trades).set_index("date")

    # Metrics
    total_ret  = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    n          = len(trades_df)
    ann_ret    = (1 + total_ret) ** (252 / max(n, 1)) - 1
    rets       = trades_df["pnl"]
    sharpe     = (rets.mean() - 0.05/252) / (rets.std() + 1e-10) * np.sqrt(252)
    downside   = rets[rets < 0]
    sortino    = (rets.mean() - 0.05/252) / (downside.std() + 1e-10) * np.sqrt(252)
    max_dd     = trades_df["drawdown"].min()
    win_rate   = (trades_df["trade_return"] > 0).mean()

    print(f"\n    ── RESULTS ────────────────────────────")
    print(f"    Total Return      : {total_ret:.2%}")
    print(f"    Annualized Return : {ann_ret:.2%}")
    print(f"    Sharpe Ratio      : {sharpe:.3f}")
    print(f"    Sortino Ratio     : {sortino:.3f}")
    print(f"    Max Drawdown      : {max_dd:.2%}")
    print(f"    Win Rate          : {win_rate:.2%}")
    print(f"    Total Trades      : {n}")
    print(f"    ───────────────────────────────────────")

    return trades_df

# ============================================================
# STEP 5: CHARTS
# ============================================================

def save_charts(trades_df, results, df, ticker):
    print(f"\n[5/5] Saving charts...")
    Path("results").mkdir(exist_ok=True)

    dark = "#0d1117"
    plt.rcParams.update({
        "axes.facecolor": "#161b22", "figure.facecolor": dark,
        "text.color": "#e6edf3", "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d", "grid.color": "#21262d",
        "font.family": "monospace",
    })

    # Chart 1: Equity curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"RF Strategy — {ticker}", fontsize=14, color="#e6edf3")
    ax1.plot(trades_df.index, trades_df["capital"], color="#58a6ff", linewidth=1.5)
    ax1.axhline(INITIAL_CAPITAL, color="#8b949e", linestyle="--", linewidth=0.8)
    ax1.fill_between(trades_df.index, trades_df["capital"], INITIAL_CAPITAL, alpha=0.1, color="#58a6ff")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(trades_df.index, trades_df["drawdown"]*100, 0, color="#f85149", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{ticker}_equity.png", dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close()
    print(f"    Saved: results/{ticker}_equity.png")

    # Chart 2: WFV accuracy per fold
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Walk-Forward Accuracy — {ticker}", fontsize=13, color="#e6edf3")
    folds = [str(r["test_start"]) for r in results]
    accs  = [r["accuracy"] for r in results]
    aucs  = [r["roc_auc"]  for r in results]
    x     = range(len(folds))
    ax.bar([i - 0.2 for i in x], accs, width=0.4, color="#58a6ff", alpha=0.8, label="Accuracy")
    ax.bar([i + 0.2 for i in x], aucs, width=0.4, color="#3fb950", alpha=0.8, label="AUC-ROC")
    ax.axhline(0.5, color="#f85149", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(folds, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"results/{ticker}_wfv.png", dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close()
    print(f"    Saved: results/{ticker}_wfv.png")

    # Chart 3: Feature importance
    all_imp = {}
    for r in results:
        for feat, imp in r["feat_imp"].items():
            all_imp[feat] = all_imp.get(feat, 0) + imp
    top = sorted(all_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    feats, imps = zip(*reversed(top))
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"Top 20 Features — {ticker}", fontsize=13, color="#e6edf3")
    ax.barh(feats, imps, color="#58a6ff", alpha=0.85)
    ax.set_xlabel("Avg Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"results/{ticker}_features.png", dpi=150, bbox_inches="tight", facecolor=dark)
    plt.close()
    print(f"    Saved: results/{ticker}_features.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER

    df_raw              = download_data(ticker, TRAIN_START, "2025-01-01")
    df_feat, feat_cols  = build_features(df_raw)
    results             = walk_forward_validate(df_feat, feat_cols)
    trades_df           = backtest(results, df_feat)
    save_charts(trades_df, results, df_feat, ticker)

    print(f"\n{'='*55}")
    print(f"  DONE. Charts saved in results/ folder.")
    print(f"  Paper trade for 6 months before real money.")
    print(f"{'='*55}\n")