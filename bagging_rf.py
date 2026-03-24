import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def fetch_stock_data(ticker="AAPL", period="5y"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df


def engineer_features(df):
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)

    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_ratio'] = df['ma_5'] / df['ma_20']

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # The two new features from last time
    df['volume_price_corr'] = df['return_1d'].rolling(10).corr(df['volume_ratio'])
    df['vol_regime'] = df['volatility_10d'] / df['volatility_20d']

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def walk_forward_train(df, features, model, model_name, target='target', n_splits=5):
    split_size = len(df) // (n_splits + 1)
    results = []

    for i in range(1, n_splits + 1):
        train = df.iloc[:split_size * i]
        test  = df.iloc[split_size * i : split_size * (i + 1)]

        X_train, y_train = train[features], train[target]
        X_test,  y_test  = test[features],  test[target]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            'fold': i,
            'f1': f1_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds)
        })
        print(f"[{model_name}] Fold {i} | F1: {results[-1]['f1']:.3f} | "
              f"Precision: {results[-1]['precision']:.3f} | "
              f"Recall: {results[-1]['recall']:.3f}")

    return pd.DataFrame(results)


def plot_comparison(rf_results, bag_results):
    folds = rf_results['fold']
    x = range(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], rf_results['f1'],
           width, label='Random Forest', color='#2ecc71', alpha=0.85)
    ax.bar([i + width/2 for i in x], bag_results['f1'],
           width, label='Bagging (Decision Tree)', color='#3498db', alpha=0.85)

    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score')
    ax.set_title('Random Forest vs Bagging — Walk-Forward F1')
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'Fold {i+1}' for i in x])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.54, color='red', linestyle='--',
               linewidth=1, label='XGBoost baseline (0.54)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("\nSaved comparison.png")


if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_stock_data(ticker="AAPL", period="5y")
    df = engineer_features(df)

    features = [
        'return_1d', 'return_5d', 'return_10d',
        'volatility_10d', 'volatility_20d',
        'ma_ratio', 'rsi', 'volume_ratio',
        'volume_price_corr', 'vol_regime'
    ]

    # Model 1 — Random Forest (bagging of decision trees with feature sampling)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        max_features='sqrt',      # random feature subset per split — key bagging idea
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1                 # parallel — all cores
    )

    # Model 2 — Pure Bagging (no feature sampling, just row sampling)
    bag_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=6),
        n_estimators=200,
        max_samples=0.8,          # 80% of rows per tree
        max_features=1.0,         # all features — pure row bagging
        random_state=42,
        n_jobs=-1
    )

    print("\n--- Random Forest ---")
    rf_results = walk_forward_train(df, features, rf_model, "RandomForest")

    print("\n--- Bagging Classifier ---")
    bag_results = walk_forward_train(df, features, bag_model, "Bagging")

    # Summary
    print("\n========== SUMMARY ==========")
    print(f"Random Forest  — Mean F1: {rf_results['f1'].mean():.3f}")
    print(f"Bagging        — Mean F1: {bag_results['f1'].mean():.3f}")
    print(f"XGBoost        — Mean F1: 0.540  (your baseline)")
    print("==============================")

    plot_comparison(rf_results, bag_results)