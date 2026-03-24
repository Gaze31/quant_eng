import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def walk_forward_train(df, features, target='target', n_splits=5):
    split_size = len(df) // (n_splits + 1)
    results = []
    model = None  # keep last model for feature importance

    for i in range(1, n_splits + 1):
        train = df.iloc[:split_size * i]
        test  = df.iloc[split_size * i : split_size * (i + 1)]

        X_train, y_train = train[features], train[target]
        X_test,  y_test  = test[features],  test[target]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        preds = model.predict(X_test)
        results.append({
            'fold': i,
            'f1': f1_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds)
        })
        print(f"Fold {i} | F1: {results[-1]['f1']:.3f} | "
              f"Precision: {results[-1]['precision']:.3f} | "
              f"Recall: {results[-1]['recall']:.3f}")

    return results, model


def plot_importance(model):
    xgb.plot_importance(model, max_num_features=15)
    plt.tight_layout()
    plt.show()


# ── MAIN ─────────────────────────────────────────────────────────────────────
# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Fetch data
    print("Fetching data...")
    df = fetch_stock_data(ticker="AAPL", period="5y")

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Define features
    features = [
        'return_1d', 'return_5d', 'return_10d',
        'volatility_10d', 'volatility_20d',
        'ma_ratio', 'rsi', 'volume_ratio'
    ]

    # 4. Train + evaluate
    print("\nWalk-Forward Results:")
    results, model = walk_forward_train(df, features)

    # 5. Summary
    results_df = pd.DataFrame(results)
    print(f"\nMean F1:        {results_df['f1'].mean():.3f}")
    print(f"Mean Precision: {results_df['precision'].mean():.3f}")
    print(f"Mean Recall:    {results_df['recall'].mean():.3f}")

    # 6. Feature importance — 3 types, all different perspectives
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Weight: how many times a feature is used in splits
    xgb.plot_importance(model, ax=axes[0], importance_type='weight',
                        max_num_features=8, title='Importance: Weight')

    # Gain: average improvement in loss when feature is used
    xgb.plot_importance(model, ax=axes[1], importance_type='gain',
                        max_num_features=8, title='Importance: Gain')

    # Cover: how many samples are affected by splits on this feature
    xgb.plot_importance(model, ax=axes[2], importance_type='cover',
                        max_num_features=8, title='Importance: Cover')

    plt.tight_layout()
    plt.savefig("importance.png", dpi=150)
    print("\nSaved importance.png — open it and tell me what you see.")