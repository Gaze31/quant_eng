import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier


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
    df['volume_price_corr'] = df['return_1d'].rolling(10).corr(df['volume_ratio'])
    df['vol_regime'] = df['volatility_10d'] / df['volatility_20d']

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def walk_forward_stacking(df, features, target='target', n_splits=5):
    split_size = len(df) // (n_splits + 1)
    results = []

    # Base models — your three from before
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            eval_metric='logloss', random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=6,
            max_features='sqrt', min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )),
        ('bag', BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6),
            n_estimators=200, max_samples=0.8,
            random_state=42, n_jobs=-1
        ))
    ]

    # Meta-learner — learns WHEN to trust which base model
    # Logistic Regression is ideal here — simple, interpretable,
    # won't overfit on the small meta-feature set (just 3 predictions)
    meta_learner = LogisticRegression(max_iter=1000)

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,                    # cross-val inside training to prevent leakage
        stack_method='predict_proba',  # pass probabilities not just 0/1
        n_jobs=-1
    )

    for i in range(1, n_splits + 1):
        train = df.iloc[:split_size * i]
        test  = df.iloc[split_size * i : split_size * (i + 1)]

        X_train, y_train = train[features], train[target]
        X_test,  y_test  = test[features],  test[target]

        print(f"Training fold {i} — {len(X_train)} rows train, {len(X_test)} rows test...")
        stacking_model.fit(X_train, y_train)
        preds = stacking_model.predict(X_test)

        results.append({
            'fold': i,
            'f1': f1_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds)
        })
        print(f"Fold {i} | F1: {results[-1]['f1']:.3f} | "
              f"Precision: {results[-1]['precision']:.3f} | "
              f"Recall: {results[-1]['recall']:.3f}")

    return pd.DataFrame(results)


def plot_all(stacking_results):
    # Previous results hardcoded for comparison
    xgb_f1  = [0.495, 0.476, 0.523, 0.554, 0.647]
    rf_f1   = [0.404, 0.553, 0.553, 0.564, 0.628]
    bag_f1  = [0.418, 0.568, 0.576, 0.574, 0.631]
    stack_f1 = stacking_results['f1'].tolist()

    folds = [f'Fold {i}' for i in range(1, 6)]
    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, xgb_f1,   width, label='XGBoost',       color='#3498db', alpha=0.85)
    ax.bar(x - 0.5*width, rf_f1,    width, label='Random Forest',  color='#2ecc71', alpha=0.85)
    ax.bar(x + 0.5*width, bag_f1,   width, label='Bagging',        color='#e67e22', alpha=0.85)
    ax.bar(x + 1.5*width, stack_f1, width, label='Stacking',       color='#9b59b6', alpha=0.85)

    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score')
    ax.set_title('All Models — Walk-Forward F1 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('stacking_comparison.png', dpi=150)
    print("\nSaved stacking_comparison.png")


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

    print("\n--- Stacking Classifier ---")
    stacking_results = walk_forward_stacking(df, features)

    print("\n========== FINAL SUMMARY ==========")
    print(f"XGBoost        — Mean F1: 0.540")
    print(f"Random Forest  — Mean F1: 0.541")
    print(f"Bagging        — Mean F1: 0.553")
    print(f"Stacking       — Mean F1: {stacking_results['f1'].mean():.3f}")
    print("====================================")

    plot_all(stacking_results)