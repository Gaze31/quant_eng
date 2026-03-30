import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)

# Optional imports – will be skipped if not installed
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Will skip XGBoost model.")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("imbalanced-learn not installed. Will proceed without SMOTE (class imbalance will hurt performance).")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Will skip SHAP plots.")

# -------------------------------------------------------------------
# 1. Load or generate dataset
# -------------------------------------------------------------------
def load_data():
    if os.path.exists('creditcard.csv'):
        df = pd.read_csv('creditcard.csv')
        print("Loaded real creditcard.csv")
    else:
        print("creditcard.csv not found. Generating synthetic data for demonstration.")
        np.random.seed(42)
        n_samples = 100000
        n_features = 29
        X_synth = np.random.randn(n_samples, n_features)
        y_synth = np.zeros(n_samples, dtype=int)
        fraud_idx = np.random.choice(n_samples, size=int(0.0017 * n_samples), replace=False)
        y_synth[fraud_idx] = 1
        X_synth[fraud_idx, :5] += 3.0
        df = pd.DataFrame(X_synth, columns=[f'V{i}' for i in range(1, n_features+1)])
        df['Class'] = y_synth
        df['Amount'] = np.random.exponential(100, n_samples)
        df['Time'] = np.random.randint(0, 172800, n_samples)
    return df

df = load_data()
print(f"Dataset shape: {df.shape}")
print(f"Fraud ratio: {df['Class'].mean():.4f}")

# -------------------------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------------------------
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)} (fraud: {y_train.mean():.4f})")
print(f"Test size:  {len(X_test)}  (fraud: {y_test.mean():.4f})")

# -------------------------------------------------------------------
# 3. Handle imbalance (optional SMOTE)
# -------------------------------------------------------------------
if SMOTE_AVAILABLE:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"\nAfter SMOTE: Train size {len(X_train_res)} (fraud: {y_train_res.mean():.4f})")
else:
    X_train_res, y_train_res = X_train, y_train
    print("\nSMOTE not used – training on original imbalanced data.")

# -------------------------------------------------------------------
# 4. Model training and evaluation
# -------------------------------------------------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Avg Precision: {average_precision_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return y_proba

models = {}

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
y_proba_lr = train_and_evaluate(lr, X_train_res, y_train_res, X_test, y_test, "Logistic Regression")
models['Logistic Regression'] = lr

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
y_proba_rf = train_and_evaluate(rf, X_train_res, y_train_res, X_test, y_test, "Random Forest")
models['Random Forest'] = rf

# XGBoost (if available)
if XGB_AVAILABLE:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    y_proba_xgb = train_and_evaluate(xgb, X_train_res, y_train_res, X_test, y_test, "XGBoost")
    models['XGBoost'] = xgb
else:
    print("\nSkipping XGBoost – package not installed.")

# -------------------------------------------------------------------
# 5. ROC and Precision-Recall curves
# -------------------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.subplot(1,2,2)
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 6. Cost analysis
# -------------------------------------------------------------------
def calculate_cost(y_true, y_pred, cost_fp=10, cost_fn=100):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn

print("\nCost Analysis (cost_fp=10, cost_fn=100):")
for name, model in models.items():
    y_pred = model.predict(X_test)
    cost = calculate_cost(y_test, y_pred)
    print(f"{name}: Total Cost = {cost}")

# -------------------------------------------------------------------
# 7. SHAP explainability (optional)
# -------------------------------------------------------------------
if SHAP_AVAILABLE and XGB_AVAILABLE and 'XGBoost' in models:
    print("\nGenerating SHAP explanations for XGBoost...")
    X_sample = X_test.sample(n=100, random_state=42)
    explainer = shap.TreeExplainer(models['XGBoost'])
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns.tolist())
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_sample.iloc[0,:],
                    matplotlib=True, feature_names=X.columns.tolist())
else:
    print("\nSHAP explanations skipped (SHAP or XGBoost missing).")

print("\nDone.")