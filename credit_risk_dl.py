"""
Credit Risk Neural Network
==========================
Predicts probability of loan default using a feedforward neural network.
Compares against Logistic Regression baseline.

Dataset: credit_risk_dataset.csv (32,581 samples)
Target:  loan_status (0 = no default, 1 = default)

Run:
    pip install torch scikit-learn pandas numpy matplotlib
    python credit_risk_dl.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_and_preprocess(path="credit_risk_dataset.csv"):
    print("\nLoading data...")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['loan_status'].mean():.2%}")

    # --- Handle missing values ---
    # loan_int_rate: fill with median per loan_grade (rates vary by grade)
    df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
        lambda x: x.fillna(x.median())
    )
    # person_emp_length: fill with median
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    # --- Remove outliers ---
    # Age > 100 is data error
    df = df[df["person_age"] <= 100]
    # Employment length > 60 years is unrealistic
    df = df[df["person_emp_length"] <= 60]

    # --- Encode categoricals ---
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade",
                "cb_person_default_on_file"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # --- Feature engineering ---
    df["debt_to_income"] = df["loan_amnt"] / (df["person_income"] + 1)
    df["income_per_year_employed"] = df["person_income"] / (df["person_emp_length"] + 1)
    df["loan_to_income_x_grade"] = df["loan_percent_income"] * df["loan_grade"]

    feature_cols = [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
        "cb_person_cred_hist_length", "debt_to_income",
        "income_per_year_employed", "loan_to_income_x_grade"
    ]

    X = df[feature_cols].values
    y = df["loan_status"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Samples after cleaning: {len(df)}")

    # --- Train/val/test split: 70/15/15 ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols


# =============================================================================
# 2. PYTORCH DATASET
# =============================================================================

class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 3. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class CreditRiskNet(nn.Module):
    """
    Feedforward neural network for binary credit risk classification.
    Architecture: 14 -> 64 -> 32 -> 16 -> 1
    Uses BatchNorm for training stability and Dropout for regularization.
    """
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(32, 16),
            nn.ReLU(),
            # Output
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(X_train, X_val, y_train, y_val, input_dim, epochs=50):
    print("\n" + "="*60)
    print("  TRAINING: Credit Risk Neural Network")
    print("="*60)

    train_ds = CreditDataset(X_train, y_train)
    val_ds = CreditDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = CreditRiskNet(input_dim).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights to handle imbalance (78% no default, 22% default)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses, val_aucs = [], [], []
    best_val_auc = 0
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Validate
        model.eval()
        v_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                v_loss += criterion(preds, y_batch).item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_preds)
        scheduler.step(v_loss)

        train_losses.append(t_loss / len(train_loader))
        val_losses.append(v_loss / len(val_loader))
        val_aucs.append(val_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_credit_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest Val AUC: {best_val_auc:.4f}")
    model.load_state_dict(torch.load("best_credit_model.pt", map_location=DEVICE))
    return model, train_losses, val_losses, val_aucs


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, name="Neural Network"):
    model.eval()
    test_ds = CreditDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            probs = model(X_batch)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS: {name}")
    print(f"{'='*60}")
    print(f"AUC-ROC:           {auc:.4f}")
    print(f"Avg Precision:     {ap:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["No Default", "Default"]))

    return all_probs, all_labels, auc, ap


# =============================================================================
# 6. BASELINE: LOGISTIC REGRESSION
# =============================================================================

def run_baseline(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  BASELINE: Logistic Regression")
    print("="*60)

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    cw = {0: weights[0], 1: weights[1]}

    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight=cw, random_state=SEED)
    lr.fit(X_train, y_train)

    probs = lr.predict_proba(X_test)[:, 1]
    preds = lr.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    print(f"AUC-ROC:           {auc:.4f}")
    print(f"Avg Precision:     {ap:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds,
                                target_names=["No Default", "Default"]))

    return probs, auc, ap


# =============================================================================
# 7. PLOTTING
# =============================================================================

def plot_results(train_losses, val_losses, val_aucs,
                 nn_probs, lr_probs, y_test,
                 nn_auc, lr_auc):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Credit Risk Neural Network — Results", fontsize=14)

    # Training curves
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses, label="Val Loss", color="orange")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Validation AUC over epochs
    ax2 = axes[0, 1]
    ax2.plot(val_aucs, label="Val AUC", color="green")
    ax2.axhline(y=lr_auc, color="red", linestyle="--",
                label=f"LogReg AUC ({lr_auc:.3f})")
    ax2.set_title("Validation AUC vs Logistic Regression Baseline")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ROC Curves
    ax3 = axes[1, 0]
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    ax3.plot(nn_fpr, nn_tpr, label=f"Neural Network (AUC={nn_auc:.3f})", color="blue")
    ax3.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC={lr_auc:.3f})",
             color="red", linestyle="--")
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax3.set_title("ROC Curve Comparison")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Default probability distribution
    ax4 = axes[1, 1]
    ax4.hist(nn_probs[y_test == 0], bins=50, alpha=0.6,
             label="No Default", color="green", density=True)
    ax4.hist(nn_probs[y_test == 1], bins=50, alpha=0.6,
             label="Default", color="red", density=True)
    ax4.set_title("Predicted Default Probability Distribution")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("credit_risk_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: credit_risk_results.png")


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CREDIT RISK NEURAL NETWORK")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = \
        load_and_preprocess("credit_risk_dataset.csv")

    # Baseline
    lr_probs, lr_auc, lr_ap = run_baseline(X_train, X_test, y_train, y_test)

    # Train neural network
    model, train_losses, val_losses, val_aucs = train_model(
        X_train, X_val, y_train, y_val,
        input_dim=X_train.shape[1], epochs=50
    )

    # Evaluate
    nn_probs, nn_labels, nn_auc, nn_ap = evaluate_model(model, X_test, y_test)

    # Summary
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'AUC-ROC':<12} {'Avg Precision':<15}")
    print("-" * 52)
    print(f"{'Logistic Regression':<25} {lr_auc:<12.4f} {lr_ap:<15.4f}")
    print(f"{'Neural Network':<25} {nn_auc:<12.4f} {nn_ap:<15.4f}")
    improvement = ((nn_auc - lr_auc) / lr_auc) * 100
    print(f"\nAUC improvement: {improvement:+.2f}%")

    # Plot
    plot_results(train_losses, val_losses, val_aucs,
                 nn_probs, lr_probs, np.array(nn_labels),
                 nn_auc, lr_auc)


if __name__ == "__main__":
    main()