import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# ============================================================================
# 1. LINEAR SVM FOR BINARY CLASSIFICATION
# ============================================================================
print("=" * 70)
print("1. LINEAR SVM - BINARY CLASSIFICATION")
print("=" * 70)

# Load breast cancer dataset
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear SVM
linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
linear_svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = linear_svm.predict(X_test_scaled)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================================
# 2. NON-LINEAR SVM WITH RBF KERNEL
# ============================================================================
print("\n" + "=" * 70)
print("2. NON-LINEAR SVM - RBF KERNEL")
print("=" * 70)

# Load iris dataset
iris = datasets.load_iris()
X_iris = iris.data[:, :2]  # Use only 2 features for visualization
y_iris = iris.target

# Binary classification (setosa vs versicolor)
X_binary = X_iris[y_iris != 2]
y_binary = y_iris[y_iris != 2]

# Split and scale
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# Train RBF SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
rbf_svm.fit(X_train_iris_scaled, y_train_iris)

# Predictions
y_pred_iris = rbf_svm.predict(X_test_iris_scaled)
print(f"RBF SVM Accuracy: {accuracy_score(y_test_iris, y_pred_iris):.4f}")

# ============================================================================
# 3. MULTI-CLASS SVM
# ============================================================================
print("\n" + "=" * 70)
print("3. MULTI-CLASS SVM (One-vs-Rest)")
print("=" * 70)

# Use full iris dataset (3 classes)
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

scaler_mc = StandardScaler()
X_train_mc_scaled = scaler_mc.fit_transform(X_train_mc)
X_test_mc_scaled = scaler_mc.transform(X_test_mc)

# Multi-class SVM (uses One-vs-Rest by default)
multiclass_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
multiclass_svm.fit(X_train_mc_scaled, y_train_mc)

y_pred_mc = multiclass_svm.predict(X_test_mc_scaled)
print(f"Multi-class Accuracy: {accuracy_score(y_test_mc, y_pred_mc):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_mc, y_pred_mc, target_names=iris.target_names))

# ============================================================================
# 4. HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================
print("\n" + "=" * 70)
print("4. HYPERPARAMETER TUNING")
print("=" * 70)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_mc_scaled, y_train_mc)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")

# Test best model
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_mc_scaled)
print(f"Test Accuracy with Best Model: {accuracy_score(y_test_mc, y_pred_best):.4f}")

# ============================================================================
# 5. DIFFERENT KERNEL COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("5. KERNEL COMPARISON")
print("=" * 70)

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_scores = {}

for kernel in kernels:
    svm_model = SVC(kernel=kernel, C=1.0, random_state=42)
    scores = cross_val_score(svm_model, X_train_mc_scaled, y_train_mc, cv=5)
    kernel_scores[kernel] = scores.mean()
    print(f"{kernel.upper():10s} Kernel - CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# ============================================================================
# 6. SVM FOR REGRESSION (SVR)
# ============================================================================
print("\n" + "=" * 70)
print("6. SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 70)

# Generate regression data
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_reg_scaled, y_train_reg)

# Predictions
y_pred_reg = svr.predict(X_test_reg_scaled)

# Evaluation
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ============================================================================
# 7. PRACTICAL TIPS AND KEY PARAMETERS
# ============================================================================
print("\n" + "=" * 70)
print("7. KEY PARAMETERS AND TIPS")
print("=" * 70)

tips = """
KEY SVM PARAMETERS:
-------------------
1. C (Regularization): Controls trade-off between smooth decision boundary and 
   classifying training points correctly.
   - Small C: More regularization, wider margin, more misclassifications
   - Large C: Less regularization, narrower margin, fewer misclassifications

2. kernel: Specifies kernel type
   - 'linear': For linearly separable data
   - 'rbf': Most common, for non-linear data
   - 'poly': Polynomial kernel
   - 'sigmoid': Similar to neural networks

3. gamma (for RBF, poly, sigmoid): Defines influence of single training example
   - Small gamma: Far reach, smoother decision boundary
   - Large gamma: Close reach, more complex decision boundary

4. degree (for poly kernel): Degree of polynomial

BEST PRACTICES:
---------------
✓ Always scale features (StandardScaler or MinMaxScaler)
✓ Use GridSearchCV for hyperparameter tuning
✓ Start with RBF kernel for non-linear problems
✓ Consider class imbalance with 'class_weight' parameter
✓ Use probability estimates with probability=True if needed
✓ For large datasets, consider LinearSVC for speed

WHEN TO USE SVM:
----------------
✓ Small to medium-sized datasets
✓ High-dimensional spaces
✓ Clear margin of separation
✓ Non-linear decision boundaries (with appropriate kernel)
"""

print(tips)

# ============================================================================
# 8. EXAMPLE: HANDLING IMBALANCED DATA
# ============================================================================
print("\n" + "=" * 70)
print("8. HANDLING IMBALANCED DATA")
print("=" * 70)

# SVM with class weights
balanced_svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
balanced_svm.fit(X_train_mc_scaled, y_train_mc)
y_pred_balanced = balanced_svm.predict(X_test_mc_scaled)

print(f"Balanced SVM Accuracy: {accuracy_score(y_test_mc, y_pred_balanced):.4f}")
print("\nNote: Use class_weight='balanced' for imbalanced datasets")

print("\n" + "=" * 70)
print("SVM TUTORIAL COMPLETE!")
print("=" * 70)