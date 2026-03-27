"""
Feature Extraction Project: Digit Recognition
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import cv2
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Load and Explore Dataset
# ---------------------------
print("Loading digits dataset...")
digits = load_digits()
X_raw = digits.images
y = digits.target

print(f"Dataset size: {X_raw.shape[0]} images, each {X_raw.shape[1]}x{X_raw.shape[2]} pixels")
print(f"Number of classes: {len(np.unique(y))}")

# Visualize a few samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_raw[i], cmap='gray')
    ax.set_title(f"Digit: {y[i]}")
    ax.axis('off')
plt.suptitle("Sample Images from Digits Dataset")
plt.tight_layout()
plt.show()

# ---------------------------
# 2. Feature Extraction Functions
# ---------------------------
def extract_raw_pixels(images):
    return images.reshape(images.shape[0], -1)

def extract_hog(images):
    hog_features = []
    for img in images:
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False, feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def extract_lbp(images):
    lbp_features = []
    radius = 1
    n_points = 8 * radius
    for img in images:
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # Scale from [0,16] to [0,255]
        img_uint8 = (img_resized * (255.0 / 16.0)).astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        lbp_features.append(hist)
    return np.array(lbp_features)

# ---------------------------
# 3. Extract Features
# ---------------------------
print("\nExtracting features...")
X_raw_flat = extract_raw_pixels(X_raw)
print(f"Raw pixels: {X_raw_flat.shape}")

X_hog = extract_hog(X_raw)
print(f"HOG features: {X_hog.shape}")

X_lbp = extract_lbp(X_raw)
print(f"LBP features: {X_lbp.shape}")

# ---------------------------
# 4. Train-Test Split & Standardization
# ---------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw_flat, y, test_size=0.2, random_state=42, stratify=y
)
X_train_hog, X_test_hog, _, _ = train_test_split(
    X_hog, y, test_size=0.2, random_state=42, stratify=y
)
X_train_lbp, X_test_lbp, _, _ = train_test_split(
    X_lbp, y, test_size=0.2, random_state=42, stratify=y
)

scaler_raw = StandardScaler()
X_train_raw = scaler_raw.fit_transform(X_train_raw)
X_test_raw = scaler_raw.transform(X_test_raw)

scaler_hog = StandardScaler()
X_train_hog = scaler_hog.fit_transform(X_train_hog)
X_test_hog = scaler_hog.transform(X_test_hog)

scaler_lbp = StandardScaler()
X_train_lbp = scaler_lbp.fit_transform(X_train_lbp)
X_test_lbp = scaler_lbp.transform(X_test_lbp)

# ---------------------------
# 5. Train & Evaluate
# ---------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_name):
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{feature_name} - Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return acc

print("\n=== Model Evaluation ===")
acc_raw = train_and_evaluate(X_train_raw, X_test_raw, y_train, y_test, "Raw Pixels")
acc_hog = train_and_evaluate(X_train_hog, X_test_hog, y_train, y_test, "HOG")
acc_lbp = train_and_evaluate(X_train_lbp, X_test_lbp, y_train, y_test, "LBP")

# ---------------------------
# 6. Visualize Comparison
# ---------------------------
methods = ['Raw Pixels', 'HOG', 'LBP']
accuracies = [acc_raw, acc_hog, acc_lbp]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Comparison of Feature Extraction Methods')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')
plt.show()

print("\n✅ Feature extraction comparison complete!")