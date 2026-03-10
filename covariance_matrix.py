import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Data MUST be defined first
X = np.array([
    [2.5, 2.4, 3.5, 1.2, 4.1],
    [0.5, 0.7, 1.2, 0.9, 2.3],
    [2.2, 2.9, 3.1, 1.8, 3.8],
    [1.9, 2.2, 2.8, 1.5, 3.5],
    [3.1, 3.0, 4.0, 2.1, 4.5],
    [2.3, 2.7, 3.3, 1.7, 3.9]
])

# Sklearn PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='steelblue', s=100, edgecolors='black')
for i, (x, y) in enumerate(X_pca):
    plt.annotate(f'P{i+1}', (x, y), textcoords="offset points", xytext=(8, 4))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - sklearn')
plt.grid(True)
plt.tight_layout()
plt.show()