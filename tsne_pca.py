# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

# ---------------------------
# 1. Load and prepare data
# ---------------------------
iris = load_iris()
X = iris.data          # Features (150, 4)
y = iris.target        # Labels (0,1,2)
target_names = iris.target_names

# Standardize features (important for PCA and t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 2. Apply PCA
# ---------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance: {explained_var[0]:.2f} + {explained_var[1]:.2f} = {sum(explained_var):.2f}")

# ---------------------------
# 3. Apply t-SNE
# ---------------------------
# t-SNE is stochastic; set random_state for reproducibility
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# ---------------------------
# 4. Plotting function
# ---------------------------
def plot_2d_embedding(X_2d, y, title, ax):
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # Add legend
    handles, _ = scatter.legend_elements()
    ax.legend(handles, target_names, title='Species')
    return ax

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_2d_embedding(X_pca, y, 'PCA (2D projection)', ax1)
plot_2d_embedding(X_tsne, y, 't-SNE (2D projection)', ax2)
plt.tight_layout()
plt.show()