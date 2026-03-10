import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansFromScratch:
    """K-Means clustering algorithm implemented from scratch"""
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        """Fit K-Means to the data"""
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)
            
            # Store old centroids to check convergence
            old_centroids = self.centroids.copy()
            
            # Update centroids
            self.centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        self.labels_ = labels
        return self
    
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid"""
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Keep old centroid if no points assigned
                new_centroids[k] = self.centroids[k]
        return new_centroids
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self._assign_clusters(X)


def demonstrate_kmeans():
    """Demonstrate K-Means clustering with visualizations"""
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, 
                           cluster_std=0.60, random_state=42)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('K-Means Clustering Demonstration', fontsize=16, fontweight='bold')
    
    # 1. Original data
    axes[0, 0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # 2. K-Means with k=2
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    labels_2 = kmeans_2.fit_predict(X)
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=labels_2, cmap='viridis', alpha=0.6, s=50)
    axes[0, 1].scatter(kmeans_2.cluster_centers_[:, 0], 
                       kmeans_2.cluster_centers_[:, 1],
                       c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0, 1].set_title(f'K=2 (Silhouette: {silhouette_score(X, labels_2):.3f})')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # 3. K-Means with k=3
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    labels_3 = kmeans_3.fit_predict(X)
    axes[0, 2].scatter(X[:, 0], X[:, 1], c=labels_3, cmap='viridis', alpha=0.6, s=50)
    axes[0, 2].scatter(kmeans_3.cluster_centers_[:, 0], 
                       kmeans_3.cluster_centers_[:, 1],
                       c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0, 2].set_title(f'K=3 (Silhouette: {silhouette_score(X, labels_3):.3f})')
    axes[0, 2].set_xlabel('Feature 1')
    axes[0, 2].set_ylabel('Feature 2')
    
    # 4. K-Means with k=4 (optimal)
    kmeans_4 = KMeans(n_clusters=4, random_state=42)
    labels_4 = kmeans_4.fit_predict(X)
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=labels_4, cmap='viridis', alpha=0.6, s=50)
    axes[1, 0].scatter(kmeans_4.cluster_centers_[:, 0], 
                       kmeans_4.cluster_centers_[:, 1],
                       c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[1, 0].set_title(f'K=4 (Silhouette: {silhouette_score(X, labels_4):.3f})')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # 5. K-Means with k=5
    kmeans_5 = KMeans(n_clusters=5, random_state=42)
    labels_5 = kmeans_5.fit_predict(X)
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=labels_5, cmap='viridis', alpha=0.6, s=50)
    axes[1, 1].scatter(kmeans_5.cluster_centers_[:, 0], 
                       kmeans_5.cluster_centers_[:, 1],
                       c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[1, 1].set_title(f'K=5 (Silhouette: {silhouette_score(X, labels_5):.3f})')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    
    # 6. Elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    ax2 = axes[1, 2]
    ax2.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_title('Elbow Method for Optimal K')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(K_range, silhouette_scores, 'ro--', linewidth=2, markersize=8)
    ax2_twin.set_ylabel('Silhouette Score', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING RESULTS")
    print("="*60)
    print(f"\nOptimal number of clusters: 4")
    print(f"Silhouette Score (k=4): {silhouette_score(X, labels_4):.4f}")
    print(f"\nCluster sizes:")
    for i in range(4):
        print(f"  Cluster {i}: {np.sum(labels_4 == i)} points")


def compare_implementations():
    """Compare custom K-Means with scikit-learn"""
    
    # Generate data
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    
    # Custom implementation
    custom_kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
    custom_kmeans.fit(X)
    
    # Scikit-learn implementation
    sklearn_kmeans = KMeans(n_clusters=3, random_state=42)
    sklearn_kmeans.fit(X)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_, 
                    cmap='viridis', alpha=0.6, s=50)
    axes[0].scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1],
                    c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_title('Custom K-Means Implementation')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    axes[1].scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_, 
                    cmap='viridis', alpha=0.6, s=50)
    axes[1].scatter(sklearn_kmeans.cluster_centers_[:, 0], 
                    sklearn_kmeans.cluster_centers_[:, 1],
                    c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[1].set_title('Scikit-learn K-Means')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION COMPARISON")
    print("="*60)
    print(f"\nCustom Implementation:")
    print(f"  Silhouette Score: {silhouette_score(X, custom_kmeans.labels_):.4f}")
    print(f"\nScikit-learn Implementation:")
    print(f"  Silhouette Score: {silhouette_score(X, sklearn_kmeans.labels_):.4f}")
    print(f"  Inertia: {sklearn_kmeans.inertia_:.2f}")


if __name__ == "__main__":
    print("Running K-Means Clustering Demonstrations...\n")
    
    # Run demonstrations
    demonstrate_kmeans()
    print("\n")
    compare_implementations()
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60)