import numpy as np
from collections import Counter

class Node:
    """Represents a node in the decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left subtree
        self.right = right         # Right subtree
        self.value = value         # Value if leaf node

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """Decision Tree Classifier using CART algorithm"""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        """Build decision tree from training data"""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Check minimum samples per leaf
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Recursively build subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calculate information gain for a split"""
        parent_entropy = self._entropy(y)
        
        # Split data
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain
        ig = parent_entropy - child_entropy
        return ig
    
    def _entropy(self, y):
        """Calculate entropy of labels"""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def _most_common_label(self, y):
        """Return most common label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for single sample"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train decision tree
    dt = DecisionTree(max_depth=5, min_samples_split=2)
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Compare with sklearn
    from sklearn.tree import DecisionTreeClassifier
    
    sklearn_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_dt.fit(X_train, y_train)
    sklearn_pred = sklearn_dt.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"\nSklearn Accuracy: {sklearn_accuracy:.4f}")
    print(f"Our Implementation Accuracy: {accuracy:.4f}")