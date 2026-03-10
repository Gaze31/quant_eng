from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# labels = kmeans.labels_

# plt.scatter(X[:,0], X[:,1], c=labels)
# plt.title("K-Means Clustering")
# plt.show()
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "Accuracy:", model.score(X_test, y_test))
