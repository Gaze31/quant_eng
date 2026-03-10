from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

iris = load_iris()

X = iris.data[:,2:].copy()      # Petal length & width
y = (iris.target == 2).astype(int)  # Virginica = 1

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print("\nClassification Report:\n", classification_report(y_test,pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,pred))
print("Predicted probability:", model.predict_proba([[5.5, 2.0]]))
print("Class:", model.predict([[5.5, 2.0]]))
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

probs = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr,tpr,label=f"AUC = {roc_auc_score(y_test,probs):.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
