# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # Features (sqft)
# X = np.array([[500],[700],[800],[1200],[1500],[1800]])

# # Target (price in lakhs)
# y = np.array([30,40,45,65,80,95])

# # Train Model
# model = LinearRegression()
# model.fit(X,y)

# print("Slope (Coefficient):", model.coef_)
# print("Intercept:", model.intercept_)
# print("Price prediction for 1000 sqft =", model.predict([[1000]])[0])
# plt.scatter(X,y,label="Actual Data")
# plt.plot(X,model.predict(X),label="Regression Line")
# plt.xlabel("Square Feet")
# plt.ylabel("Price (Lakhs)")
# plt.legend()
# plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'area':[1000,1200,1500,1800,2500],
    'bedrooms':[2,2,3,3,4],
    'age':[5,6,8,3,2],
    'price':[50,58,72,90,140]
}

df = pd.DataFrame(data)

X = df[['area','bedrooms','age']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)

# Predict 1600 sq ft, 3 bhk, 4 years old
print("Predicted Price:",model.predict([[1600,3,4]])[0])
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X)

print("MSE:", mean_squared_error(y,y_pred))
print("R² Score:", r2_score(y,y_pred))   # closer to 1 = better
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Train R²:", model.score(X_train,y_train))
print("Test R² :", model.score(X_test,y_test))
