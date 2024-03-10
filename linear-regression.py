from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# Loading dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Spliting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f'Linear Regression Mean Squared Error: {mean_squared_error(y_test, y_pred_lr)}')

# Stochastic Gradient Descent 
sgd = SGDRegressor()
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print(f'SGD Mean Squared Error: {mean_squared_error(y_test, y_pred_sgd)}')


# Visualizing the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_lr)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Add line y=x
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Linear Regression")

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_sgd)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)  # Add line y=x
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("SGD")

plt.tight_layout()
plt.show()