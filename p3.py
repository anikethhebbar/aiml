import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, X, k):
    weights = np.exp(np.sum((X - point) ** 2, axis=1) / (-2.0 * k**2))
    return np.diag(weights)

def localWeight(point, X, y, k):
    weights = kernel(point, X, k)
    W = np.linalg.inv(X.T @ (weights @ X)) @ (X.T @ (weights @ y.T))
    return W

def localWeightRegression(X, y, k):
    m = X.shape[0]
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = X[i] @ localWeight(X[i], X, y, k)
    return ypred

# Load data
data = pd.read_csv('pg3.csv')
X = np.column_stack((np.ones(len(data)), data['total_bill']))
y = data['tip'].values

# Set k here
k = 0.5
ypred = localWeightRegression(X, y, k)

# Sort data for plotting
SortIndex = X[:, 1].argsort()
xsort = X[SortIndex][:, 1]

# Plot
plt.scatter(data['total_bill'], data['tip'], color='green')
plt.plot(xsort, ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()
