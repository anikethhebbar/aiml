# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
print("IRIS DATASET LOADED")

# Print the target names (classes) in the Iris dataset
print("\n IRIS Features", iris.target_names)

# Split the dataset into training and testing data (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)
print("Dataset is split into training and test data")

# Print the size of the training and testing datasets
print("Size of training data and its label", X_train.shape, y_train.shape)
print("Size of testing data and its label", X_test.shape, y_test.shape)

# Create a K-Nearest Neighbors classifier with k=1
kn = KNeighborsClassifier(n_neighbors=1)

# Fit the K-Nearest Neighbors classifier to the training data
kn.fit(X_train, y_train)

# Loop through the testing data and make predictions for each sample
for i in range(len(X_test)):
    X = X_test[i]
    X_new = np.array([X])
    prediction = kn.predict(X_new)
    
    # Print the true target, the predicted target, and their corresponding class names
    print("TARGET =", y_test[i], iris["target_names"][y_test[i]], "PREDICTED =", prediction, iris["target_names"][prediction])
    
# Calculate and print the accuracy of the K-Nearest Neighbors classifier on the testing data
print(kn.score(X_test, y_test))
