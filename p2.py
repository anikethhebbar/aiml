#p2
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np 
from sklearn.mixture import GaussianMixture

iris=datasets.load_iris()
X=pd.DataFrame(iris.data)
y=pd.DataFrame(iris.target)
X.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
y.columns=['Targets']
model=KMeans(n_clusters=3).fit(X)
plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])

plt.subplot(1,3,1)
plt.scatter(X.PetalLengthCm,X.PetalWidthCm,c=colormap[y.Targets],s=40)
plt.title('Real Clusters')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.subplot(1,3,2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm,c=colormap[model.labels_],s=40)
plt.title('K Means Clustering')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')

gmm=GaussianMixture(n_components=3, random_state=0).fit(X)
y_pred=gmm.predict(X)
plt.subplot(1,3,3)
plt.title('GMM Clustering')
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.scatter(X.PetalLengthCm, X.PetalWidthCm,c=colormap[y_pred])
print("observation: The GMM using EM algorithm based clustering matched the true labels more closely then KMeans")