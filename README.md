# EX 08 Implementation of K Means Clustering for Customer Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages
2. Insert the dataset to perform the k - means clustering
3. Perform k - means clustering on the dataset
4. Then print the centroids and labels
5. Plot graph and display the clusters

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
NAME: DEEPIKA S 
RegisterNumber:212222230028 
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X=data[['Annual Income (k$)', 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
              color=colors[i], label=f'Cluster {i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centrois')

plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```
## Output:
### DATASET:
![1MM](https://github.com/deepikasrinivasans/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393935/fcc3de50-465b-4dad-b628-e9ba2cfa0194)
### CENTROID AND LABEL VALUES:
![2MM](https://github.com/deepikasrinivasans/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393935/a340c7f5-4249-440a-9a49-e9a8e7936c04)
### CLUSTERING:
![3MM](https://github.com/deepikasrinivasans/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393935/ba8003cb-bd31-41a7-a8ed-2e9a3e326d2e)
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
