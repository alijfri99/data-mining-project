import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

dataset = np.load('digikala_clustering.npy')

print("Clustering...")

kmeans_results = dict()

for i in range(2, 9):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(dataset)
    kmeans_results[i] = silhouette_score(dataset, kmeans.labels_, metric='euclidean')

plt.figure()
plt.plot(list(kmeans_results.keys()), list(kmeans_results.values()))
plt.show()

k = int(input("Which k yielded the best silhouette score? "))

kmeans = KMeans(n_clusters=k)
kmeans.fit(dataset)
print(kmeans.labels_)
print(silhouette_score(dataset, kmeans.labels_, metric='euclidean'))
print("*****")

agglomerative = AgglomerativeClustering()
agglomerative.fit(dataset)
print(agglomerative.labels_)
print(silhouette_score(dataset, agglomerative.labels_, metric='euclidean'))
print("*****")
