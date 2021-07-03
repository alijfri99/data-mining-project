import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

dataset = np.load('digikala_clustering.npy')
dataset_dict_file = open('digikala_clustering_dict.pkl', 'rb')
dataset_dict = pickle.load(dataset_dict_file)

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

scan = DBSCAN(eps=100, min_samples=2)
scan.fit(dataset)
print(scan.labels_)
print(silhouette_score(dataset, scan.labels_, metric='euclidean'))
print("*****")

scan2 = DBSCAN(eps=150, min_samples=2)
scan2.fit(dataset)
print(scan2.labels_)
print(silhouette_score(dataset, scan2.labels_, metric='euclidean'))
print("*****")

scan3 = DBSCAN(eps=220, min_samples=3)
scan3.fit(dataset)
print(scan3.labels_)
print(silhouette_score(dataset, scan3.labels_, metric='euclidean'))
print("*****")


results = dict()
for label in set(kmeans.labels_):
    results[label] = list()

i = 0
for key in dataset_dict.keys():
    prediction = kmeans.predict(dataset[i].reshape(-1, len(dataset[i])))[0]
    results[prediction].append(key)
    i += 1

print(results)
