import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

dataset = pd.read_csv("divar_posts_dataset.csv")

dataset = dataset[['city', 'cat1', 'cat2', 'cat3']]
dataset.dropna(inplace=True)
group = dataset.groupby(['city', 'cat1', 'cat2', 'cat3'])
dataset = group.size().to_frame(name='count').reset_index()
dataset['cat'] = dataset['cat1'] + '-' + dataset['cat2'] + '-' + dataset['cat3']
dataset = dataset[['city', 'cat', 'count']]

print("Creating the dictionary...")

my_dict = dict()

for city in dataset['city'].unique():
    my_dict[city] = dict()

    for cat in dataset['cat'].unique():
        if dataset[(dataset['city'] == city) & (dataset['cat'] == cat)].empty:
            my_dict[city][cat] = 0
        else:
            my_dict[city][cat] = dataset.loc[(dataset['city'] == city) & (dataset['cat'] == cat)]['count'].values[0]

# Converting the dictionary to a list of lists
dataset = []
for city in my_dict.keys():
    dataset.append([cat_count for cat_count in my_dict[city].values()])

dataset = np.array(dataset).astype(float)
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

print("Clustering...")

kmeans_results = dict()

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataset)
    kmeans_results[k] = silhouette_score(dataset, kmeans.labels_, metric='euclidean')

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

scan = DBSCAN(eps=3, min_samples=2)
scan.fit(dataset)
print(scan.labels_)
print(silhouette_score(dataset, scan.labels_, metric='euclidean'))
print("*****")

scan2 = DBSCAN(eps=1.5, min_samples=2)
scan2.fit(dataset)
print(scan2.labels_)
print(silhouette_score(dataset, scan2.labels_, metric='euclidean'))
print("*****")

scan3 = DBSCAN(eps=0.8, min_samples=3)
scan3.fit(dataset)
print(scan3.labels_)
print(silhouette_score(dataset, scan3.labels_, metric='euclidean'))
print("*****")

results = dict()
for label in set(kmeans.labels_):
    results[label] = list()

i = 0
for key in my_dict.keys():
    prediction = kmeans.predict(dataset[i].reshape(-1, len(dataset[i])))[0]
    results[prediction].append(key)
    i += 1

print(results)
