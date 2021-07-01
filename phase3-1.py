import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("divar_posts_dataset.csv")

dataset = dataset[['city', 'cat1', 'cat2', 'cat3']]
dataset.dropna(inplace=True)
group = dataset.groupby(['city', 'cat1', 'cat2', 'cat3'])
dataset = group.size().to_frame(name='count').reset_index()
for index, row in dataset.iterrows():
    print(index, row)
input()

for column in dataset.columns:
    label_encoder = LabelEncoder()
    dataset[column] = label_encoder.fit_transform(dataset[column])

true_labels = dataset['city']
dataset = dataset.drop('city', axis=1)
dataset = np.array(dataset.astype(float))
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
true_labels = np.array(true_labels)

print("Clustering...")

kmeans = KMeans(n_clusters=9)
kmeans.fit(dataset)

print("Calculating the clustering accuraacy...")

correct = 0
for i in range(len(dataset)):
    if i % 1000 == 0:
        print(i)
    prediction = kmeans.predict(dataset[i].reshape(-1, len(dataset[i])))
    if prediction[0] == true_labels[i]:
        correct += 1

print("Accuracy:", correct/len(dataset))
