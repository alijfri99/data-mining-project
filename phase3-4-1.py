import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv("divar_posts_dataset.csv")
dataset = dataset[['cat1', 'cat2', 'cat3', 'price']]
dataset.dropna(inplace=True)
dataset = dataset.loc[dataset['price'] != -1]
group = dataset.groupby(['cat1', 'cat2', 'cat3'])
dataset = group.mean().reset_index()
dataset['cat'] = dataset['cat1'] + '-' + dataset['cat2'] + '-' + dataset['cat3']
dataset = dataset[['cat', 'price']]

print("Creating the dictionary...")
my_dict = dict()

for cat in dataset['cat'].unique():
    my_dict[cat] = dataset.loc[dataset['cat'] == cat]['price'].values[0]

dict_output = open('divar_clustering_dict.pkl', 'wb')
pickle.dump(my_dict, dict_output)
dict_output.close()
print(my_dict)

# Converting the dictionary to a list of lists
dataset = []
for cat in my_dict.keys():
    dataset.append([my_dict[cat]])

dataset = np.array(dataset).astype(float)
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
np.save('divar_clustering.npy', dataset)
print(dataset)
