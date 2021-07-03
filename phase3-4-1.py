import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv("divar_posts_dataset.csv")
dataset = dataset[['cat1', 'cat2', 'cat3', 'price']]
dataset.dropna(inplace=True)
dataset = dataset[dataset['price'] != -1]
group = dataset.groupby(['cat1', 'cat2', 'cat3', 'price'])
dataset = group.size().to_frame(name='count').reset_index()
dataset['cat'] = dataset['cat1'] + '-' + dataset['cat2'] + '-' + dataset['cat3']
dataset = dataset[['cat', 'price', 'count']]

print("Creating the dictionary...")
my_dict = dict()
i = 0

for cat in dataset['cat'].unique():
    my_dict[cat] = dict()

    for price in dataset['price'].unique():
        if dataset[(dataset['cat'] == cat) & (dataset['price'] == price)].empty:
            my_dict[cat][price] = 0
        else:
            my_dict[cat][price] = dataset.loc[(dataset['cat'] == cat) & (dataset['price'] == price)]['count'].values[0]

        i += 1
        if i % 1000 == 0:
            print(i, len(dataset['cat'].unique()) * len(dataset['price'].unique()))

dict_output = open('divar_clustering_dict.pkl', 'wb')
pickle.dump(my_dict, dict_output)
dict_output.close()
print(my_dict)

# Converting the dictionary to a list of lists
dataset = []
for cat in my_dict.keys():
    dataset.append([price_count for price_count in my_dict[cat].values()])

dataset = np.array(dataset).astype(float)
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
np.save('divar_clustering.npy', dataset)
print(dataset)
