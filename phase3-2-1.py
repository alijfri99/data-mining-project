import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("orders.csv")
dataset = dataset[['city_name_fa', 'ID_Item']]
dataset = dataset.loc[dataset['city_name_fa'].isin(['تهران', 'مشهد', 'کرج', 'قم', 'اصفهان', 'شیراز', 'تبریز', 'اهواز',
                                                    'کرمانشاه'])]
print(dataset.head())
print(dataset.isna().sum())
group = dataset.groupby(['city_name_fa', 'ID_Item'])
dataset = group.size().to_frame(name='count').reset_index()

print("Creating the dictionary...")
my_dict = dict()
i = 0

for city in dataset['city_name_fa'].unique():
    my_dict[city] = dict()

    for item in dataset['ID_Item'].unique():
        if dataset[(dataset['city_name_fa'] == city) & (dataset['ID_Item'] == item)].empty:
            my_dict[city][item] = 0
        else:
            my_dict[city][item] = dataset.loc[(dataset['city_name_fa'] == city) &
                                              (dataset['ID_Item'] == item)]['count'].values[0]

        i += 1
        if i % 1000 == 0:
            print(i, len(dataset['city_name_fa'].unique()) * len(dataset['ID_Item'].unique()))

dict_output = open('digikala_clustering_dict.pkl', 'wb')
pickle.dump(my_dict, dict_output)
dict_output.close()
print(my_dict)

dataset = []
for city in my_dict.keys():
    dataset.append([item_count for item_count in my_dict[city].values()])

dataset = np.array(dataset).astype(float)
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
np.save('digikala_clustering.npy', dataset)
print(dataset)
