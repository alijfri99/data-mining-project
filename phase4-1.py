import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('divar_posts_dataset.csv')
dataset = dataset[['brand', 'cat1', 'cat2', 'cat3', 'price']]
dataset = dataset.loc[dataset['price'] != -1]
dataset.dropna(inplace=True)

dataset['brand'].replace({'سایر': 'Others', 'غیره': 'Others', 'پراید هاچ\u200cبک::Pride': 'Pride 1',
                          'پراید صندوق\u200cدار::Pride': 'Pride 2', 'پژو ۲۰۶\u200d::Peugeot 206': 'Peugeot 206 1',
                          'پژو ۲۰۶\u200d صندوق\u200cدار::Peugeot 206': 'Peugeot 206 2'}, inplace=True)

dataset['brand'] = dataset.apply(lambda x: re.sub('[^a-zA-Z0-9 /]+', '', x['brand']), axis=1)

dataset = dataset.loc[dataset['brand'] != '']

dataset['brand'].replace({' Peugeot Pars': 'Peugeot Pars', '  / RD/ROA': 'RD/ROA', ' Hyundai': 'Hyundai',
                          ' Tondar 90': 'Tondar 90', 'Sony Ericsson ': 'Sony Ericsson',
                          ' Hyundai Sonata': 'Hyundai Sonata', ' Peugeot 405': 'Peugeot 405'}, inplace=True)

dataset = pd.get_dummies(dataset, columns=['brand', 'cat1', 'cat2', 'cat3'])
dataset['temp'] = 1
dataset[['price', 'temp']] = MinMaxScaler().fit_transform(dataset[['price', 'temp']])
dataset = dataset.drop('temp', axis=1)

y = dataset['price']
X = dataset.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree = DecisionTreeRegressor(max_depth=len(X_train.columns), min_samples_leaf=0.1, random_state=3)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(mean_squared_error(y_pred, y_test))
