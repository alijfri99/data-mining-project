import pandas as pd
import re

dataset = pd.read_csv('divar_posts_dataset.csv')
dataset = dataset[['brand', 'cat1', 'cat2', 'cat3', 'price']]
dataset = dataset[dataset['price'] != -1]
dataset.dropna(inplace=True)
dataset['brand'].replace({'سایر': 'Others', 'غیره': 'Others', 'پراید هاچ\u200cبک::Pride': 'Pride 1',
                          'پراید صندوق\u200cدار::Pride': 'Pride 2', 'پژو ۲۰۶\u200d::Peugeot 206': 'Peugeot 206 1',
                          'پژو ۲۰۶\u200d صندوق\u200cدار::Peugeot 206': 'Peugeot 206 2'}, inplace=True)

dataset['position'] = dataset['brand'].str.find(':')
dataset['brand'] = dataset.apply(lambda x: re.sub('[^a-zA-Z0-9 /]+', '', x['brand']), axis=1)

print(dataset['brand'].unique())
