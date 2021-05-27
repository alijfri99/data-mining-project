import pandas as pd
from apyori import apriori

# question 2

dataset = pd.read_csv("orders.csv")

# preprocessing

dataset = dataset.dropna()
dataset['DateTime_CartFinalize'] = dataset['DateTime_CartFinalize'].str[0:10]
dataset['ID_Order'] = 'Ord' + dataset['ID_Order'].map(str)
dataset['ID_Item'] = 'It' + dataset['ID_Item'].map(str)
dataset = dataset.drop(['ID_Customer', 'Quantity_item'], axis=1)

# now let's construct the transaction dictionary for each Order ID

transaction_dict = dict()

for index, row in dataset.iterrows():
    if index % 10000 == 0:
        print(index)
    if row['ID_Order'] not in transaction_dict:
        transaction_dict[row['ID_Order']] = {row['ID_Item']}
    else:
        transaction_dict[row['ID_Order']] = transaction_dict[row['ID_Order']].union({row['ID_Item']})

for i in transaction_dict:
    if len(transaction_dict[i]) > 1:
        print(i, transaction_dict[i])
